import json

import numpy as np
import pytest

from loopforge.analysis.loop import analyze_loop
from loopforge.audio_io import file_sha1, load_audio
from loopforge.cli import main
from loopforge.config import GenParams
from loopforge.diagnostics import seam_ratio
from loopforge.generator.pattern import generate_pattern
from loopforge.generator.profile import build_profile
from loopforge.generator.recipe import Recipe, RecipeEvent, mix_events, render_recipe, soft_limit
from loopforge.generator.run import generate_one, loop_length_samples, selection_mask
from loopforge.library import FeatureSpace, NearestNeighbors, load_slice_table, update_index
from loopforge.library.slices import recipe_loader
from tests.synth import SR, Hit, render_loop, sparse_pattern, write

BPM = 128.0


def ref_pattern(bars: int, variant: int = 0) -> list[Hit]:
    hits = []
    for b in range(bars):
        o = 16 * b
        for s in range(16):
            if s % 4 == 2:
                hits.append(Hit(o + s, "open", 0.9))
            elif s in (7, 15) and variant == 0:
                hits.append(Hit(o + s, "shaker", 0.6))
            elif s in (5, 13) and variant == 1:
                hits.append(Hit(o + s, "shaker", 0.6))
            elif s % 2 == 0:
                hits.append(Hit(o + s, "closed", 1.0 if s % 4 == 0 else 0.6))
    return hits


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("gen")
    lib = root / "lib"
    # library: several loops per voice, different seeds/tempi/sample rates
    for i, (voice, steps) in enumerate([("closed", range(16)), ("closed", range(0, 16, 2)), ("open", [2, 6, 10, 14]),
                                        ("open", [2, 10]), ("shaker", range(0, 16, 2)), ("shaker", [3, 7, 11, 15]),
                                        ("perc", [0, 6, 12])]):
        sr = 44100 if i % 3 == 0 else SR
        write(lib / f"{voice}_{i}.wav",
              render_loop(sparse_pattern(list(steps), voice, 2), 120.0 + i, 2, sr=sr, seed=10 + i), sr=sr)
    refs = [write(root / "refs" / f"ref{v}_128bpm.wav", render_loop(ref_pattern(4, v), BPM, 4, seed=99 + v))
            for v in (0, 1)]
    update_index(lib, jobs=1)
    table = load_slice_table(lib)
    gp = GenParams(bars=2)
    space = FeatureSpace.fit(table.features, group_weights=dict(gp.timbre_group_weights))
    nn = NearestNeighbors(space, table.features)
    analyses, rms = [], []
    for r in refs:
        x, sr = load_audio(r)
        analyses.append(analyze_loop(x, sr, name=r.name))
        rms.append(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))))
    profile = build_profile(analyses, rms, space, gp)
    exclude = selection_mask(table, gp, {file_sha1(r) for r in refs})
    return dict(root=root, lib=lib, refs=refs, table=table, space=space, nn=nn, profile=profile, gp=gp,
                exclude=exclude)


def gen(world, seed=1, **kw):
    gp = GenParams(**{**world["gp"].to_dict(), **kw})
    return generate_one(world["profile"], world["table"], world["nn"], gp, seed, world["exclude"], recipe_loader())


def test_profile(world):
    pr = world["profile"]
    assert pr.bpm == BPM and pr.n_steps == 32 and pr.sr == SR
    assert len(pr.templates) == 4  # 2 refs x (4 bars / 2 bars)
    assert len(pr.roles) >= 2
    assert pr.onset_prob[2] == 1.0 and pr.onset_prob[1] == 0.0  # open on every 2nd 16th of each beat, never on .2
    long_roles = [r for r in pr.roles if r.long]
    assert long_roles, "open hats should form a ringing role"


def test_similarity_one_keeps_template_rhythm(world):
    pr = world["profile"]
    for seed in range(5):
        rng = np.random.default_rng(seed)
        events, name = generate_pattern(pr, 1.0, rng, world["gp"])
        t = pr.templates[pr.template_names.index(name)]
        assert sorted({e.step for e in events}) == sorted({e.step for e in t})


def test_low_similarity_mutates(world):
    pr = world["profile"]
    diffs = []
    for seed in range(10):
        events, name = generate_pattern(pr, 0.0, np.random.default_rng(seed), world["gp"])
        t = {e.step for e in pr.templates[pr.template_names.index(name)]}
        diffs.append(len({e.step for e in events} ^ t))
    assert np.mean(diffs) > 2


def test_exact_length_seamless_and_in_tempo(world):
    recipe, audio = gen(world, seed=3)
    n = loop_length_samples(BPM, 2, SR)
    assert audio.shape == (2, n) and recipe.n_samples == n
    assert seam_ratio(audio) < 1.0
    tiled = np.tile(audio, (1, 2))
    d = np.abs(np.diff(tiled, axis=1))
    assert d[:, n - 1].max() <= np.percentile(d, 99.9)  # junction no rougher than the audio itself
    a = analyze_loop(audio, SR, name="out.wav")
    assert a.tempo.is_loop and abs(a.tempo.bpm - BPM) < 0.05 and a.tempo.bars == 2
    assert np.abs(audio).max() <= 10 ** (-1 / 20) + 1e-6


def test_deterministic_and_seed_dependent(world):
    r1, a1 = gen(world, seed=7)
    r2, a2 = gen(world, seed=7)
    _, a3 = gen(world, seed=8)
    assert np.array_equal(a1, a2)
    assert r1.to_dict() == r2.to_dict()
    assert not np.array_equal(a1, a3)


def test_constraints(world):
    for seed in range(6):
        recipe, _ = gen(world, seed=seed)
        evs = sorted(recipe.events, key=lambda e: e.position)
        ids = [(e.source_rel, e.slice_index) for e in evs]
        assert all(a != b for a, b in zip(ids, ids[1:])), "same slice back-to-back"
        cap = int(np.floor(0.4 * len(evs)))
        from collections import Counter

        assert max(Counter(e.source_rel for e in evs).values()) <= cap
        assert all(not e.source_rel.startswith("ref") for e in evs)
        assert all(abs(e.gain_db) <= 12.0 for e in evs)


def test_refs_inside_library_are_never_used(world, tmp_path):
    import shutil

    lib2 = tmp_path / "lib2"
    shutil.copytree(world["lib"], lib2, ignore=shutil.ignore_patterns(".loopforge"))
    shutil.copy(world["refs"][0], lib2 / "ref_copy.wav")
    update_index(lib2, jobs=1)
    t2 = load_slice_table(lib2)
    nn2 = NearestNeighbors(world["space"], t2.features)
    ex = selection_mask(t2, world["gp"], {file_sha1(r) for r in world["refs"]})
    for seed in range(4):
        recipe, _ = generate_one(world["profile"], t2, nn2, world["gp"], seed, ex, recipe_loader())
        assert all(e.source_rel != "ref_copy.wav" for e in recipe.events)


def test_recipe_roundtrip_renders_identically(world, tmp_path):
    recipe, audio = gen(world, seed=11)
    p = recipe.save(tmp_path / "r.json")
    r2 = Recipe.load(p)
    assert np.array_equal(render_recipe(r2, recipe_loader()), audio)
    # library moved: resolve by relative path
    assert np.array_equal(render_recipe(r2, recipe_loader(str(world["lib"]))), audio)


def test_swing_shifts_odd_steps(world):
    r0, _ = gen(world, seed=5, swing=0.0)
    r1, _ = gen(world, seed=5, swing=0.25)
    step = 60 / BPM / 4 * SR
    for a, b in zip(sorted(r0.events, key=lambda e: (e.step, e.sub_offset)),
                    sorted(r1.events, key=lambda e: (e.step, e.sub_offset))):
        expect = 0.25 * step if a.step % 2 else 0
        assert abs((b.position - a.position) - expect) <= 1


def test_tail_wraps_into_start():
    sr, n = 1000, 100
    ev = RecipeEvent(step=0, sub_offset=0, role="x", position=90, length=30, source="", source_rel="", source_sha1="",
                     source_sr=sr, src_start=0, src_end=30, slice_index=0, gain_db=0.0, fade_in_ms=0, fade_out_ms=0,
                     target_db=0, slice_peak_db=0)
    rec = Recipe(seed=0, bpm=120, bars=1, sr=sr, n_samples=n, swing=0, similarity=1, params={}, refs=[],
                 library_root="", template="", roles=[], pattern=[], events=[ev])
    out = mix_events(rec, lambda e, s: np.ones((2, 30), np.float32))
    assert out.shape == (2, 100)
    assert np.all(out[:, 90:] == 1) and np.all(out[:, :20] == 1) and np.all(out[:, 20:90] == 0)


def test_soft_limiter_bounds():
    x = np.linspace(-3, 3, 1001, dtype=np.float32)[None, :]
    y = soft_limit(x, -1.0, 0.7)
    c = 10 ** (-1 / 20)
    assert np.abs(y).max() <= c + 1e-6
    small = np.abs(x) < 0.6
    assert np.allclose(y[small], x[small])
    assert np.all(np.diff(y[0]) >= 0)  # monotonic


def test_cli_generate_and_render(world, tmp_path, capsys):
    out = tmp_path / "out"
    args = ["generate", "--refs", *map(str, world["refs"]), "--library", str(world["lib"]), "--bars", "2",
            "--variations", "2", "--seed", "5", "--out", str(out), "--no-update", "--audition"]
    assert main(args) == 0
    assert (out / "report.md").is_file() and (out / "profile.json").is_file()
    wav, js = out / "var_01_seed5.wav", out / "var_01_seed5.json"
    assert wav.is_file() and js.is_file() and (out / "var_01_seed5.png").is_file()
    assert (out / "listen" / "var_01_seed5_x4.wav").is_file() and (out / "listen" / "var_01_seed5_AB.wav").is_file()
    d = json.loads(js.read_text())
    assert d["seed"] == 5 and d["events"] and d["metrics"]["length_exact"]
    assert main(["render", str(js), "--out", str(tmp_path / "re.wav")]) == 0
    assert file_sha1(tmp_path / "re.wav") == file_sha1(wav)
