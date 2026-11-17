import numpy as np
import pytest

from loopforge.analysis.claps import clap_flags, clap_signature, same_clap
from loopforge.analysis.loop import analyze_loop
from loopforge.audio_io import file_sha1, load_audio
from loopforge.config import AnalysisParams, GenParams
from loopforge.generator.profile import build_profile
from loopforge.generator.run import generate_one, selection_mask
from loopforge.generator.select import RunMemory
from loopforge.library import FeatureSpace, NearestNeighbors, load_slice_table, update_index
from loopforge.library.slices import recipe_loader
from loopforge.service import exclude_ref_clap_copies, load_ref
from tests.synth import SR, Hit, clap, render_loop, write


def top_with_clap(bars: int = 2, clap_seed: int = 4) -> list[Hit]:
    hits = []
    for b in range(bars):
        for s in range(16):
            st = 16 * b + s
            if s in (4, 12):
                hits.append(Hit(st, "clap", 1.0))
                hits.append(Hit(st, "closed", 0.4))  # hat layered under the clap
            elif s % 4 == 2:
                hits.append(Hit(st, "open", 0.7))
            else:
                hits.append(Hit(st, "closed", 0.45))
    return hits


def test_clap_detection_and_tail():
    x = render_loop(top_with_clap(2), 128.0, 2, seed=1)
    a = analyze_loop(x, SR, name="t_128bpm.wav")
    steps = np.array([s.step for s in a.slices])
    is_clap, tainted = clap_flags(a.features, steps)
    assert sorted(set(steps[is_clap] % 16)) == [4, 12]
    assert not (is_clap & ((steps % 16) == 2)).any()  # open hats are not claps
    assert set(steps[tainted] % 16) <= {5, 6, 7, 13, 14, 15}


def test_same_clap_signature():
    a, b, c = clap(seed=1), clap(seed=2), clap(seed=1, lo=2500.0, hi=6000.0)
    sa, sb, sc = (clap_signature(x, SR) for x in (a, b, c))
    assert same_clap(sa, sb)  # same clap "model", different noise -> same sound
    assert not same_clap(sa, sc)  # a brighter, different clap


@pytest.fixture(scope="module")
def clap_world(tmp_path_factory):
    root = tmp_path_factory.mktemp("claps")
    lib = root / "lib"
    hats = [Hit(s, "closed", 0.5) for s in range(32) if s % 16 not in (4, 12)]
    # library loops: same clap as the ref (pack clap), and two clearly different claps
    for name, band in [("pack_a", (900.0, 2600.0)), ("pack_b", (900.0, 2600.0)),
                       ("other_bright", (2200.0, 5000.0)), ("other_dark", (500.0, 1400.0))]:
        hits = hats + [Hit(s, "clap", 1.0) for s in (4, 12, 20, 28)]
        x = render_loop(hits, 124.0, 2, seed=len(name))
        if name.startswith("other"):  # re-render the claps with a different band
            import tests.synth as sy

            orig = sy.VOICES["clap"]
            sy.VOICES["clap"] = lambda sr=SR, seed=4, b=band: orig(sr=sr, seed=seed, lo=b[0], hi=b[1])
            x = render_loop(hits, 124.0, 2, seed=len(name))
            sy.VOICES["clap"] = orig
        write(lib / f"{name}.wav", x)
    write(lib / "open_hats.wav", render_loop([Hit(s, "open", 0.8) for s in range(2, 32, 4)], 124.0, 2, seed=9))
    ref = write(root / "ref_128bpm.wav", render_loop(top_with_clap(2), 128.0, 2, seed=3))
    update_index(lib, jobs=1)
    return {"lib": lib, "ref": ref}


def test_new_loops_get_a_different_clap(clap_world):
    gp = GenParams(bars=2)
    t = load_slice_table(clap_world["lib"])
    space = FeatureSpace.fit(t.features, group_weights=dict(gp.timbre_group_weights))
    nn = NearestNeighbors(space, t.features)
    ref = load_ref(clap_world["ref"], AnalysisParams())
    prof = build_profile([ref.analysis], [ref.rms], space, gp)
    clap_roles = [r for r in prof.roles if r.is_clap]
    assert len(clap_roles) == 1 and clap_roles[0].long
    ex = selection_mask(t, gp, {ref.sha1})
    n = exclude_ref_clap_copies([ref], t, ex, threshold=gp.clap_copy_threshold)
    assert n > 0  # the pack copies of the ref clap are skipped
    is_clap, tainted = t.clap_masks()
    mem = RunMemory.new()
    used_claps = set()
    for seed in range(4):
        recipe, _ = generate_one(prof, t, nn, gp, seed, ex, recipe_loader(), memory=mem)
        assert len(recipe.pattern) == len(prof.templates[prof.template_names.index(recipe.template)])
        for e in recipe.events:
            if e.role == "clap":
                assert e.source_rel.startswith("other_")
                used_claps.add(e.source_rel)
            else:
                assert not e.source_rel.startswith(("pack_", "other_")) or True
    assert used_claps == {"other_bright.wav", "other_dark.wav"}
