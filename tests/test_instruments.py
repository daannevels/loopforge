import numpy as np
import pytest

from loopforge.analysis.claps import clap_signature, same_clap
from loopforge.analysis.instruments import classify_hits, summarize
from loopforge.analysis.loop import analyze_loop
from loopforge.config import AnalysisParams, GenParams
from loopforge.generator.profile import build_profile
from loopforge.generator.run import generate_one, selection_mask
from loopforge.generator.select import RunMemory
from loopforge.library import FeatureSpace, NearestNeighbors, load_slice_table, update_index
from loopforge.library.slices import recipe_loader
from loopforge.service import exclude_ref_clap_copies, load_ref
from tests.synth import SR, Hit, clap, render_loop, write


def loop(backbeat: str | None, bars: int = 2, extra: str | None = None) -> list[Hit]:
    """Top loop: closed hats on 16ths, open hats on the offbeat 8ths, optional backbeat clap/snare."""
    hits = []
    for b in range(bars):
        for s in range(16):
            st = 16 * b + s
            if backbeat and s in (4, 12):
                hits.append(Hit(st, backbeat, 1.0))
            elif s % 4 == 2:
                hits.append(Hit(st, "open", 0.7))
            elif extra and s in (7, 15):
                hits.append(Hit(st, extra, 0.6))
            else:
                hits.append(Hit(st, "closed", 0.45))
    return hits


def scan(hits, bpm=128.0, seed=1):
    a = analyze_loop(render_loop(hits, bpm, 2, seed=seed), SR, name="t_128bpm.wav")
    return a, summarize(s.role for s in a.slices)


def test_scan_hats_only():
    a, c = scan(loop(None))
    assert set(c) == {"closed_hat", "open_hat"}


def test_scan_snare_and_perc():
    _, s = scan(loop("snare"))
    assert s.get("snare") == 4 and "clap" not in s
    _, c = scan(loop(None, extra="perc"))
    assert c.get("perc") == 4


def _vec(**kw):
    from loopforge.models import FEATURE_INDEX as FI, N_FEATURES

    v = np.zeros(N_FEATURES, np.float32)
    base = dict(rms_db=-25.0, band_low=0.0, band_mid=0.15, band_high=0.85, centroid_hz=7000.0, attack_ms=3.0,
                decay_ms=60.0, tonal_db=8.0)
    for k, val in {**base, **kw}.items():
        v[FI[k]] = val
    return v


def test_classify_rules_on_measured_values():
    """Feature values as measured on the user's refs (see CLAUDE.md): one bar, steps 0..15."""
    hats = {s: _vec() for s in range(16)}
    hats[2] = hats[6] = hats[10] = hats[14] = _vec(decay_ms=210.0, rms_db=-19.0)            # open hats
    hats[9] = _vec(attack_ms=25.0, decay_ms=55.0)                                              # shaker
    hats[0] = _vec(band_mid=0.97, band_high=0.03, centroid_hz=774.0, tonal_db=30.0)           # perc
    hats[1] = _vec(band_low=0.6, band_mid=0.4, band_high=0.0, centroid_hz=150.0)              # low -> other
    clap = dict(hats)
    clap[4] = clap[12] = _vec(band_mid=0.83, band_high=0.17, centroid_hz=2500.0, attack_ms=35.5, rms_db=-13.9)
    clap[5] = _vec(band_mid=0.86, band_high=0.14, centroid_hz=2160.0, attack_ms=7.5, rms_db=-25.5)  # its tail
    snare = dict(hats)
    snare[4] = snare[12] = _vec(band_mid=0.78, band_high=0.22, centroid_hz=3115.0, attack_ms=2.0, rms_db=-18.0)
    snare[7] = _vec(band_mid=0.60, band_high=0.40, centroid_hz=4000.0, attack_ms=12.5, rms_db=-22.0)  # ghost
    steps = np.arange(16)
    lab = classify_hits(np.stack([clap[s] for s in steps]), steps)
    assert lab[4] == lab[12] == "clap" and lab[5] == "tail"
    assert lab[2] == "open_hat" and lab[3] == "closed_hat" and lab[9] == "shaker"
    assert lab[0] == "perc" and lab[1] == "other"
    lab = classify_hits(np.stack([snare[s] for s in steps]), steps)
    assert lab[4] == lab[12] == "snare" and lab[7] == "snare"  # ghost note takes the loop's choice
    assert "clap" not in set(lab)


def test_same_clap_signature():
    a, b, c = clap(seed=1), clap(seed=2), clap(seed=1, lo=2500.0, hi=6000.0)
    sa, sb, sc = (clap_signature(x, SR) for x in (a, b, c))
    assert same_clap(sa, sb)  # same clap "model", different noise -> same sound
    assert not same_clap(sa, sc)  # a brighter, different clap


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("inst")
    lib = root / "lib"
    import tests.synth as sy

    def variant(name, hits, voice=None, band=None):
        orig = sy.VOICES.get(voice)
        if band:
            sy.VOICES[voice] = lambda sr=SR, seed=5, o=orig: o(sr=sr, seed=seed, lo=500.0, hi=2200.0)
        write(lib / f"{name}.wav", render_loop(hits, 124.0, 2, seed=len(name)))
        if band:
            sy.VOICES[voice] = orig

    variant("hats_a", loop(None))
    variant("hats_b", loop(None, extra="shaker"))
    variant("snare_pack", loop("snare"))  # the same snare as the ref (sample pack)
    variant("snare_other", loop("snare"), "snare", "bright")  # a clearly different snare
    variant("perc_loop", loop(None, extra="perc"))
    refs = {k: write(root / f"ref_{k}_128bpm.wav", render_loop(h, 128.0, 2, seed=3))
            for k, h in {"hats": loop(None), "snare": loop("snare")}.items()}
    update_index(lib, jobs=1)
    t = load_slice_table(lib)
    gp = GenParams(bars=2)
    space = FeatureSpace.fit(t.features, group_weights=dict(gp.timbre_group_weights))
    return {"lib": lib, "refs": refs, "t": t, "gp": gp, "space": space, "nn": NearestNeighbors(space, t.features)}


def run(world, key, n=3):
    t, gp = world["t"], world["gp"]
    ref = load_ref(world["refs"][key], AnalysisParams())
    prof = build_profile([ref.analysis], [ref.rms], world["space"], gp)
    ex = selection_mask(t, gp, {ref.sha1})
    exclude_ref_clap_copies([ref], t, ex, threshold=gp.clap_copy_threshold)
    mem = RunMemory.new()
    label = {(t.files[int(t.file_idx[i])]["rel"], int(t.slice_idx[i])): t.role[i] for i in range(len(t))}
    recipes = [generate_one(prof, t, world["nn"], gp, s, ex, recipe_loader(), memory=mem)[0] for s in range(n)]
    return prof, recipes, label


def test_hats_only_ref_gets_only_hats(world):
    prof, recipes, label = run(world, "hats")
    assert {r.instrument for r in prof.roles} == {"closed_hat", "open_hat"}
    for rec in recipes:
        for e in rec.events:
            assert label[(e.source_rel, e.slice_index)] == e.role  # same instrument as the ref hit
            assert e.role in ("closed_hat", "open_hat")


def test_snare_replaced_by_a_different_snare(world):
    prof, recipes, label = run(world, "snare")
    assert "snare" in {r.instrument for r in prof.roles}
    for rec in recipes:
        sn = [e for e in rec.events if e.role == "snare"]
        assert sn and all(label[(e.source_rel, e.slice_index)] == "snare" for e in sn)
        assert all(e.source_rel == "snare_other.wav" for e in sn)  # the pack copy of the ref snare is skipped
        others = [e for e in rec.events if e.role != "snare"]
        assert all(label[(e.source_rel, e.slice_index)] == e.role for e in others)
