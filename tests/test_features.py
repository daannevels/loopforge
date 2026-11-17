import numpy as np

from loopforge.analysis.features import attack_decay_ms, describe_role, slice_features
from loopforge.config import AnalysisParams
from loopforge.models import FEATURE_INDEX as FI
from loopforge.models import N_FEATURES
from tests.synth import SR, closed_hat, open_hat, perc, shaker

P = AnalysisParams()


def feats(x, sr=SR):
    return slice_features(x, sr, P)


def test_vector_shape_and_finite():
    f = feats(closed_hat())
    assert f.shape == (N_FEATURES,) and np.all(np.isfinite(f))


def test_decay_measured_and_extrapolated():
    _, d = attack_decay_ms(closed_hat(decay_ms=25.0), SR)
    assert 15 < d < 40
    oh = open_hat(decay_ms=300.0)
    _, d_trunc = attack_decay_ms(oh[: int(0.1 * SR)], SR)  # cut at 100 ms by the next hit
    assert 180 < d_trunc < 500


def test_feature_ordering_between_voices():
    ch, oh, sh, pc = feats(closed_hat()), feats(open_hat()), feats(shaker()), feats(perc())
    assert ch[FI["centroid_hz"]] > sh[FI["centroid_hz"]] > pc[FI["centroid_hz"]]
    assert oh[FI["decay_ms"]] > 4 * ch[FI["decay_ms"]]
    assert sh[FI["attack_ms"]] > 3 * ch[FI["attack_ms"]]
    assert ch[FI["band_high"]] > 0.8 and pc[FI["band_mid"]] > 0.5
    assert pc[FI["flatness"]] < ch[FI["flatness"]]
    assert abs(sum(ch[FI[k]] for k in ("band_low", "band_mid", "band_high")) - 1) < 1e-3


def test_roles():
    assert describe_role(feats(closed_hat())) == "closed"
    assert describe_role(feats(open_hat())) == "open"
    assert describe_role(feats(shaker())) == "shaker"
    assert describe_role(feats(perc())) == "perc"


def test_sample_rate_independent_timbre():
    from loopforge.audio_io import resample

    x48 = closed_hat(sr=48000)
    x44 = resample(x48, 48000, 44100)
    a, b = feats(x48, 48000), feats(x44, 44100)
    mf = slice(FI["mfcc_0"], FI["mfcc_0"] + 13)
    assert np.linalg.norm(a[mf] - b[mf]) / np.linalg.norm(a[mf]) < 0.1
    assert abs(a[FI["centroid_hz"]] - b[FI["centroid_hz"]]) / a[FI["centroid_hz"]] < 0.05


def test_gain_changes_only_loudness():
    x = shaker()
    a, b = feats(x), feats(x * 0.25)
    assert abs((a[FI["rms_db"]] - b[FI["rms_db"]]) - 12.04) < 0.1
    for k in ("centroid_hz", "decay_ms", "attack_ms", "band_high"):
        assert abs(a[FI[k]] - b[FI[k]]) <= 1e-3 * max(1, abs(a[FI[k]]))
