"""Per-slice feature extraction. See models.FEATURE_SPEC for the vector layout."""

from __future__ import annotations

import numpy as np

from loopforge.analysis import dsp
from loopforge.config import AnalysisParams
from loopforge.models import FEATURE_INDEX, N_FEATURES

_EPS = 1e-10


def _envelope_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """1 ms RMS envelope sampled every 0.5 ms, in dB. Returns (env_db, hop_ms)."""
    win = max(2, int(0.001 * sr))
    hop = max(1, win // 2)
    power = np.convolve(x.astype(np.float64) ** 2, np.ones(win) / win, mode="full")[: len(x)]
    env = power[::hop]
    return 10.0 * np.log10(env + _EPS), 1000.0 * hop / sr


def attack_decay_ms(x: np.ndarray, sr: int) -> tuple[float, float]:
    """Attack = start -> envelope peak. Decay = peak -> -20 dB.

    When the slice is cut off before reaching -20 dB (the next hit follows), the
    decay is extrapolated from a linear fit of the dB envelope after the peak.
    """
    env_db, hop_ms = _envelope_db(x, sr)
    if len(env_db) < 2:
        return 0.0, 0.0
    ip = int(np.argmax(env_db))
    peak = env_db[ip]
    attack = ip * hop_ms
    tail = env_db[ip:]
    below = np.nonzero(tail <= peak - 20.0)[0]
    if len(below):
        return attack, float(below[0] * hop_ms)
    valid = tail > peak - 40.0
    tt = np.arange(len(tail))[valid] * hop_ms
    if len(tt) < 4 or tt[-1] <= 0:
        return attack, float(min(2000.0, max(len(tail) * hop_ms, 0.5)))
    slope = np.polyfit(tt, tail[valid], 1)[0]  # dB per ms
    if slope >= -1e-3:
        return attack, 2000.0
    return attack, float(np.clip(-20.0 / slope, 0.5, 2000.0))


def tonal_db(x: np.ndarray, sr: int, ms: float = 80.0) -> float:
    """How strongly one pitch stands out between 100 Hz and 3 kHz: the largest spectral peak relative to
    the local spectrum around it (moving median over +-~150 Hz, so filter slopes don't count), in dB, in
    the first `ms`. Tonal percussion (conga, bongo, rim, wood) scores well above noise (hats, claps, snares)."""
    from scipy.ndimage import median_filter

    n = max(64, int(ms * sr / 1000))
    y = np.asarray(x[:n], dtype=np.float64)
    y = np.pad(y, (0, n - len(y)))
    X = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1.0 / sr)
    sel = (f >= 100.0) & (f < 3000.0)
    if sel.sum() < 8 or X[sel].sum() <= 0:
        return 0.0
    w = max(5, int(round(300.0 / (f[1] - f[0]))) | 1)  # ~300 Hz window
    local = median_filter(X, size=w, mode="nearest")
    ratio = X[sel] / (local[sel] + 1e-20)
    # weight by how much energy the band carries: a pitch peak in an empty band means nothing
    share = X[sel].sum() / (X.sum() + 1e-20)
    return float(10.0 * np.log10(ratio.max() + 1e-12) * min(1.0, share / 0.2))


def slice_features(x: np.ndarray, sr: int, p: AnalysisParams) -> np.ndarray:
    """Feature vector for one mono slice."""
    x = np.asarray(x, dtype=np.float32)
    v = np.zeros(N_FEATURES, dtype=np.float64)
    if len(x) < 8:
        x = np.pad(x, (0, 8 - len(x)))
    fi = FEATURE_INDEX

    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    v[fi["rms_db"]] = 20.0 * np.log10(rms + _EPS)
    v[fi["peak_db"]] = 20.0 * np.log10(float(np.abs(x).max()) + _EPS)
    v[fi["attack_ms"]], v[fi["decay_ms"]] = attack_decay_ms(x, sr)
    v[fi["duration_ms"]] = 1000.0 * len(x) / sr

    n_fft = dsp.fft_size(sr, p.feature_win_s)
    S_full = dsp.power_spectrogram(x, n_fft)  # (freq, frames)
    freqs_full = np.fft.rfftfreq(n_fft, 1.0 / sr)
    keep = freqs_full <= p.spectral_fmax
    S, freqs = S_full[keep], freqs_full[keep]
    frame_e = S.sum(axis=0)
    w = frame_e / (frame_e.sum() + _EPS)

    centroid = (freqs[:, None] * S).sum(0) / (frame_e + _EPS)
    cum = np.cumsum(S, axis=0)
    roll_idx = np.argmax(cum >= p.rolloff_pct * cum[-1:, :], axis=0)
    rolloff = freqs[roll_idx]
    flat = np.exp(np.mean(np.log(S + _EPS), axis=0)) / (np.mean(S, axis=0) + _EPS)
    v[fi["centroid_hz"]] = float(w @ centroid)
    v[fi["rolloff_hz"]] = float(w @ rolloff)
    v[fi["flatness"]] = float(w @ flat)

    total = S.sum(axis=1)
    lo, hi = p.band_edges_hz
    tot = total.sum() + _EPS
    v[fi["band_low"]] = total[freqs < lo].sum() / tot
    v[fi["band_mid"]] = total[(freqs >= lo) & (freqs < hi)].sum() / tot
    v[fi["band_high"]] = total[freqs >= hi].sum() / tot

    v[fi["tonal_db"]] = tonal_db(x, sr)

    mf = dsp.mfcc_from_power(S_full, sr, n_fft, p.n_mfcc, p.mfcc_fmax)
    v[fi["mfcc_0"] : fi["mfcc_0"] + p.n_mfcc] = mf @ w
    return v.astype(np.float32)


def describe_role(f: np.ndarray) -> str:
    """Human-readable role guess for one feature vector (diagnostics only)."""
    fi = FEATURE_INDEX
    low_mid = f[fi["band_low"]] + f[fi["band_mid"]]
    if low_mid > 0.6:
        return "perc"
    if f[fi["decay_ms"]] >= 150.0:
        return "open"
    if f[fi["attack_ms"]] >= 8.0:
        return "shaker"
    return "closed"
