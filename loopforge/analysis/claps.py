"""Clap detection for top loops (pure, array in -> masks out).

A clap is a mid-heavy hit on the backbeat that stands out in the midrange: >= CLAP_MIN_MID of its energy
in 250 Hz-4 kHz, and its midrange level (rms_db + 10*log10(band_mid)) >= CLAP_MIN_REL_DB above the median
midrange level of its own loop, on beat 2 or 4 (step % 8 == 4).
Extra hits on the same step (flams) that are mid-heavy belong to the clap too.

A clap's room/reverb tail rings over the next few 16ths, so in a sample-pack loop the hits right after
a clap carry that clap. Those slices are "tainted": using one as a hat would bring the pack's clap
along. Tainted = within `clap_tail_steps` after a clap in the same loop and still mid-heavy.
"""

from __future__ import annotations

import numpy as np

from loopforge.models import FEATURE_INDEX as FI

CLAP_MIN_MID = 0.5
CLAP_MIN_REL_DB = 5.0
CLAP_TAIL_STEPS = 3
TAIL_MIN_MID = 0.45


def clap_flags(features: np.ndarray, steps: np.ndarray, groups: np.ndarray | None = None,
               steps_per_bar: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(is_clap, tainted) boolean masks for slices; `groups` = loop/file id per slice (one loop if None).
    Slices of one loop must be in time order."""
    F = np.asarray(features)
    n = len(F)
    steps = np.asarray(steps, dtype=np.int64)
    groups = np.zeros(n, np.int64) if groups is None else np.asarray(groups)
    is_clap = np.zeros(n, bool)
    tainted = np.zeros(n, bool)
    if n == 0:
        return is_clap, tainted
    mid = F[:, FI["band_mid"]]
    rms = F[:, FI["rms_db"]]
    mid_lvl = rms + 10.0 * np.log10(np.maximum(mid, 1e-4))
    backbeat = (steps % (steps_per_bar // 2)) == (steps_per_bar // 4)  # beats 2 and 4
    for g in np.unique(groups):
        idx = np.nonzero(groups == g)[0]
        audible = rms[idx] > -50.0
        ml = mid_lvl[idx][audible] if audible.any() else mid_lvl[idx]
        med = float(np.median(ml))
        core = idx[(mid[idx] >= CLAP_MIN_MID) & (mid_lvl[idx] >= med + CLAP_MIN_REL_DB) & backbeat[idx]]
        is_clap[core] = True
        clap_steps = set(steps[core].tolist())
        for i in idx:  # flams of a clap
            if not is_clap[i] and int(steps[i]) in clap_steps and mid[i] >= CLAP_MIN_MID:
                is_clap[i] = True
        if not clap_steps:
            continue
        cs = np.array(sorted(clap_steps))
        for i in idx:
            if is_clap[i] or mid[i] < TAIL_MIN_MID:
                continue
            d = int(steps[i]) - cs
            if ((d >= 1) & (d <= CLAP_TAIL_STEPS)).any():
                tainted[i] = True
    return is_clap, tainted


def clap_signature(mono: np.ndarray, sr: int, ms: float = 50.0, n_bands: int = 12,
                   lo_hz: float = 300.0, hi_hz: float = 5000.0) -> np.ndarray:
    """Log-band spectrum of a hit's first `ms` (its body) in the clap's own range, for "is this the same
    clap sample?" checks. Above ~5 kHz the hats layered on top dominate, so that range is ignored."""
    n = int(round(ms * sr / 1000))
    x = np.asarray(mono, dtype=np.float64)[:n]
    x = np.pad(x, (0, n - len(x)))
    X = np.abs(np.fft.rfft(x * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    edges = np.logspace(np.log10(lo_hz), np.log10(min(hi_hz, sr / 2 - 1)), n_bands + 1)
    return np.log10(np.array([X[(f >= lo) & (f < hi)].mean() + 1e-9 for lo, hi in zip(edges[:-1], edges[1:])]))


def same_clap(sig_a: np.ndarray, sig_b: np.ndarray, threshold: float = 0.85) -> bool:
    return float(np.corrcoef(sig_a, sig_b)[0, 1]) >= threshold
