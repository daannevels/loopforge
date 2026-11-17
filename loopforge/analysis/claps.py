"""Clap/snare identity check: is this the same sample as the reference's clap/snare?

(Detection of claps, snares and their tails lives in analysis/instruments.py.)
"""

from __future__ import annotations

import numpy as np


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
