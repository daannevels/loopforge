"""Tail extension for hits that were cut short in their source loop.

Library slices end where the next hit starts in the source loop, so in dense loops
an open hat is truncated after one 16th. Hats and shakers are noise-like, so a
convincing continuation can be built by overlap-adding short random grains taken
from the slice's own last few milliseconds (before the cut) and shaping them with
the sound's measured exponential decay. Deterministic for a given seed.
"""

from __future__ import annotations

import numpy as np


def extend_tail(
    x: np.ndarray,
    sr: int,
    extra_ms: float,
    decay_ms: float,
    seed: int,
    region_ms: float = 30.0,
    guard_ms: float = 2.0,
    grain_ms: float = 20.0,
    xfade_ms: float = 6.0,
) -> np.ndarray:
    """Return `x` (channels, n) followed by `extra_ms` of synthesized decaying tail."""
    x = np.asarray(x, dtype=np.float32)
    ch, n = x.shape
    extra = int(round(extra_ms * sr / 1000))
    reg = int(round(region_ms * sr / 1000))
    guard = int(round(guard_ms * sr / 1000))
    g = max(32, int(round(grain_ms * sr / 1000)) // 2 * 2)
    xf = int(round(xfade_ms * sr / 1000))
    if extra <= 0 or n < reg + guard + 8 or decay_ms <= 0:
        return x
    src = x[:, n - guard - reg : n - guard]  # clean material just before the cut
    level = float(np.sqrt(np.mean(src.astype(np.float64) ** 2)))
    if level <= 1e-7:
        return x
    rng = np.random.default_rng(seed)
    total = extra + xf
    hop = g // 2
    win = np.hanning(g).astype(np.float32)
    out = np.zeros((ch, total + g), np.float32)
    max_off = max(1, src.shape[1] - g)
    grain_src = src if src.shape[1] >= g else np.pad(src, ((0, 0), (0, g - src.shape[1])))
    for pos in range(0, total, hop):
        o = int(rng.integers(0, max_off)) if max_off > 1 else 0
        seg = grain_src[:, o : o + g]
        if rng.random() < 0.5:
            seg = seg[:, ::-1]  # reversed grains reduce audible repetition
        out[:, pos : pos + g] += seg * win
    out = out[:, :total]
    syn = float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) + 1e-12
    out *= np.float32(level / syn)
    t_ms = np.arange(total) * 1000.0 / sr
    out *= (0.1 ** (t_ms / decay_ms)).astype(np.float32)[None, :]  # -20 dB per decay_ms
    # equal-power crossfade over the last `xf` samples of x
    y = np.zeros((ch, n + extra), np.float32)
    y[:, :n] = x
    k = min(xf, n)
    ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
    y[:, n - k : n] *= np.sqrt(1.0 - ramp)
    y[:, n - k : n + extra] += np.concatenate([out[:, :k] * np.sqrt(ramp), out[:, k : k + extra]], axis=1)[:, : k + extra]
    return y
