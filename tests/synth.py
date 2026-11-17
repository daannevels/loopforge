"""Synthetic test audio: hats, shakers, percussion and loops built from them.

No copyrighted material is used anywhere in the tests; everything is rendered
here from seeded noise and sines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48000


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _hp(x: np.ndarray, f: float, sr: int, order: int = 4) -> np.ndarray:
    return sosfilt(butter(order, f, btype="highpass", fs=sr, output="sos"), x).astype(np.float32)


def _bp(x: np.ndarray, lo: float, hi: float, sr: int, order: int = 4) -> np.ndarray:
    return sosfilt(butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos"), x).astype(np.float32)


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    return (x / (np.abs(x).max() + 1e-12) * peak).astype(np.float32)


def closed_hat(sr: int = SR, decay_ms: float = 25.0, seed: int = 0, cutoff: float = 7000.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * decay_ms * 6 / 1000)
    t = np.arange(n) / sr
    env = np.exp(-t / (decay_ms / 1000 / np.log(10)))  # -20 dB at decay_ms
    return _norm(_hp(_noise(n, rng), cutoff, sr) * env)


def open_hat(sr: int = SR, decay_ms: float = 300.0, seed: int = 1) -> np.ndarray:
    return closed_hat(sr, decay_ms=decay_ms, seed=seed, cutoff=6000.0)


def shaker(sr: int = SR, attack_ms: float = 18.0, decay_ms: float = 70.0, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * (attack_ms + decay_ms * 4) / 1000)
    t = np.arange(n) / sr
    a = attack_ms / 1000
    env = np.where(t < a, 0.5 - 0.5 * np.cos(np.pi * t / a), np.exp(-(t - a) / (decay_ms / 1000 / np.log(10))))
    return _norm(_bp(_noise(n, rng), 3000.0, 9000.0, sr) * env)


def perc(sr: int = SR, freq: float = 700.0, decay_ms: float = 60.0, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * decay_ms * 5 / 1000)
    t = np.arange(n) / sr
    env = np.exp(-t / (decay_ms / 1000 / np.log(10)))
    tone = np.sin(2 * np.pi * freq * t) * env
    click = _bp(_noise(n, rng), 1500.0, 5000.0, sr) * np.exp(-t / 0.003) * 0.3
    return _norm(tone + click)


def clap(sr: int = SR, seed: int = 4, lo: float = 900.0, hi: float = 2600.0, decay_ms: float = 90.0) -> np.ndarray:
    """Hand clap: three quick noise bursts then a decaying room tail, band-passed to the midrange."""
    rng = np.random.default_rng(seed)
    n = int(sr * 0.35)
    t = np.arange(n) / sr
    env = np.zeros(n)
    for k, d in enumerate((0.0, 0.011, 0.022)):
        m = t >= d
        env[m] += (0.6 + 0.2 * k) * np.exp(-(t[m] - d) / 0.004)
    m = t >= 0.022
    env[m] += 0.5 * np.exp(-(t[m] - 0.022) / (decay_ms / 1000 / np.log(10)))
    return _norm(_bp(_noise(n, rng), lo, hi, sr) * env)


def click(sr: int = SR) -> np.ndarray:
    x = np.zeros(int(0.005 * sr), np.float32)
    x[:24] = np.hanning(48)[24:] * 0.9
    return x


VOICES = {"closed": closed_hat, "open": open_hat, "shaker": shaker, "perc": perc, "click": click, "clap": clap}


@dataclass
class Hit:
    step: int
    voice: str = "closed"
    vel: float = 1.0
    pan: float = 0.0  # -1..1


def render_loop(
    hits: list[Hit],
    bpm: float,
    bars: int,
    sr: int = SR,
    swing: float = 0.0,
    stereo: bool = True,
    seed: int = 0,
    circular: bool = True,
    pre_silence_s: float = 0.0,
    tail_s: float = 0.0,
) -> np.ndarray:
    """Render hits on a 16th grid into an exact N-bar loop (channels, samples).

    `swing` delays odd steps by that fraction of a step. Tails past the loop end
    wrap to the start when `circular`, so the loop is seamless.
    """
    step_s = 60.0 / bpm / 4
    n = int(round(bars * 16 * step_s * sr))
    lead = int(round(pre_silence_s * sr))
    total = lead + n + int(round(tail_s * sr))
    out = np.zeros((2, total + 3 * sr), np.float32)
    cache: dict[str, np.ndarray] = {}
    for h in hits:
        if h.voice not in cache:
            fn = VOICES[h.voice]
            cache[h.voice] = fn(sr) if h.voice == "click" else fn(sr=sr, seed=seed + list(VOICES).index(h.voice))
        x = cache[h.voice] * h.vel
        pos = lead + int(round((h.step + (swing if h.step % 2 else 0.0)) * step_s * sr))
        gl, gr = np.sqrt(0.5 * (1 - h.pan)), np.sqrt(0.5 * (1 + h.pan))
        out[0, pos : pos + len(x)] += x * gl * np.sqrt(2)
        out[1, pos : pos + len(x)] += x * gr * np.sqrt(2)
    if circular and lead == 0:
        body = out[:, :n].copy()
        tail = out[:, n:]
        k = min(n, tail.shape[1])
        body[:, :k] += tail[:, :k]
        res = body
    else:
        res = out[:, :total]
    res = res * (0.9 / max(1e-9, float(np.abs(res).max())))
    return res if stereo else res.mean(axis=0, keepdims=True)


def basic_pattern(bars: int = 1) -> list[Hit]:
    """Busy top loop: closed 16ths with accents, open hat on offbeat 8ths, shaker, perc."""
    hits: list[Hit] = []
    for b in range(bars):
        o = b * 16
        for s in range(16):
            if s % 4 == 2:
                hits.append(Hit(o + s, "open", 0.8))
            elif s in (5, 13):
                hits.append(Hit(o + s, "perc", 0.7))
            elif s in (7, 15):
                hits.append(Hit(o + s, "shaker", 0.6))
            else:
                hits.append(Hit(o + s, "closed", 1.0 if s % 4 == 0 else 0.55))
    return hits


def sparse_pattern(steps: list[int], voice: str = "closed", bars: int = 1) -> list[Hit]:
    return [Hit(b * 16 + s, voice, 1.0) for b in range(bars) for s in steps]


def write(path: Path, audio: np.ndarray, sr: int = SR, subtype: str = "PCM_24") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = {".wav": "WAV", ".aif": "AIFF", ".aiff": "AIFF", ".flac": "FLAC"}[path.suffix.lower()]
    sf.write(str(path), np.asarray(audio).T, sr, subtype=subtype, format=fmt)
    return path
