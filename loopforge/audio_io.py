"""Audio file I/O. All audio inside loopforge is float32 shaped (channels, samples)."""

from __future__ import annotations

import hashlib
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SUPPORTED_EXTENSIONS = (".wav", ".aif", ".aiff", ".flac")


def is_audio_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load wav/aiff/flac as float32 (channels, samples), keeping all channels."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), int(sr)


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x if x.ndim == 1 else x.mean(axis=0)


def to_stereo(x: np.ndarray) -> np.ndarray:
    """Mono -> centred stereo (equal copies); stereo passes through; >2 ch keep first two."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[0] == 1:
        return np.repeat(x, 2, axis=0)
    return x[:2]


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resampling along the last axis."""
    if sr_from == sr_to:
        return np.asarray(x, dtype=np.float32)
    g = gcd(int(sr_from), int(sr_to))
    y = resample_poly(x, sr_to // g, sr_from // g, axis=-1)
    return y.astype(np.float32)


def write_wav(path: str | Path, x: np.ndarray, sr: int, subtype: str = "PCM_24") -> Path:
    """Write stereo 24-bit WAV. Mono input is duplicated to both channels."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(to_stereo(x), -1.0, 1.0)
    sf.write(str(path), y.T, int(sr), subtype=subtype, format="WAV")
    return path


def file_sha1(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
