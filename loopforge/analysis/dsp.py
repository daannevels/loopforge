"""Thin wrappers around the DSP primitives we borrow from librosa.

Everything librosa-specific is funnelled through here so a C++ port only has to
reimplement these few functions (spectral-flux onset envelope, mel filterbank,
DCT) with matching parameters.
"""

from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np
from scipy.fft import dct


def onset_envelope(mono: np.ndarray, sr: int, hop: int, compression: float = 10.0) -> np.ndarray:
    """Spectral-flux onset strength (log-mel, lag 1), one value per hop.

    The envelope is log-compressed (log1p(c * env / max)) so quiet hats next to
    a loud perc hit or under a ringing open hat still form clear peaks.
    """
    env = librosa.onset.onset_strength(y=np.asarray(mono, dtype=np.float32), sr=sr, hop_length=hop, n_fft=1024)
    if compression > 0 and env.max() > 0:
        env = np.log1p(compression * env / env.max())
    return env


def pick_peaks(env: np.ndarray, sr: int, hop: int, delta: float) -> np.ndarray:
    """Peak-pick an onset envelope; returns frame indices."""
    return librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, units="frames", delta=delta, backtrack=False
    )


def fft_size(sr: int, win_s: float) -> int:
    return int(2 ** np.ceil(np.log2(max(64, win_s * sr))))


def power_spectrogram(mono: np.ndarray, n_fft: int) -> np.ndarray:
    """|STFT|^2, Hann window, hop n_fft/4, no centering; input zero-padded to >= n_fft."""
    y = np.asarray(mono, dtype=np.float32)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    hop = n_fft // 4
    n_frames = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * _hann(n_fft)[None, :]
    return (np.abs(np.fft.rfft(frames, axis=1)) ** 2).T.astype(np.float64)  # (freq, frames)


@lru_cache(maxsize=16)
def _hann(n: int) -> np.ndarray:
    return np.hanning(n + 1)[:-1].astype(np.float32)


@lru_cache(maxsize=16)
def mel_basis(sr: int, n_fft: int, n_mels: int, fmax: float) -> np.ndarray:
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=min(fmax, sr / 2))


def mfcc_from_power(S: np.ndarray, sr: int, n_fft: int, n_mfcc: int, fmax: float, n_mels: int = 40) -> np.ndarray:
    """MFCCs (n_mfcc, frames) from a power spectrogram: mel -> dB -> DCT-II (ortho)."""
    mel = mel_basis(sr, n_fft, n_mels, fmax) @ S
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    return dct(mel_db, type=2, axis=0, norm="ortho")[:n_mfcc]
