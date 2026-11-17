"""analyze_loop(): tempo -> onsets -> slices -> features. Pure (no file I/O)."""

from __future__ import annotations

import numpy as np

from loopforge.analysis.features import slice_features
from loopforge.analysis.instruments import classify_hits
from loopforge.analysis.onsets import detect_onsets
from loopforge.analysis.slicing import make_slices
from loopforge.analysis.tempo import estimate_tempo
from loopforge.audio_io import to_mono
from loopforge.config import AnalysisParams
from loopforge.models import N_FEATURES, LoopAnalysis


def analyze_loop(
    audio: np.ndarray,
    sr: int,
    name: str = "",
    params: AnalysisParams | None = None,
    bpm: float | None = None,
) -> LoopAnalysis:
    """Analyse a loop given as (channels, samples) or mono float32 audio."""
    p = params or AnalysisParams()
    audio = np.asarray(audio, dtype=np.float32)
    channels = 1 if audio.ndim == 1 else audio.shape[0]
    mono = to_mono(audio)
    n = len(mono)
    notes: list[str] = []

    # Pass 1: linear onsets to estimate tempo.
    ons = detect_onsets(mono, sr, p, circular=False)
    tempo = estimate_tempo(ons.samples, ons.peaks, n, sr, name, p, manual_bpm=bpm)
    # Pass 2: if the file is a seamless loop, re-detect circularly so a hit at
    # t=0 (or one pushed slightly early across the loop point) is caught.
    if tempo.is_loop:
        ons = detect_onsets(mono, sr, p, circular=True)
    else:
        notes.append("file is not a whole number of beats; treated as non-looping")
    if tempo.confidence < p.min_grid_score:
        notes.append(f"low grid confidence ({tempo.confidence:.2f}); consider --bpm")

    slices, swing, grid_offset, steps_total = make_slices(ons.samples, n, sr, tempo, p)
    feats = np.zeros((len(slices), N_FEATURES), dtype=np.float32)
    for i, s in enumerate(slices):
        feats[i] = slice_features(mono[s.start : s.end], sr, p)
    labels = classify_hits(feats, np.array([s.step for s in slices], dtype=np.int64))
    for s, lab in zip(slices, labels):
        s.role = str(lab)  # instrument scan: closed_hat / open_hat / shaker / clap / snare / perc / other / tail

    return LoopAnalysis(
        name=name, sr=sr, n_samples=n, channels=channels, tempo=tempo, swing=swing,
        grid_offset=grid_offset, slices=slices, features=feats, steps_total=steps_total, notes=notes,
    )
