"""Tempo and bar-grid estimation for top loops.

Top loops rarely contain a kick, so a generic beat tracker is unreliable. We
use, in order of precedence:

1. manual BPM,
2. BPM from the filename ("..._145_BPM.wav", "124bpm"), validated against length,
3. duration fit: the file is N whole bars for exactly one BPM in range, checked
   by how well the onsets sit on the implied 16th grid,
4. a grid search over BPM and phase that maximises onset/grid alignment
   (for files that are not trimmed loops).
"""

from __future__ import annotations

import re

import numpy as np

from loopforge.config import AnalysisParams
from loopforge.models import TempoEstimate

_BPM_PATTERNS = (
    re.compile(r"(\d{2,3}(?:[.,]\d+)?)\s*[-_ ]?\s*bpm", re.IGNORECASE),
    re.compile(r"bpm\s*[-_ ]?\s*(\d{2,3}(?:[.,]\d+)?)", re.IGNORECASE),
)


def bpm_from_name(name: str) -> float | None:
    for pat in _BPM_PATTERNS:
        m = pat.search(name)
        if m:
            bpm = float(m.group(1).replace(",", "."))
            if 40.0 <= bpm <= 250.0:
                return bpm
    return None


SWING_PENALTY = 0.3
BEAT_LOOP_TOLERANCE_S = 0.005


def swing_from_positions(pos: np.ndarray, min_hits: int = 3) -> float:
    """Swing (delay of odd 16ths, in steps, 0..0.5) from latency-corrected grid positions.

    Onsets that fall in the first half of an odd step are offbeat hits; their
    median delay is the swing. Fewer than `min_hits` offbeats -> 0.
    """
    k = np.floor(pos)
    frac = pos - k
    late = ((k.astype(np.int64) % 2) == 1) & (frac <= 0.5)
    if late.sum() < min_hits:
        return 0.0
    return float(np.clip(np.median(frac[late]), 0.0, 0.5))


def grid_score(
    onset_s: np.ndarray, weights: np.ndarray, step_s: float, phase_s: float
) -> tuple[float, float, float]:
    """How well onsets sit on a 16th grid (allowing swing and constant latency).

    Returns (score in 0..1, swing in steps, latency in steps).
    """
    if len(onset_s) == 0:
        return 0.0, 0.0, 0.0
    pos = (onset_s - phase_s) / step_s
    dev = pos - np.round(pos)
    even = (np.round(pos).astype(np.int64) % 2) == 0
    lat = float(np.median(dev[even])) if even.any() else 0.0  # constant latency
    pl = pos - lat
    swing = swing_from_positions(pl)
    k = np.floor(pl)
    odd_k = (k.astype(np.int64) % 2) == 1
    d_straight = np.abs(pl - np.round(pl))
    d_swung = np.where(odd_k, np.abs(pl - (k + swing)), np.inf)
    d = np.minimum(d_straight, d_swung)
    w = np.asarray(weights, dtype=np.float64)
    score = float(np.sum(w * np.clip(1.0 - d / 0.25, 0.0, 1.0)) / (np.sum(w) + 1e-12))
    return score, swing, lat


def _bar_fit(dur_s: float, bpm: float, p: AnalysisParams) -> tuple[int, int, bool]:
    """(bars, beats, is_loop). A loop is a whole number of beats (>= 2), e.g. 4 bars or 10 beats."""
    beat_s = 60.0 / bpm
    bar_s = p.beats_per_bar * beat_s
    tol = max(p.loop_tolerance_s, p.loop_tolerance_bar_frac * bar_s)
    beats = int(round(dur_s / beat_s))
    err = abs(dur_s - beats * beat_s)
    if beats % p.beats_per_bar:
        tol = min(tol, BEAT_LOOP_TOLERANCE_S)  # odd lengths must be tighter to count as loops
    is_loop = beats >= 2 and err <= tol
    if not is_loop:
        beats = int(np.floor(dur_s / beat_s + 1e-6))
    bars = max(1, beats // p.beats_per_bar)
    return bars, beats, is_loop


def _snap_bpm(bpm: float, dur_s: float, p: AnalysisParams) -> float:
    """Prefer a round BPM (integer, then 1 or 2 decimals) if it still fits the length as a loop."""
    for nd in (0, 1, 2):
        cand = round(bpm, nd)
        # length error of the loop at the snapped tempo must stay below 2 ms
        if abs(dur_s * (1.0 - bpm / cand)) <= 0.002:
            return float(cand)
    return float(bpm)


def _search(
    onset_s: np.ndarray, w: np.ndarray, p: AnalysisParams, bpm_lo: float, bpm_hi: float,
    step_bpm: float, n_phase: int, fixed_phase: bool,
) -> tuple[float, float, float]:
    best = (-1.0, 0.0, 0.0, 0.0)  # objective, bpm, phase, score
    for bpm in np.arange(bpm_lo, bpm_hi + 1e-9, step_bpm):
        step_s = 60.0 / bpm / p.steps_per_beat
        phases = [0.0] if fixed_phase else np.linspace(0.0, 2 * step_s, n_phase, endpoint=False)
        for ph in phases:
            sc, sw, lat = grid_score(onset_s, w, step_s, ph)
            obj = sc - SWING_PENALTY * sw  # prefer the straighter reading when ambiguous
            if obj > best[0] + 1e-9:
                best = (obj, float(bpm), float((ph + lat * step_s) % (2 * step_s)), sc)
    return best[3], best[1], best[2]


def _best_phase(onset_s: np.ndarray, w: np.ndarray, step_s: float, n: int = 48) -> tuple[float, float]:
    """Best grid phase in [0, 2 steps) (two steps so even/odd parity is resolved)."""
    best = (-1.0, 0.0, 0.0)
    for ph in np.linspace(0.0, 2 * step_s, n, endpoint=False):
        sc, sw, lat = grid_score(onset_s, w, step_s, ph)
        obj = sc - SWING_PENALTY * sw
        if obj > best[0] + 1e-9:
            best = (obj, float((ph + lat * step_s) % (2 * step_s)), sc)
    return best[2], best[1]


def estimate_tempo(
    onset_samples: np.ndarray,
    onset_weights: np.ndarray,
    n_samples: int,
    sr: int,
    name: str,
    params: AnalysisParams,
    manual_bpm: float | None = None,
) -> TempoEstimate:
    p = params
    dur = n_samples / sr
    t = np.asarray(onset_samples, dtype=np.float64) / sr
    w = np.asarray(onset_weights, dtype=np.float64)
    if len(w) != len(t) or len(w) == 0:
        w = np.ones_like(t)

    def from_bpm(bpm: float, method: str) -> TempoEstimate:
        bars, beats, is_loop = _bar_fit(dur, bpm, p)
        step_s = 60.0 / bpm / p.steps_per_beat
        score, phase = 0.0, 0.0
        if is_loop:
            score, _, lat = grid_score(t, w, step_s, 0.0)
            if beats % p.beats_per_bar and (score < p.min_grid_score or abs(lat) > 0.15):
                is_loop = False  # whole beats by coincidence; onsets do not sit on a grid from t=0
        if not is_loop:
            score, phase = _best_phase(t, w, step_s)
        return TempoEstimate(bpm=float(bpm), bars=bars, method=method, confidence=round(score, 4),
                             is_loop=is_loop, phase_s=phase, beats=beats)

    if manual_bpm:
        return from_bpm(float(manual_bpm), "manual")

    fb = bpm_from_name(name)
    if fb is not None:
        est = from_bpm(fb, "filename")
        if est.is_loop or est.confidence >= p.min_grid_score:
            return est

    # Duration fit: whole bars at a tempo inside the configured range.
    cands: list[TempoEstimate] = []
    for nb in p.candidate_bars:
        bpm = _snap_bpm(nb * p.beats_per_bar * 60.0 / dur, dur, p)
        if p.bpm_min - 1e-6 <= bpm <= p.bpm_max + 1e-6:
            step_s = 60.0 / bpm / p.steps_per_beat
            sc, _, _ = grid_score(t, w, step_s, 0.0)
            cands.append(TempoEstimate(bpm=float(bpm), bars=nb, method="duration",
                                       confidence=round(sc, 4), is_loop=True, phase_s=0.0,
                                       beats=nb * p.beats_per_bar))
    if cands:
        centre = np.sqrt(p.bpm_min * p.bpm_max)
        top = max(c.confidence for c in cands)
        good = [c for c in cands if c.confidence >= top - 0.05]
        best = min(good, key=lambda c: abs(np.log(c.bpm / centre)))
        if best.confidence >= p.min_grid_score:
            return best

    # Fallback: coarse-to-fine search over BPM and grid phase.
    sc, bpm, _ = _search(t, w, p, p.bpm_min, p.bpm_max, 0.5, 24, fixed_phase=False)
    sc, bpm, phase = _search(t, w, p, max(p.bpm_min, bpm - 0.6), min(p.bpm_max, bpm + 0.6), 0.02, 64,
                             fixed_phase=False)
    bars, beats, _ = _bar_fit(dur, bpm, p)
    return TempoEstimate(bpm=round(bpm, 3), bars=bars, method="autocorr", confidence=round(sc, 4),
                         is_loop=False, phase_s=phase, beats=beats)
