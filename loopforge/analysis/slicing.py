"""Turn onsets into slices mapped onto the 16th-note grid."""

from __future__ import annotations

import numpy as np

from loopforge.analysis.tempo import swing_from_positions
from loopforge.config import AnalysisParams
from loopforge.models import Slice, TempoEstimate


def grid_positions(onset_samples: np.ndarray, sr: int, tempo: TempoEstimate, p: AnalysisParams) -> np.ndarray:
    step_s = tempo.step_len_s(p.steps_per_beat)
    return (np.asarray(onset_samples, dtype=np.float64) / sr - tempo.phase_s) / step_s


def estimate_timing(pos: np.ndarray) -> tuple[float, float]:
    """Return (grid_offset, swing) in steps from raw grid positions."""
    if len(pos) == 0:
        return 0.0, 0.0
    r = np.round(pos)
    even = (r.astype(np.int64) % 2) == 0
    grid_offset = float(np.median((pos - r)[even])) if even.any() else 0.0
    return grid_offset, swing_from_positions(pos - grid_offset)


def assign_steps(pos: np.ndarray, grid_offset: float, swing: float) -> np.ndarray:
    """Nearest grid step for each onset, where odd steps are expected `swing` late."""
    pl = pos - grid_offset
    k0 = np.floor(pl)
    cands = np.stack([k0 - 1, k0, k0 + 1, k0 + 2])  # (4, n)
    expected = cands + np.where((cands.astype(np.int64) % 2) == 1, swing, 0.0)
    best = np.argmin(np.abs(pl[None, :] - expected), axis=0)
    return cands[best, np.arange(len(pl))].astype(np.int64)


def make_slices(
    onset_samples: np.ndarray, n_samples: int, sr: int, tempo: TempoEstimate, p: AnalysisParams
) -> tuple[list[Slice], float, float, int]:
    """Slice at onsets. Returns (slices, swing, grid_offset, steps_total)."""
    onsets = np.asarray(onset_samples, dtype=np.int64)
    steps_per_bar = p.beats_per_bar * p.steps_per_beat
    steps_total = (tempo.beats * p.steps_per_beat) if tempo.beats else tempo.bars * steps_per_bar
    if len(onsets) == 0:
        return [], 0.0, 0.0, steps_total

    pos = grid_positions(onsets, sr, tempo, p)
    grid_offset, swing = estimate_timing(pos)
    unwrapped = assign_steps(pos, grid_offset, swing)
    offsets = pos - unwrapped  # micro-timing incl. swing and latency, in steps
    steps = np.mod(unwrapped, steps_total) if tempo.is_loop else unwrapped

    beat_samples = 60.0 / tempo.bpm * sr
    max_len = int(round(p.max_slice_beats * beat_samples))
    slices: list[Slice] = []
    for i, start in enumerate(onsets.tolist()):
        nxt = int(onsets[i + 1]) if i + 1 < len(onsets) else n_samples
        end = max(start + 1, min(nxt, start + max_len, n_samples))
        slices.append(Slice(index=i, start=int(start), end=int(end), step=int(steps[i]),
                            offset=round(float(offsets[i]), 4)))
    return slices, swing, grid_offset, steps_total
