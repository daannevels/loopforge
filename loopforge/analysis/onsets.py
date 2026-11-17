"""Onset detection with sample-accurate attack refinement.

Loops are treated circularly: the tail of the loop is prepended (and the head
appended) before detection, so a hit right at sample 0 is still found and its
attack is not confused with the file start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from loopforge.analysis import dsp
from loopforge.config import AnalysisParams


@dataclass
class Onsets:
    samples: np.ndarray  # attack start (incl. pre-roll), int64, sorted, in [0, n)
    peaks: np.ndarray  # linear peak amplitude of each hit (first 30 ms)


def detect_onsets(mono: np.ndarray, sr: int, params: AnalysisParams, circular: bool) -> Onsets:
    mono = np.asarray(mono, dtype=np.float32)
    n = len(mono)
    hop = params.hop_length
    pad = min(int(params.circular_pad_s * sr), n) if circular else int(0.05 * sr)
    if circular:
        y = np.concatenate([mono[n - pad :], mono, mono[:pad]])
    else:
        y = np.concatenate([np.zeros(pad, np.float32), mono, np.zeros(pad, np.float32)])

    env = dsp.onset_envelope(y, sr, hop, params.onset_compression)
    frames = dsp.pick_peaks(env, sr, hop, params.onset_delta)

    # Short moving max of |y| as a sample-level amplitude envelope.
    amp = maximum_filter1d(np.abs(y), size=max(3, int(0.0005 * sr)))
    global_peak = float(np.abs(mono).max()) + 1e-12
    gate = max(global_peak * 10 ** (params.onset_gate_db / 20), 10 ** (params.onset_abs_gate_db / 20))
    hit_win = int(0.03 * sr)
    preroll = int(round(params.preroll_s * sr))

    starts: list[int] = []
    peaks: list[float] = []
    prev_start = -(10**9)
    for f in frames:
        c = int(f) * hop  # padded-domain sample of the onset frame
        lo = max(0, c - 3 * hop, prev_start + 1)
        hi = min(len(y), c + 3 * hop + int(0.01 * sr))
        if hi - lo < 4:
            continue
        ip = lo + int(np.argmax(amp[lo:hi]))
        floor = float(amp[lo : ip + 1].min())
        thr = floor + 0.15 * (float(amp[ip]) - floor)
        i = ip
        while i > lo and amp[i - 1] > thr:
            i -= 1
        start = max(lo, i - preroll)
        pk = float(np.abs(y[start : start + hit_win]).max())
        if pk < gate:
            continue
        starts.append(start)
        peaks.append(pk)
        prev_start = start

    s = np.asarray(starts, dtype=np.int64) - pad
    p = np.asarray(peaks, dtype=np.float64)
    tol = int(params.min_onset_gap_s * sr)
    # Hits in the padded head duplicate hits near the file end (and vice versa);
    # hits starting slightly before 0 belong to step 0.
    keep = (s >= -tol) & (s < n - tol) if circular else (s >= -tol) & (s < n)
    s, p = s[keep], p[keep]
    s = np.clip(s, 0, n - 1)

    # Merge onsets closer than the minimum gap (keep the first, max of peaks).
    out_s: list[int] = []
    out_p: list[float] = []
    for si, pi in zip(s.tolist(), p.tolist()):
        if out_s and si - out_s[-1] < tol:
            out_p[-1] = max(out_p[-1], pi)
            continue
        out_s.append(si)
        out_p.append(pi)
    return Onsets(np.asarray(out_s, dtype=np.int64), np.asarray(out_p, dtype=np.float64))
