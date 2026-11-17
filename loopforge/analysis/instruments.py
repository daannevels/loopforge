"""Instrument scan: label every hit of a loop as one instrument class.

Classes: closed_hat, open_hat, shaker, clap, snare, perc, other, plus "tail" for hits that are mostly
the room tail of a preceding clap/snare (not an instrument of their own).

Rules (top loops are usually high-passed, so snare "body" frequencies are gone; what remains is used):
- low-heavy (band_low >= 0.35)                         -> other   (kick, low tom, bass)
- midrange-heavy (band_mid >= 0.5, or dull & not bright):
    tonal (tonal_db >= 25: clear pitch peak 100-1200 Hz) -> perc    (conga, bongo, rim, wood)
    noisy                                                 -> clap or snare, decided per loop from its
      backbeat hits (beats 2 & 4): median attack >= 8 ms = clap (several hand bursts, peak comes late),
      else snare (one sharp stick transient). Ghost notes off the backbeat take the loop's choice.
- bright (hats):
    attack >= 15 ms and decay < 130 ms -> shaker
    decay >= 130 ms                    -> open_hat
    otherwise                          -> closed_hat
- tails: a noisy-midrange hit 1-3 steps after a clap/snare that is >= 8 dB quieter than it.
Pure function over feature arrays; per-loop context comes from `groups`.
"""

from __future__ import annotations

import numpy as np

from loopforge.models import FEATURE_INDEX as FI

CLASSES = ("closed_hat", "open_hat", "shaker", "clap", "snare", "perc", "other")
TAIL = "tail"
BACKBEAT_SNARE_CLAP = ("clap", "snare")


def classify_hits(features: np.ndarray, steps: np.ndarray, groups: np.ndarray | None = None,
                  steps_per_bar: int = 16) -> np.ndarray:
    """Instrument label per hit (object array of str). Hits of one loop must be in time order."""
    F = np.asarray(features)
    n = len(F)
    steps = np.asarray(steps, dtype=np.int64)
    groups = np.zeros(n, np.int64) if groups is None else np.asarray(groups)
    out = np.empty(n, dtype=object)
    if n == 0:
        return out
    lo, mid, hi = F[:, FI["band_low"]], F[:, FI["band_mid"]], F[:, FI["band_high"]]
    cen, atk, dec = F[:, FI["centroid_hz"]], F[:, FI["attack_ms"]], F[:, FI["decay_ms"]]
    tonal, rms = F[:, FI["tonal_db"]], F[:, FI["rms_db"]]
    backbeat = (steps % (steps_per_bar // 2)) == (steps_per_bar // 4)

    low = lo >= 0.35
    midish = ~low & ((mid >= 0.5) | ((cen < 4000.0) & (hi < 0.5)))
    perc = midish & (tonal >= 25.0)
    noisy_mid = midish & ~perc
    hat = ~low & ~midish

    out[low] = "other"
    out[perc] = "perc"
    out[hat & (atk >= 15.0) & (dec < 130.0)] = "shaker"
    out[hat & (dec >= 130.0) & ~((atk >= 15.0) & (dec < 130.0))] = "open_hat"
    rest_hat = hat & np.array([v is None for v in out])
    out[rest_hat] = "closed_hat"

    for g in np.unique(groups):
        idx = np.nonzero((groups == g) & noisy_mid)[0]
        if len(idx) == 0:
            continue
        # the loop's backbeat noisy hits that stand out in the midrange decide clap vs snare
        gi = np.nonzero(groups == g)[0]
        mid_lvl = rms + 10.0 * np.log10(np.maximum(mid, 1e-4))
        med = float(np.median(mid_lvl[gi]))
        bb = idx[backbeat[idx] & (mid_lvl[idx] >= med + 5.0)]
        if len(bb):
            kind = "clap" if float(np.median(atk[bb])) >= 8.0 else "snare"
            out[idx] = kind
        else:
            for i in idx:
                out[i] = "clap" if atk[i] >= 8.0 else "snare"
        # tails: right after a clap/snare and much quieter -> its room tail
        src = idx[backbeat[idx]] if len(bb) else idx
        src_steps = steps[src]
        src_rms = rms[src]
        for i in idx:
            d = steps[i] - src_steps
            near = (d >= 1) & (d <= 3)
            if near.any() and rms[i] <= float(src_rms[near].max()) - 8.0:
                out[i] = TAIL
    return out


def tainted_mask(labels: np.ndarray, features: np.ndarray, steps: np.ndarray, groups: np.ndarray | None = None,
                 tail_steps: int = 3, min_mid: float = 0.45) -> np.ndarray:
    """Hits (of any class) that ring under a clap/snare tail: within `tail_steps` after one in the same
    loop and still midrange-heavy. Used to keep the source pack's clap/snare out of other sounds."""
    labels = np.asarray(labels, dtype=object)
    steps = np.asarray(steps, dtype=np.int64)
    groups = np.zeros(len(labels), np.int64) if groups is None else np.asarray(groups)
    mid = np.asarray(features)[:, FI["band_mid"]]
    out = np.zeros(len(labels), bool)
    for g in np.unique(groups):
        gi = np.nonzero(groups == g)[0]
        src = steps[gi][np.isin(labels[gi], BACKBEAT_SNARE_CLAP)]
        if len(src) == 0:
            continue
        for i in gi:
            if labels[i] in BACKBEAT_SNARE_CLAP:
                continue
            d = steps[i] - src
            if ((d >= 1) & (d <= tail_steps)).any() and mid[i] >= min_mid:
                out[i] = True
    return out


def summarize(labels) -> dict[str, int]:
    """{class: count} in canonical order (tails left out)."""
    labels = list(labels)
    return {c: labels.count(c) for c in CLASSES if labels.count(c)}
