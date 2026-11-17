"""Slice selection: per-role palettes from the library, then per-event assignment.

Each role gets a small palette (main sound + variations) drawn at random from its
nearest library neighbours; the candidate pool widens as similarity drops.
Constraints: no slice twice in a row, at most `max_file_share` of events per file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loopforge.config import GenParams
from loopforge.generator.pattern import PatternEvent
from loopforge.generator.profile import RefProfile
from loopforge.library.index import SliceTable
from loopforge.library.search import NearestNeighbors


@dataclass
class Palette:
    role: int
    rows: list[int]  # palette (rank order: main sound first)
    dists: list[float]
    backups: list[int]  # further candidates, used when constraints block the palette


def fit_penalty(table: SliceTable, rows: np.ndarray, role, p: GenParams) -> np.ndarray:
    """Extra distance for candidates that would need a lot of gain, or are too short for a ringing role."""
    from loopforge.models import FEATURE_INDEX as FI

    peak = table.features[rows, FI["peak_db"]]
    gain_needed = np.abs(role.level_db - peak)
    pen = p.level_penalty * np.maximum(0.0, gain_needed - p.level_free_db)
    if role.long:
        want = float(role.mean_features[FI["decay_ms"]])
        dur = table.features[rows, FI["duration_ms"]]
        pen += p.short_penalty * np.clip(1.0 - dur / max(want, 1.0), 0.0, 1.0)
    return pen


def candidate_k(similarity: float, p: GenParams) -> int:
    return int(round(p.k_min + (1 - np.clip(similarity, 0, 1)) * (p.k_max - p.k_min)))


def build_palettes(
    profile: RefProfile,
    events: list[PatternEvent],
    table: SliceTable,
    nn: NearestNeighbors,
    similarity: float,
    rng: np.random.Generator,
    p: GenParams,
    exclude: np.ndarray,
) -> dict[int, Palette]:
    k = candidate_k(similarity, p)
    n_events = len(events)
    cap = max(1.0, p.max_file_share * n_events)
    expected_use: dict[int, float] = {}
    counts = {r.id: sum(1 for e in events if e.role == r.id) for r in profile.roles}
    palettes: dict[int, Palette] = {}
    tau = 0.15 + 0.6 * (1 - similarity)
    for role in sorted(profile.roles, key=lambda r: -counts[r.id]):
        if counts[role.id] == 0:
            continue
        idx, dist = nn.query_diverse(role.centroid, 3 * k + p.palette_size + 4, table.file_idx,
                                     max_per_group=p.candidates_per_file, exclude=exclude,
                                     min_separation=0.05, embedded=True)
        if len(idx) == 0:
            raise RuntimeError(f"no library candidates for role {role.name}")
        adj = dist + fit_penalty(table, idx, role, p)
        keep = np.argsort(adj, kind="stable")[: k + p.palette_size]
        idx, dist = idx[keep], adj[keep]
        pool, pdist = list(idx[:k]), list(dist[:k])
        w = np.exp(-(np.array(pdist) - pdist[0]) / tau)
        order = list(rng.choice(len(pool), size=len(pool), replace=False, p=w / w.sum()))
        n_pal = min(p.palette_size, len(pool))
        share_each = counts[role.id] / n_pal
        chosen: list[int] = []
        for oi in order:
            row = int(pool[oi])
            f = int(table.file_idx[row])
            if expected_use.get(f, 0.0) + share_each > cap and len(order) - len(chosen) > n_pal:
                continue
            chosen.append(oi)
            expected_use[f] = expected_use.get(f, 0.0) + share_each
            if len(chosen) == n_pal:
                break
        for oi in order:  # fill up if the cap was too strict
            if len(chosen) >= n_pal:
                break
            if oi not in chosen:
                chosen.append(oi)
        chosen.sort(key=lambda oi: pdist[oi])  # main sound = closest
        rows = [int(pool[oi]) for oi in chosen]
        backups = [int(r) for r in idx if int(r) not in rows]
        palettes[role.id] = Palette(role.id, rows, [float(pdist[oi]) for oi in chosen], backups)
    return palettes


def assign_slices(
    events: list[PatternEvent],
    palettes: dict[int, Palette],
    table: SliceTable,
    rng: np.random.Generator,
    p: GenParams,
) -> list[int]:
    """Library row for each event (same order as `events`)."""
    cap = max(1, int(np.floor(p.max_file_share * len(events))))
    used: dict[int, int] = {}
    out: list[int] = []
    prev = -1
    for e in events:
        pal = palettes[e.role]
        w = np.array([p.palette_weights[min(i, len(p.palette_weights) - 1)] for i in range(len(pal.rows))])
        cands = list(pal.rows)
        ok = [i for i, r in enumerate(cands) if r != prev and used.get(int(table.file_idx[r]), 0) < cap]
        if ok:
            ww = w[ok] / w[ok].sum()
            row = cands[ok[int(rng.choice(len(ok), p=ww))]]
        else:
            row = next((r for r in pal.backups if r != prev and used.get(int(table.file_idx[r]), 0) < cap),
                       next((r for r in cands + pal.backups if r != prev), cands[0]))
        out.append(int(row))
        f = int(table.file_idx[row])
        used[f] = used.get(f, 0) + 1
        prev = row
    return out
