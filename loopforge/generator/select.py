"""Slice selection: per-role palettes from the library, then per-event assignment.

Each role gets a small palette (main sound + variations) drawn at random from its
nearest library neighbours; the candidate pool widens as similarity drops.
Constraints: no slice twice in a row, at most `max_file_share` of events per file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class RunMemory:
    """What earlier variations of the same run used, so later ones pick different sounds."""

    rows: set[int]
    files: set[int]
    by_role: dict[int, list[int]] = field(default_factory=dict)  # samples earlier variations used, per type

    @classmethod
    def new(cls) -> "RunMemory":
        return cls(set(), set(), {})


def same_type_candidates(profile: RefProfile, role_id: int, table: SliceTable, nn: NearestNeighbors,
                         exclude: np.ndarray, p: GenParams) -> np.ndarray:
    """Library rows of the SAME instrument as reference role `role_id` (instrument scan labels: a snare is
    only replaced by a snare, an open hat by an open hat, ...), usable at the ref's level (gain within the
    clamp) and, for hats/shakers/percs, ringing about as long as the ref's sound (a fast-dying "open hat"
    sounds choppy). Never a slice that carries a clap/snare tail from its source loop. Closest first."""
    from loopforge.models import FEATURE_INDEX as FI

    role = profile.roles[role_id]
    own = nn.distances(role.centroid, embedded=True)[0]
    own[exclude] = np.inf
    labels = np.asarray(table.role, dtype=object)
    tainted = table.tainted_mask()
    same = labels == role.instrument
    gain_ok = np.abs(role.level_db - table.features[:, FI["peak_db"]]) <= p.gain_clamp_db
    ref_decay = float(role.mean_features[FI["decay_ms"]])
    ratio = table.features[:, FI["decay_ms"]] / max(ref_decay, 1.0)
    decay_ok = (ratio >= p.simple_decay_range[0]) & (ratio <= p.simple_decay_range[1])
    if role.is_clap:  # clap/snare: a *different* one is the point, its tail is synthesized anyway
        decay_ok = np.ones_like(decay_ok)
    base = np.isfinite(own) & ~tainted
    rows = np.array([], dtype=np.int64)
    for ok in (base & same & gain_ok & decay_ok, base & same & gain_ok, base & same):  # relax step by step
        rows = np.nonzero(ok)[0]
        if len(rows) >= p.palette_size:
            break
    if len(rows) == 0:
        # the library has no such instrument at all: closest sounds that are not another drum type
        not_drum = ~np.isin(labels, ("clap", "snare", "tail")) if not role.is_clap else ~np.isin(labels, ("tail",))
        rows = np.nonzero(base & not_drum)[0]
        rows = rows[np.argsort(own[rows], kind="stable")[: max(p.palette_size, 4)]]
    rows = rows[np.argsort(own[rows], kind="stable")]
    return _cap_per_file(rows, table, p)


def _cap_per_file(rows: np.ndarray, table: SliceTable, p: GenParams) -> np.ndarray:
    """Loops repeat the same hit many times: keep at most `simple_per_file` per source file."""
    per_file: dict[int, int] = {}
    keep = []
    for x in rows:
        f = int(table.file_idx[x])
        if per_file.get(f, 0) < p.simple_per_file:
            per_file[f] = per_file.get(f, 0) + 1
            keep.append(int(x))
    return np.array(keep[: p.simple_pool_max], dtype=np.int64)


def choose_role_samples(
    profile: RefProfile,
    events: list[PatternEvent],
    table: SliceTable,
    nn: NearestNeighbors,
    rng: np.random.Generator,
    p: GenParams,
    exclude: np.ndarray,
    memory: RunMemory | None = None,
) -> dict[int, list[int]]:
    """Simple mode: `palette_size` library slices per sound type, picked at random among same-type
    candidates. Preference order: sample AND source file unused by earlier variations > sample unused >
    anything. Within one loop, different sound types never share a sample or (if avoidable) a file."""
    memory = memory or RunMemory.new()
    taken_rows: set[int] = set()
    taken_files: set[int] = set()
    out: dict[int, list[int]] = {}
    for r in sorted({e.role for e in events}):
        pool = [int(x) for x in same_type_candidates(profile, r, table, nn, exclude, p)]
        picks: list[int] = []
        for _ in range(min(p.palette_size, len(pool))):
            def f(row: int) -> int:
                return int(table.file_idx[row])

            tiers = [
                [x for x in pool if x not in picks and x not in taken_rows and x not in memory.rows
                 and f(x) not in memory.files and f(x) not in taken_files],
                [x for x in pool if x not in picks and x not in taken_rows and x not in memory.rows],
                [x for x in pool if x not in picks and x not in taken_rows],
                [x for x in pool if x not in picks],
            ]
            cand = next((t for t in tiers if t), [])
            if not cand:
                break
            row = _pick_most_different(cand, memory.by_role.get(r, []) + picks, nn, rng, p)
            picks.append(row)
            taken_rows.add(row)
            taken_files.add(f(row))
        out[r] = picks
    for r, rows in out.items():
        memory.rows.update(rows)
        memory.files.update(int(table.file_idx[x]) for x in rows)
        memory.by_role.setdefault(r, []).extend(rows)
    return out


def _pick_most_different(cand: list[int], used: list[int], nn: NearestNeighbors, rng: np.random.Generator,
                         p: GenParams) -> int:
    """Random pick among the candidates that sound MOST different from the samples already used for this
    sound type (largest distance to the nearest used one). With nothing used yet: uniformly random."""
    if not used:
        return cand[int(rng.integers(len(cand)))]
    C = nn.E[cand].astype(np.float64)
    U = nn.E[used].astype(np.float64)
    novelty = np.sqrt(((C[:, None, :] - U[None, :, :]) ** 2).sum(-1)).min(axis=1)
    k = max(1, int(np.ceil(len(cand) * p.diversity_top_frac)))
    top = np.argsort(-novelty, kind="stable")[:k]
    return cand[int(top[int(rng.integers(len(top)))])]


def assign_palette(events: list[PatternEvent], choice: dict[int, list[int]], rng: np.random.Generator,
                   p: GenParams) -> list[int]:
    """Row per event: palette rank 0 is the main sound; with a palette > 1 the others are mixed in
    (weights `palette_weights`). Repeats are allowed, like a drum machine."""
    out = []
    for e in events:
        pal = choice[e.role]
        if len(pal) == 1:
            out.append(pal[0])
            continue
        w = np.array([p.palette_weights[min(i, len(p.palette_weights) - 1)] for i in range(len(pal))])
        out.append(pal[int(rng.choice(len(pal), p=w / w.sum()))])
    return out
