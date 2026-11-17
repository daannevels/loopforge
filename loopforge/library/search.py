"""Timbre space + brute-force nearest-neighbour search over slice features.

Brute force is fast enough here (100k slices x 24 dims = milliseconds per query)
and trivially portable; `NearestNeighbors` is the only thing to swap for a
KD-tree / ANN index later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from loopforge.models import FEATURE_INDEX, FEATURE_NAMES, FEATURE_SPEC, N_FEATURES

# Per-group weight of the *whole group* in the distance (members share it).
DEFAULT_GROUP_WEIGHTS: dict[str, float] = {
    "loudness": 0.0,  # gain is matched separately at assembly
    "class": 0.0,  # instrument-scan helpers (tonal_db): the class label decides, not the distance
    "envelope": 1.0,
    "spectral": 1.0,
    "bands": 0.7,
    "mfcc": 1.0,
}
# Per-feature multipliers on top of the group share.
DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "duration_ms": 0.3,  # mostly reflects pattern density (gap to next hit), not timbre
}
_LOG_FEATURES = ("centroid_hz", "rolloff_hz")
_LOG1P_FEATURES = ("attack_ms", "decay_ms", "duration_ms")
_LOG_EPS_FEATURES = {"flatness": 1e-4}


def transform(X: np.ndarray) -> np.ndarray:
    """Perceptually saner scales: log frequencies/times, log flatness."""
    T = np.array(X, dtype=np.float64, copy=True, ndmin=2)
    for n in _LOG_FEATURES:
        T[:, FEATURE_INDEX[n]] = np.log2(np.maximum(T[:, FEATURE_INDEX[n]], 20.0))
    for n in _LOG1P_FEATURES:
        T[:, FEATURE_INDEX[n]] = np.log1p(np.maximum(T[:, FEATURE_INDEX[n]], 0.0))
    for n, eps in _LOG_EPS_FEATURES.items():
        T[:, FEATURE_INDEX[n]] = np.log(np.maximum(T[:, FEATURE_INDEX[n]], 0.0) + eps)
    return T


@dataclass
class FeatureSpace:
    """z-scored, weighted embedding: Euclidean distance in it = timbre distance."""

    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    group_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        group_weights: dict[str, float] | None = None,
        feature_weights: dict[str, float] | None = None,
    ) -> "FeatureSpace":
        gw = {**DEFAULT_GROUP_WEIGHTS, **(group_weights or {})}
        fw = {**DEFAULT_FEATURE_WEIGHTS, **(feature_weights or {})}
        T = transform(X) if len(X) else np.zeros((1, N_FEATURES))
        mean = T.mean(axis=0)
        std = T.std(axis=0)
        std = np.maximum(std, 1e-3 + 0.05 * np.abs(mean) * (std < 1e-3))
        counts: dict[str, int] = {}
        for _, g in FEATURE_SPEC:
            counts[g] = counts.get(g, 0) + 1
        w = np.array(
            [np.sqrt(gw.get(g, 1.0) / counts[g]) * fw.get(n, 1.0) for n, g in FEATURE_SPEC], dtype=np.float64
        )
        return cls(mean=mean, std=std, weights=w, group_weights=gw)

    def embed(self, X: np.ndarray) -> np.ndarray:
        return ((transform(X) - self.mean) / self.std * self.weights).astype(np.float32)

    def contributions(self, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        """Squared-distance contribution per feature between two raw vectors (for debugging)."""
        d = (self.embed(a) - self.embed(b))[0] ** 2
        return {n: float(v) for n, v in zip(FEATURE_NAMES, d)}


class NearestNeighbors:
    def __init__(self, space: FeatureSpace, X: np.ndarray):
        self.space = space
        self.E = space.embed(X) if len(X) else np.zeros((0, N_FEATURES), np.float32)
        self._sq = (self.E.astype(np.float64) ** 2).sum(axis=1)

    def __len__(self) -> int:
        return len(self.E)

    def distances(self, queries: np.ndarray, embedded: bool = False) -> np.ndarray:
        """(m, n) Euclidean distances from query vectors (raw features, or already embedded) to every slice."""
        Q = (np.atleast_2d(queries) if embedded else self.space.embed(queries)).astype(np.float64)
        d2 = (Q**2).sum(1)[:, None] + self._sq[None, :] - 2.0 * Q @ self.E.T.astype(np.float64)
        return np.sqrt(np.maximum(d2, 0.0))

    def query(
        self, queries: np.ndarray, k: int = 10, exclude: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """k nearest per query. `exclude` = boolean mask over indexed slices. Returns (idx, dist), (m, k')."""
        D = self.distances(np.atleast_2d(queries))
        if exclude is not None and exclude.any():
            D[:, exclude] = np.inf
        k = min(k, int(np.isfinite(D).sum(axis=1).min()) if len(D) else 0)
        if k <= 0:
            return np.zeros((len(D), 0), np.int64), np.zeros((len(D), 0))
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(D, part, axis=1).argsort(axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        return idx, np.take_along_axis(D, idx, axis=1)

    def query_diverse(
        self,
        query: np.ndarray,
        k: int,
        groups: np.ndarray,
        max_per_group: int = 2,
        exclude: np.ndarray | None = None,
        min_separation: float = 0.0,
        embedded: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """k nearest for ONE query, with at most `max_per_group` results per group (e.g. source file)
        and optionally skipping results closer than `min_separation` to an already chosen one
        (near-duplicate repeats of the same hit inside a loop)."""
        D = self.distances(np.atleast_2d(query), embedded=embedded)[0]
        if exclude is not None and exclude.any():
            D[exclude] = np.inf
        chosen: list[int] = []
        per: dict[int, int] = {}
        for j in np.argsort(D, kind="stable"):
            if not np.isfinite(D[j]) or len(chosen) >= k:
                break
            g = int(groups[j])
            if per.get(g, 0) >= max_per_group:
                continue
            if min_separation > 0 and chosen:
                dd = np.sqrt(((self.E[chosen] - self.E[j]) ** 2).sum(axis=1)).min()
                if dd < min_separation:
                    continue
            chosen.append(int(j))
            per[g] = per.get(g, 0) + 1
        idx = np.array(chosen, dtype=np.int64)
        return idx, D[idx]
