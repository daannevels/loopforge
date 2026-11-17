"""Reference profile: what the refs have in common, on the output grid.

- roles: clusters of all reference hits in timbre space (k-means, k by silhouette),
- templates: the refs' actual patterns cut/tiled to the output length,
- per-step onset probability, role distribution and accent (dB vs. the role's level).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.vq import kmeans2

from loopforge.analysis.instruments import BACKBEAT_SNARE_CLAP, CLASSES, TAIL
from loopforge.config import GenParams
from loopforge.library.search import FeatureSpace
from loopforge.models import FEATURE_INDEX, LoopAnalysis

SUB_MIN_OFFSET = 0.1  # extra hit on an occupied step counts as a flam/roll if >= this many steps later


@dataclass
class Role:
    id: int
    name: str
    centroid: np.ndarray  # embedded timbre centroid
    mean_features: np.ndarray  # raw feature mean (for diagnostics)
    level_db: float  # mean peak level of hits in this role
    level_std: float
    long: bool  # may ring past the next hit (open hats, long shakers)
    share: float  # fraction of reference hits
    n: int
    radius: float = float("inf")  # 90th-pct distance of the ref's own hits to the centroid (timbre spread)
    is_clap: bool = False  # clap or snare: its room tail rings on; copies of the ref's own are skipped
    instrument: str = ""  # instrument class from the scan (analysis/instruments.py)

    def to_dict(self) -> dict:
        fi = FEATURE_INDEX
        f = self.mean_features
        return {
            "id": self.id, "name": self.name, "share": round(self.share, 3), "n": self.n,
            "level_db": round(self.level_db, 2), "level_std": round(self.level_std, 2), "long": self.long,
            "centroid_hz": round(float(f[fi["centroid_hz"]]), 1), "decay_ms": round(float(f[fi["decay_ms"]]), 1),
            "attack_ms": round(float(f[fi["attack_ms"]]), 1), "band_high": round(float(f[fi["band_high"]]), 3),
            "radius": round(self.radius, 3) if np.isfinite(self.radius) else None, "instrument": self.instrument,
        }


@dataclass
class TemplateEvent:
    step: int
    role: int
    vel_db: float
    sub_offset: float = 0.0  # > 0: flam/roll hit this many steps after the step's main hit


@dataclass
class RefProfile:
    bpm: float
    sr: int
    n_steps: int
    roles: list[Role]
    templates: list[list[TemplateEvent]]
    template_names: list[str]
    onset_prob: np.ndarray  # (n_steps,)
    step_role_prob: np.ndarray  # (n_steps, n_roles)
    accent_db: np.ndarray  # (n_steps,)
    hits_per_template: float
    ref_rms: float  # mean linear RMS of the refs (loudness target)
    ref_swing: list[float] = field(default_factory=list)

    @property
    def role_shares(self) -> np.ndarray:
        return np.array([r.share for r in self.roles])

    def summary(self) -> dict:
        return {
            "bpm": self.bpm, "sr": self.sr, "n_steps": self.n_steps,
            "roles": [r.to_dict() for r in self.roles],
            "templates": self.template_names,
            "hits_per_template": round(self.hits_per_template, 2),
            "onset_prob": [round(float(v), 3) for v in self.onset_prob],
            "accent_db": [round(float(v), 2) for v in self.accent_db],
            "ref_swing": [round(v, 3) for v in self.ref_swing],
        }


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    ks = np.unique(labels)
    if len(ks) < 2:
        return -1.0
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        n_same = same.sum() - 1
        if n_same == 0:
            continue
        a = D[i, same].sum() / n_same
        b = min(D[i, labels == k].mean() for k in ks if k != labels[i])
        s[i] = (b - a) / max(a, b, 1e-12)
    return float(s.mean())


def _fold_small(E: np.ndarray, labels: np.ndarray, min_size: int) -> np.ndarray:
    """Reassign members of clusters smaller than min_size to the nearest remaining centroid."""
    labels = labels.copy()
    while True:
        ks, cnt = np.unique(labels, return_counts=True)
        small = ks[cnt < min_size]
        if len(small) == 0 or len(ks) - len(small) < 1:
            break
        big = ks[cnt >= min_size]
        C = np.stack([E[labels == k].mean(axis=0) for k in big])
        for k in small:
            m = labels == k
            labels[m] = big[np.argmin(((E[m][:, None, :] - C[None]) ** 2).sum(-1), axis=1)]
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def cluster_roles(
    E: np.ndarray, max_k: int, min_sil: float, seed: int = 0, min_share: float = 0.03, rel_sil: float = 0.6,
    n_init: int = 4,
) -> np.ndarray:
    """k-means role labels for embedded reference hits.

    Silhouette alone favours very coarse splits (e.g. bright vs. dull), so we take the
    LARGEST k whose silhouette is within `rel_sil` of the best one. Tiny clusters
    (< min_share of hits, e.g. one odd hit) are folded into their nearest neighbour.
    Returns all zeros (one role) if nothing separates at all.
    """
    n = len(E)
    min_size = max(2, int(np.ceil(min_share * n)))
    results: dict[int, tuple[float, np.ndarray]] = {}
    for k in range(2, min(max_k, n - 1) + 1):
        best: tuple[float, np.ndarray] | None = None
        for i in range(n_init):
            _, lab = kmeans2(E.astype(np.float64), k, seed=np.random.default_rng(seed + 100 * k + i), minit="++")
            lab = _fold_small(E, lab, min_size)
            k_eff = int(lab.max()) + 1
            if k_eff < 2:
                continue
            sil = _silhouette(E, lab)
            if best is None or sil > best[0] + 1e-9:
                best = (sil, lab)
        if best is not None:
            k_eff = int(best[1].max()) + 1
            if k_eff not in results or best[0] > results[k_eff][0]:
                results[k_eff] = best
    if not results:
        return np.zeros(n, np.int64)
    top = max(sil for sil, _ in results.values())
    if top < min_sil:
        return np.zeros(n, np.int64)
    k_pick = max(k for k, (sil, _) in results.items() if sil >= rel_sil * top)
    labels = results[k_pick][1]
    order = np.argsort(-np.bincount(labels), kind="stable")  # largest role = 0
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[labels]


def _chunks(a: LoopAnalysis, n_steps: int) -> list[tuple[str, int, int]]:
    """(label, step_offset, reps) windows of the ref that fill the output length."""
    L = a.steps_total or n_steps
    if L >= n_steps:
        return [(f"{a.name}[{c * n_steps}:{(c + 1) * n_steps}]", c * n_steps, 1) for c in range(L // n_steps)]
    return [(f"{a.name} x{int(np.ceil(n_steps / L))}", 0, int(np.ceil(n_steps / L)))]


def build_profile(
    analyses: list[LoopAnalysis],
    ref_rms: list[float],
    space: FeatureSpace,
    params: GenParams,
    bpm: float | None = None,
) -> RefProfile:
    if not analyses:
        raise ValueError("need at least one reference loop")
    fi = FEATURE_INDEX
    n_steps = params.bars * 16
    bpm = float(bpm or np.median([a.tempo.bpm for a in analyses]))
    srs = [a.sr for a in analyses]
    sr = max(set(srs), key=srs.count)

    # Collect usable hits (primary per step + flam/roll sub-hits). Every hit already carries its
    # instrument from the scan (analysis/instruments.py); each instrument found in the refs becomes one
    # sound type, so a ref with only open + closed hats only ever gets open + closed hats. Hits that are a
    # clap/snare's room tail stay in the rhythm with the instrument clean hits on that step position have;
    # the new clap/snare rings over them like the original did.
    rows: list[tuple[int, int, float, float]] = []  # (ref, slice idx, sub_offset, peak_db)
    feats: list[np.ndarray] = []
    inst_row: list[str] = []
    pos_row: list[int] = []
    for ai, a in enumerate(analyses):
        by_step: dict[int, list[int]] = {}
        for i, s in enumerate(a.slices):
            if a.features[i, fi["rms_db"]] < params.min_slice_rms_db:
                continue
            by_step.setdefault(s.step, []).append(i)
        for step, idx in by_step.items():
            idx.sort(key=lambda i: a.slices[i].start)
            p0 = a.slices[idx[0]].offset
            for j, i in enumerate(idx):
                sub = 0.0 if j == 0 else float(np.clip(a.slices[i].offset - p0, 0.0, 0.95))
                if j > 0 and sub < SUB_MIN_OFFSET:
                    continue
                rows.append((ai, i, sub, float(a.features[i, fi["peak_db"]])))
                feats.append(a.features[i])
                inst_row.append(a.slices[i].role if a.slices[i].role in CLASSES + (TAIL,) else "other")
                pos_row.append(int(a.slices[i].step) % 16)
    if not rows:
        raise ValueError("references contain no usable hits")
    F = np.stack(feats)
    E = space.embed(F)
    inst = np.array(inst_row, dtype=object)
    pos = np.array(pos_row, dtype=np.int64)
    tail = inst == TAIL
    if tail.all():
        inst[:] = "other"
        tail[:] = False
    present = [c for c in CLASSES if (inst == c).any()]
    labels = np.array([present.index(c) if c != TAIL else -1 for c in inst], dtype=np.int64)
    clean = ~tail
    C = np.stack([E[labels == k].mean(axis=0) for k in range(len(present))])
    for i in np.nonzero(tail)[0]:
        # what clean hits on this step position are, excluding the clap/snare itself
        same = clean & (pos == pos[i]) & ~np.isin(inst, BACKBEAT_SNARE_CLAP)
        if same.any():
            labels[i] = int(np.bincount(labels[same], minlength=len(present)).argmax())
        else:
            cand = [k for k, c in enumerate(present) if c not in BACKBEAT_SNARE_CLAP] or list(range(len(present)))
            labels[i] = min(cand, key=lambda k: float(((C[k] - E[i]) ** 2).sum()))
    peaks = np.array([r[3] for r in rows])

    roles: list[Role] = []
    for k, name in enumerate(present):
        m_all = labels == k
        m = m_all & clean if (m_all & clean).any() else m_all  # describe roles by their own clean hits
        mean_f = F[m].mean(axis=0)
        drum = name in BACKBEAT_SNARE_CLAP
        roles.append(Role(
            id=k, name=name, centroid=E[m].mean(axis=0), mean_features=mean_f,
            level_db=float(peaks[m].mean()), level_std=float(peaks[m].std()) if m.sum() > 1 else 1.0,
            long=drum or bool(np.median(F[m, fi["decay_ms"]]) > params.long_role_decay_ms),
            share=float(m_all.mean()), n=int(m_all.sum()),
            radius=float(np.percentile(np.sqrt(((E[m] - E[m].mean(axis=0)) ** 2).sum(axis=1)), 90)) if m.sum() > 1
            else float("inf"),
            is_clap=drum, instrument=name,
        ))

    # Templates on the output grid.
    hit_role = {(r[0], r[1]): int(labels[i]) for i, r in enumerate(rows)}
    hit_sub = {(r[0], r[1]): r[2] for r in rows}
    templates: list[list[TemplateEvent]] = []
    names: list[str] = []
    for ai, a in enumerate(analyses):
        for label, off, reps in _chunks(a, n_steps):
            L = a.steps_total or n_steps
            evs: list[TemplateEvent] = []
            for rep in range(reps):
                for i, s in enumerate(a.slices):
                    if (ai, i) not in hit_role:
                        continue
                    st = s.step - off + rep * L
                    if 0 <= st < n_steps:
                        evs.append(TemplateEvent(st, hit_role[(ai, i)], float(a.features[i, fi["peak_db"]]),
                                                 hit_sub[(ai, i)]))
            if evs:
                templates.append(sorted(evs, key=lambda e: (e.step, e.sub_offset)))
                names.append(label)

    onset = np.zeros(n_steps)
    role_counts = np.zeros((n_steps, len(roles)))
    accent_sum = np.zeros(n_steps)
    accent_n = np.zeros(n_steps)
    for t in templates:
        for e in t:
            if e.sub_offset > 0:
                continue
            onset[e.step] += 1
            role_counts[e.step, e.role] += 1
            accent_sum[e.step] += e.vel_db - roles[e.role].level_db
            accent_n[e.step] += 1
    nt = max(1, len(templates))
    onset /= nt
    shares = np.array([r.share for r in roles])
    step_role = (role_counts + 0.5 * shares[None, :]) / (role_counts.sum(1, keepdims=True) + 0.5)
    accent = np.where(accent_n > 0, accent_sum / np.maximum(accent_n, 1), 0.0)
    hits = float(np.mean([sum(1 for e in t if e.sub_offset == 0) for t in templates]))

    return RefProfile(
        bpm=bpm, sr=sr, n_steps=n_steps, roles=roles, templates=templates, template_names=names,
        onset_prob=onset, step_role_prob=step_role, accent_db=accent, hits_per_template=hits,
        ref_rms=float(np.mean(ref_rms)), ref_swing=[a.swing for a in analyses],
    )
