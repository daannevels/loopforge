"""Target pattern: a reference template mutated at a rate set by `similarity`."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from loopforge.config import GenParams
from loopforge.generator.profile import RefProfile


@dataclass
class PatternEvent:
    step: int
    role: int
    vel_db: float  # target peak level
    sub_offset: float = 0.0  # flam/roll hit after the step's main hit (in steps)
    origin: str = "template"  # template | mutated | added | shifted | flam

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vel_db"] = round(self.vel_db, 2)
        d["sub_offset"] = round(self.sub_offset, 3)
        return d


def metric_prior(n_steps: int) -> np.ndarray:
    """Generic top-loop hit likelihood: 8th offbeats and beats strongest, then 16ths."""
    w = np.empty(n_steps)
    for o in range(n_steps):
        k = o % 4
        w[o] = {0: 0.9, 2: 1.0, 1: 0.6, 3: 0.7}[k]
    return w


def _sample(rng: np.random.Generator, p: np.ndarray) -> int:
    p = np.asarray(p, dtype=np.float64)
    return int(rng.choice(len(p), p=p / p.sum()))


def generate_pattern(
    profile: RefProfile, similarity: float, rng: np.random.Generator, params: GenParams
) -> tuple[list[PatternEvent], str]:
    """Returns (events sorted by time, name of the base template)."""
    s = float(np.clip(similarity, 0.0, 1.0))
    N = profile.n_steps
    ti = int(rng.integers(len(profile.templates)))
    tmpl = profile.templates[ti]
    prim = {e.step: e for e in tmpl if e.sub_offset == 0}
    subs = [e for e in tmpl if e.sub_offset > 0]

    prior = metric_prior(N)
    prior *= profile.onset_prob.mean() / max(prior.mean(), 1e-9)
    p_mix = np.clip(s * profile.onset_prob + (1 - s) * prior, 0.02, 0.98)
    shares = profile.role_shares
    m = (1.0 - s) * 0.6  # per-step mutation rate

    def role_dist(o: int) -> np.ndarray:
        return s * profile.step_role_prob[o] + (1 - s) * shares

    hits: dict[int, tuple[int, str]] = {}
    for o in range(N):
        if rng.random() < m:
            if rng.random() < p_mix[o]:
                hits[o] = (_sample(rng, role_dist(o)), "mutated")
        elif o in prim:
            role, origin = prim[o].role, "template"
            if rng.random() < m * 0.5:
                role, origin = _sample(rng, role_dist(o)), "mutated"
            hits[o] = (role, origin)

    # occasional shifts by one step
    for o in sorted(hits):
        if o in hits and rng.random() < m * 0.25:
            o2 = (o + (1 if rng.random() < 0.5 else -1)) % N
            if o2 not in hits:
                hits[o2] = (hits.pop(o)[0], "shifted")

    # keep density near the refs
    target = profile.hits_per_template
    tol = 0.05 + 0.3 * (1 - s)
    lo, hi = target * (1 - tol), target * (1 + tol)
    while len(hits) > hi:
        cand = sorted(hits, key=lambda o: p_mix[o] + 1e-3 * rng.random())
        hits.pop(cand[0])
    while len(hits) < lo and len(hits) < N:
        free = [o for o in range(N) if o not in hits]
        o = max(free, key=lambda o: p_mix[o] * rng.random())
        hits[o] = (_sample(rng, role_dist(o)), "added")

    spread = 0.3 + 0.7 * (1 - s)
    events: list[PatternEvent] = []
    for o in sorted(hits):
        r, origin = hits[o]
        role = profile.roles[r]
        vel = role.level_db + profile.accent_db[o] + rng.normal() * role.level_std * spread
        events.append(PatternEvent(o, r, float(min(vel, 0.0)), 0.0, origin))
    for e in subs:
        if e.step in hits and rng.random() < s:
            events.append(PatternEvent(e.step, e.role, e.vel_db, e.sub_offset, "flam"))
    events.sort(key=lambda e: (e.step, e.sub_offset))
    return events, profile.template_names[ti]


def generate_simple_pattern(profile: RefProfile, rng: np.random.Generator) -> tuple[list[PatternEvent], str]:
    """Straight 16ths on every step. Sound type and level per step come from one ref section;
    steps the section leaves empty get the refs' most common sound type for that step."""
    ti = int(rng.integers(len(profile.templates)))
    prim = {e.step: e for e in profile.templates[ti] if e.sub_offset == 0}
    events: list[PatternEvent] = []
    for o in range(profile.n_steps):
        if o in prim:
            e = prim[o]
            events.append(PatternEvent(o, e.role, float(min(e.vel_db, 0.0)), 0.0, "template"))
        else:
            r = int(np.argmax(profile.step_role_prob[o]))
            vel = profile.roles[r].level_db + profile.accent_db[o]
            events.append(PatternEvent(o, r, float(min(vel, 0.0)), 0.0, "filled"))
    return events, profile.template_names[ti]
