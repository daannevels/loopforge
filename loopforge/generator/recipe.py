"""Recipe = everything needed to rebuild a generated loop exactly, plus the renderer.

The renderer is pure DSP: source audio comes from an injected `loader`, so the
same code runs in the CLI, the UI, tests, and (ported) in a plugin.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

RECIPE_VERSION = 1


@dataclass
class RecipeEvent:
    step: int
    sub_offset: float
    role: str
    position: int  # output sample where the slice starts
    length: int  # max output samples to play (trim point incl. overlap)
    source: str  # absolute path at generation time
    source_rel: str  # path relative to the library root
    source_sha1: str
    source_sr: int
    src_start: int  # slice in source samples
    src_end: int
    slice_index: int
    gain_db: float
    fade_in_ms: float
    fade_out_ms: float
    target_db: float  # target peak level from the pattern
    slice_peak_db: float


@dataclass
class Recipe:
    seed: int
    bpm: float
    bars: int
    sr: int
    n_samples: int
    swing: float
    similarity: float
    params: dict
    refs: list[dict]
    library_root: str
    template: str
    roles: list[dict]
    pattern: list[dict]
    events: list[RecipeEvent]
    norm_gain_db: float = 0.0
    limiter_ceiling_db: float = -1.0
    limiter_knee: float = 0.7
    analyzer_version: int = 0
    loopforge_version: str = ""
    recipe_version: int = RECIPE_VERSION
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        d = json.loads(Path(path).read_text())
        d["events"] = [RecipeEvent(**e) for e in d["events"]]
        return cls(**d)


# loader(event, out_sr) -> stereo float32 (2, n) at out_sr, starting at the slice start
SliceLoader = Callable[[RecipeEvent, int], np.ndarray]


def _fade(x: np.ndarray, n_in: int, n_out: int) -> np.ndarray:
    n = x.shape[1]
    if n_in > 1:
        k = min(n, n_in)
        x[:, :k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)
    if n_out > 1:
        k = min(n, n_out)
        x[:, n - k :] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
    return x


def mix_events(recipe: Recipe, loader: SliceLoader) -> np.ndarray:
    """Place all events (trim, fades, gain) and wrap tails into the start. Returns (2, n_samples)."""
    N, sr = recipe.n_samples, recipe.sr
    clips = []
    longest = 0
    for e in recipe.events:
        x = np.array(loader(e, sr)[:, : e.length], dtype=np.float32, copy=True)
        x = _fade(x, int(round(e.fade_in_ms * sr / 1000)), int(round(e.fade_out_ms * sr / 1000)))
        x *= np.float32(10 ** (e.gain_db / 20))
        clips.append((e.position, x))
        longest = max(longest, e.position + x.shape[1])
    buf = np.zeros((2, max(N, longest)), np.float32)
    for pos, x in clips:
        buf[:, pos : pos + x.shape[1]] += x
    out = buf[:, :N].copy()
    for k in range(N, buf.shape[1], N):  # seamless: everything past the end wraps to the start
        chunk = buf[:, k : k + N]
        out[:, : chunk.shape[1]] += chunk
    return out


def soft_limit(x: np.ndarray, ceiling_db: float, knee: float) -> np.ndarray:
    """Pointwise tanh soft-clipper: linear below knee*ceiling, asymptotic to the ceiling."""
    c = 10 ** (ceiling_db / 20)
    t = knee * c
    a = np.abs(x)
    over = a > t
    y = x.copy()
    y[over] = np.sign(x[over]) * (t + (c - t) * np.tanh((a[over] - t) / (c - t)))
    return y.astype(np.float32)


def finalize(mix: np.ndarray, recipe: Recipe) -> np.ndarray:
    y = mix * np.float32(10 ** (recipe.norm_gain_db / 20))
    return soft_limit(y, recipe.limiter_ceiling_db, recipe.limiter_knee)


def render_recipe(recipe: Recipe, loader: SliceLoader) -> np.ndarray:
    return finalize(mix_events(recipe, loader), recipe)
