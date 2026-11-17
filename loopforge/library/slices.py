"""Reading slice audio back from library source files."""

from __future__ import annotations

import numpy as np

from loopforge.audio_io import apply_fades, load_region, resample, to_stereo
from loopforge.library.index import SliceTable


def load_slice_audio(
    table: SliceTable, i: int, target_sr: int | None = None, fade_out_s: float = 0.005
) -> tuple[np.ndarray, int]:
    """Stereo float32 (2, n) audio of slice row `i`, optionally resampled, with a short fade-out."""
    x, sr = load_region(table.path(i), int(table.start[i]), int(table.end[i]))
    x = to_stereo(x)
    if target_sr and target_sr != sr:
        x, sr = resample(x, sr, target_sr), target_sr
    return apply_fades(x, sr, 0.0, fade_out_s), sr


def audition_chain(clips: list[np.ndarray], sr: int, gap_s: float = 0.25) -> np.ndarray:
    """Concatenate stereo clips with silence in between, for quick A/B listening."""
    gap = np.zeros((2, int(gap_s * sr)), np.float32)
    parts: list[np.ndarray] = []
    for c in clips:
        parts += [to_stereo(c), gap]
    return np.concatenate(parts, axis=1) if parts else np.zeros((2, 0), np.float32)


def recipe_loader(base_dir: str | None = None):
    """SliceLoader for recipes: reads the event's source region, stereo, resampled to out_sr.

    If `base_dir` is given, sources are resolved as base_dir/source_rel (library moved);
    otherwise the absolute `source` path is used. Regions are cached per call site.
    """
    from pathlib import Path

    cache: dict[tuple[str, int, int, int], np.ndarray] = {}

    def load(ev, out_sr: int) -> np.ndarray:
        path = str(Path(base_dir) / ev.source_rel) if base_dir else ev.source
        key = (path, ev.src_start, ev.src_end, out_sr)
        if key not in cache:
            x, sr = load_region(path, ev.src_start, ev.src_end)
            x = to_stereo(x)
            cache[key] = resample(x, sr, out_sr) if sr != out_sr else x
        return cache[key]

    return load
