"""Orchestration of one generated loop: pattern -> selection -> recipe -> audio."""

from __future__ import annotations

import numpy as np

from loopforge.config import ANALYZER_VERSION, GenParams
from loopforge.generator.pattern import generate_pattern, generate_simple_pattern
from loopforge.generator.profile import RefProfile
from loopforge.generator.recipe import Recipe, RecipeEvent, SliceLoader, finalize, mix_events
from loopforge.generator.select import (
    RunMemory,
    assign_palette,
    assign_slices,
    build_palettes,
    choose_role_samples,
    role_type,
)
from loopforge.library.index import SliceTable
from loopforge.library.search import NearestNeighbors
from loopforge.models import FEATURE_INDEX

FI = FEATURE_INDEX


def loop_length_samples(bpm: float, bars: int, sr: int, beats_per_bar: int = 4) -> int:
    return int(round(bars * beats_per_bar * 60.0 / bpm * sr))


def selection_mask(table: SliceTable, p: GenParams, exclude_sha1: set[str]) -> np.ndarray:
    """Library slices that may NOT be used: refs' own files, near-silent and too-short slices."""
    m = table.mask_quiet(p.min_slice_rms_db)
    m |= table.features[:, FI["duration_ms"]] < p.min_slice_ms
    if exclude_sha1:
        m |= table.mask_files(exclude_sha1)
    return m


def generate_one(
    profile: RefProfile,
    table: SliceTable,
    nn: NearestNeighbors,
    p: GenParams,
    seed: int,
    exclude: np.ndarray,
    loader: SliceLoader,
    refs_info: list[dict] | None = None,
    version: str = "",
    memory: RunMemory | None = None,
) -> tuple[Recipe, np.ndarray]:
    """One loop. `memory` (simple mode): samples/files used by earlier variations of the run; this
    variation's picks are added to it."""
    rng = np.random.default_rng(seed)
    bpm = float(p.bpm or profile.bpm)
    sr = profile.sr
    N = loop_length_samples(bpm, p.bars, sr)
    step_samples = 60.0 / bpm / 4 * sr
    beat_samples = 4 * step_samples

    simple = p.mode == "simple"
    if simple:
        pattern, template = generate_simple_pattern(profile, rng, keep_flams=p.keep_flams)
        choice = choose_role_samples(profile, pattern, table, nn, rng, p, exclude, memory)
        mixed = {r.id for r in profile.roles if role_type(r.name) in p.mix_types}
        rows = assign_palette(pattern, choice, rng, p, mixed_roles=mixed)
        swing = 0.0
    else:
        pattern, template = generate_pattern(profile, p.similarity, rng, p)
        palettes = build_palettes(profile, pattern, table, nn, p.similarity, rng, p, exclude)
        rows = assign_slices(pattern, palettes, table, rng, p)
        swing = p.swing

    positions = [
        int(round((e.step + (swing if e.step % 2 else 0.0) + e.sub_offset) * step_samples)) % N for e in pattern
    ]
    order = np.argsort(positions, kind="stable")
    overlap = int(round(p.overlap_ms * sr / 1000))
    ring = int(round(p.max_ring_beats * beat_samples))
    events: list[RecipeEvent] = []
    for j, i in enumerate(order):
        e, row, pos = pattern[i], rows[i], positions[i]
        nxt = positions[order[(j + 1) % len(order)]]
        gap = (nxt - pos) % N or N
        role = profile.roles[e.role]
        src_sr = table.sr(row)
        src_len = int(np.ceil((table.end[row] - table.start[row]) * sr / src_sr))
        want = max(gap + overlap, ring) if role.long else gap + overlap
        tail_ms, tail_decay = 0.0, 0.0
        decay = float(table.features[row, FI["decay_ms"]])
        if p.tail_extend and src_len < want:
            # The slice was cut (by the next hit in its source loop) before it could fill its slot here:
            # let it ring on naturally, to about -40 dB (2x its -20 dB decay time), at most `want`.
            natural = int(min(want, src_len + 2.0 * decay * sr / 1000))
            if natural > src_len:
                tail_ms = round((natural - src_len) * 1000.0 / sr, 3)
                tail_decay = round(decay, 3)
                src_len = natural
        length = int(min(want, src_len))
        peak = float(table.features[row, FI["peak_db"]])
        gain = float(np.clip(e.vel_db - peak, -p.gain_clamp_db, p.gain_clamp_db))
        f = table.files[int(table.file_idx[row])]
        events.append(RecipeEvent(
            step=e.step, sub_offset=e.sub_offset, role=role.name, position=pos, length=length,
            source=f["path"], source_rel=f["rel"], source_sha1=f["sha1"], source_sr=src_sr,
            src_start=int(table.start[row]), src_end=int(table.end[row]), slice_index=int(table.slice_idx[row]),
            gain_db=round(gain, 3), fade_in_ms=p.fade_in_ms, fade_out_ms=p.fade_out_ms,
            target_db=round(e.vel_db, 2), slice_peak_db=round(peak, 2),
            tail_ms=tail_ms, tail_decay_ms=tail_decay, tail_seed=int(seed * 1000 + j),
        ))

    recipe = Recipe(
        seed=seed, bpm=bpm, bars=p.bars, sr=sr, n_samples=N, swing=swing, similarity=p.similarity, mode=p.mode,
        params=p.to_dict(), refs=refs_info or [], library_root=str(table.root), template=template,
        roles=[r.to_dict() for r in profile.roles], pattern=[e.to_dict() for e in pattern], events=events,
        limiter_ceiling_db=p.limiter_ceiling_db, limiter_knee=p.limiter_knee,
        analyzer_version=ANALYZER_VERSION, loopforge_version=version,
    )
    mix = mix_events(recipe, loader)
    rms = float(np.sqrt(np.mean(mix.astype(np.float64) ** 2)))
    recipe.norm_gain_db = round(20 * np.log10(profile.ref_rms / max(rms, 1e-9)), 4) if rms > 0 else 0.0
    return recipe, finalize(mix, recipe)
