"""All tunable parameters live here, as plain frozen dataclasses.

Engine code receives these objects explicitly (no globals), which keeps the
engine easy to port to C++ later: every dataclass maps to a plain struct.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Bump whenever analysis output changes in a way that invalidates cached
# library indexes (slicing, features, tempo logic).
ANALYZER_VERSION = 3  # v3: tonal_db feature + instrument scan labels


@dataclass(frozen=True)
class AnalysisParams:
    # Tempo / grid
    bpm_min: float = 100.0
    bpm_max: float = 150.0
    beats_per_bar: int = 4
    steps_per_beat: int = 4  # 16th notes
    candidate_bars: tuple[int, ...] = (1, 2, 4, 8, 16)
    # A file counts as a trimmed loop if its length is within this many
    # seconds (or this fraction of a bar, whichever is larger) of N whole bars.
    loop_tolerance_s: float = 0.02
    loop_tolerance_bar_frac: float = 0.01
    # Minimum grid-alignment score to accept a duration-derived tempo.
    min_grid_score: float = 0.6

    # Onsets
    hop_length: int = 256
    onset_delta: float = 0.1  # peak-picking threshold on normalised, compressed flux
    onset_compression: float = 10.0  # log1p(c * flux): higher = more sensitive to quiet hits
    min_onset_gap_s: float = 0.025
    onset_gate_db: float = -45.0  # drop hits this far below the loop peak
    onset_abs_gate_db: float = -60.0  # and anything quieter than this (dBFS)
    circular_pad_s: float = 0.25  # wrap this much loop tail before t=0
    preroll_s: float = 0.0005  # keep this much before the detected attack

    # Slicing
    max_slice_beats: float = 1.0

    # Features
    n_mfcc: int = 13
    mfcc_fmax: float = 16000.0  # fixed so 44.1k and 48k files are comparable
    feature_win_s: float = 0.0213  # ~1024 samples at 48k
    band_edges_hz: tuple[float, float] = (250.0, 4000.0)
    # Spectral features ignore content above this so 44.1k and 48k files compare fairly.
    spectral_fmax: float = 20000.0
    rolloff_pct: float = 0.85

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GenParams:
    bars: int = 2
    similarity: float = 1.0  # creative mode only: 1 = keep the refs' rhythm, 0 = loose
    swing: float = 0.0  # output swing (delay of odd 16ths, in steps); straight by default
    bpm: float | None = None  # override output tempo (default: refs' tempo)
    # "simple": straight 16ths, ref rhythm + sound type per step kept, one sample per sound type
    # "creative": mutated rhythm, palettes, flams, swing (M3 behaviour)
    mode: str = "simple"
    keep_flams: bool = True  # simple mode: keep the ref's flams/rolls (part of its rhythm)
    simple_pool_max: int = 150  # simple mode: max same-type candidates per sound type
    simple_per_file: int = 4  # ... and at most this many from one source file
    clap_copy_threshold: float = 0.85  # library claps this similar (log-spectrum corr) to a ref clap are skipped
    diversity_top_frac: float = 0.2  # later variations pick randomly among the 20% most-different candidates
    simple_decay_range: tuple[float, float] = (0.6, 2.0)  # candidate decay / ref sound decay
    simple_radius_scale: float = 2.5  # candidates within this x the spread of the ref's own hits
    simple_radius_floor: float = 0.3  # minimum spread (embedding units) for very consistent ref sounds
    tail_extend: bool = True  # rebuild tails of ringing hits that were cut short in their source loop

    # Roles (clusters of reference hits)
    max_roles: int = 6
    min_silhouette: float = 0.15  # below this, all ref hits form one role
    role_rel_silhouette: float = 0.6  # pick the largest k with silhouette >= this * best
    role_min_share: float = 0.03  # smaller clusters are folded into their nearest role

    # Timbre space for role clustering + selection: decay/attack matter more than in the
    # generic search space (closed vs. open hats), MFCC detail less.
    timbre_group_weights: tuple[tuple[str, float], ...] = (("envelope", 3.0), ("mfcc", 0.5))

    # Selection
    palette_size: int = 1  # library samples per sound type (1 = one open hat, one closed hat, ...)
    palette_weights: tuple[float, ...] = (0.55, 0.3, 0.15, 0.1)  # pick probability by palette rank
    k_min: int = 4  # candidate pool per role at similarity 1
    k_max: int = 32  # ... at similarity 0
    candidates_per_file: int = 2
    max_file_share: float = 0.4  # max fraction of events from one source file
    level_free_db: float = 6.0  # gain needed up to this is free ...
    level_penalty: float = 0.05  # ... beyond it, distance += this per dB
    short_penalty: float = 1.0  # ringing roles: distance += this * (1 - slice_dur / role_decay)
    min_slice_rms_db: float = -50.0
    min_slice_ms: float = 15.0

    # Assembly
    gain_clamp_db: float = 12.0
    fade_in_ms: float = 1.0
    fade_out_ms: float = 10.0
    overlap_ms: float = 10.0  # hits ring this far into the next hit (fade-out underneath it)
    long_role_decay_ms: float = 120.0  # roles with longer decay may ring past the next hit
    max_ring_beats: float = 1.0
    limiter_ceiling_db: float = -1.0
    limiter_knee: float = 0.7  # soft-limit above this fraction of the ceiling

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Config:
    analysis: AnalysisParams = field(default_factory=AnalysisParams)
    gen: GenParams = field(default_factory=GenParams)

    def to_dict(self) -> dict:
        return asdict(self)
