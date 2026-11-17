"""All tunable parameters live here, as plain frozen dataclasses.

Engine code receives these objects explicitly (no globals), which keeps the
engine easy to port to C++ later: every dataclass maps to a plain struct.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Bump whenever analysis output changes in a way that invalidates cached
# library indexes (slicing, features, tempo logic).
ANALYZER_VERSION = 2  # v2: loops may be a whole number of beats


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
class Config:
    analysis: AnalysisParams = field(default_factory=AnalysisParams)

    def to_dict(self) -> dict:
        return asdict(self)
