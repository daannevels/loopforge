"""Plain data types shared by analysis, library and generator.

Kept free of behaviour beyond trivial helpers so they map 1:1 onto C++ structs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# --- Feature vector layout -------------------------------------------------
# (name, group). Groups are used for weighting during similarity search.
FEATURE_SPEC: tuple[tuple[str, str], ...] = (
    ("rms_db", "loudness"),
    ("peak_db", "loudness"),
    ("attack_ms", "envelope"),
    ("decay_ms", "envelope"),
    ("duration_ms", "envelope"),
    ("centroid_hz", "spectral"),
    ("rolloff_hz", "spectral"),
    ("flatness", "spectral"),
    ("band_low", "bands"),
    ("band_mid", "bands"),
    ("band_high", "bands"),
    ("tonal_db", "class"),  # pitch-peak strength 100-1200 Hz (tonal percs); used by the instrument scan only
    ("mfcc_0", "loudness"),  # c0 is log energy: gain-dependent, so not part of timbre
    *((f"mfcc_{i}", "mfcc") for i in range(1, 13)),
)
FEATURE_NAMES: tuple[str, ...] = tuple(n for n, _ in FEATURE_SPEC)
FEATURE_INDEX: dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class TempoEstimate:
    bpm: float
    bars: int  # whole bars covered by the file (loop length if is_loop)
    method: str  # "manual" | "filename" | "duration" | "autocorr"
    confidence: float  # 0..1 (grid alignment score)
    is_loop: bool  # file length is an exact number of bars
    phase_s: float = 0.0  # time of the first downbeat
    beats: int = 0  # whole beats covered (loop length in beats if is_loop; may be < bars * 4)

    def step_len_s(self, steps_per_beat: int = 4) -> float:
        return 60.0 / self.bpm / steps_per_beat


@dataclass
class Slice:
    index: int
    start: int  # sample index (attack start incl. pre-roll)
    end: int  # exclusive sample index
    step: int  # grid step (16th) within the file, modulo loop length if looped
    offset: float  # micro-timing vs. the straight grid, in steps (+ = late)
    role: str = ""  # heuristic description, e.g. "closed", "open", "shaker"

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class LoopAnalysis:
    name: str
    sr: int
    n_samples: int
    channels: int
    tempo: TempoEstimate
    swing: float  # median delay of odd 16ths, in steps (0 = straight)
    grid_offset: float  # median offset of even 16ths, in steps (latency)
    slices: list[Slice]
    features: np.ndarray  # (n_slices, N_FEATURES) float32
    steps_total: int = 0  # bars * steps_per_bar
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sr

    def feature(self, name: str) -> np.ndarray:
        return self.features[:, FEATURE_INDEX[name]]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sr": self.sr,
            "n_samples": self.n_samples,
            "channels": self.channels,
            "duration_s": round(self.duration_s, 6),
            "tempo": asdict(self.tempo),
            "swing": round(self.swing, 4),
            "grid_offset": round(self.grid_offset, 4),
            "steps_total": self.steps_total,
            "notes": self.notes,
            "slices": [
                {
                    **asdict(s),
                    "features": {
                        n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, self.features[i])
                    },
                }
                for i, s in enumerate(self.slices)
            ],
        }
