import numpy as np
import pytest

from loopforge.analysis.loop import analyze_loop
from loopforge.analysis.tempo import bpm_from_name
from loopforge.config import AnalysisParams
from tests.synth import SR, Hit, basic_pattern, render_loop


@pytest.mark.parametrize("bpm", [100.0, 112.0, 124.0, 133.3, 140.0, 145.0, 150.0])
@pytest.mark.parametrize("bars", [1, 2, 4])
def test_tempo_from_duration(bpm, bars):
    x = render_loop(basic_pattern(bars), bpm, bars)
    a = analyze_loop(x, SR, name="loop.wav")
    assert a.tempo.method == "duration"
    assert a.tempo.is_loop and a.tempo.bars == bars
    assert abs(a.tempo.bpm - bpm) < 0.5
    assert a.tempo.confidence > 0.9


@pytest.mark.parametrize(
    "name,expected",
    [
        ("TPS - Ignite - Top Loop 01 145 BPM.wav", 145.0),
        ("hat_loop_124bpm.wav", 124.0),
        ("Shaker 128 bpm A.aiff", 128.0),
        ("perc_BPM_132_x.wav", 132.0),
        ("Top 01 Am.wav", None),
        ("909 hats.wav", None),
    ],
)
def test_bpm_from_name(name, expected):
    assert bpm_from_name(name) == expected


def test_filename_wins_and_outside_range_allowed():
    # 90 BPM is outside the default range: duration fit would pick 2 bars @ 180? No — the filename says 90.
    x = render_loop(basic_pattern(1), 90.0, 1)
    a = analyze_loop(x, SR, name="perc_90bpm.wav")
    assert a.tempo.method == "filename" and a.tempo.bpm == 90.0 and a.tempo.is_loop


def test_manual_override():
    x = render_loop(basic_pattern(2), 126.0, 2)
    a = analyze_loop(x, SR, name="loop_140bpm.wav", bpm=126.0)
    assert a.tempo.method == "manual" and a.tempo.bars == 2 and a.tempo.is_loop


def test_misleading_filename_is_rejected():
    x = render_loop(basic_pattern(2), 126.0, 2)
    a = analyze_loop(x, SR, name="loop_137bpm.wav")
    assert a.tempo.method == "duration" and abs(a.tempo.bpm - 126.0) < 0.05


def test_round_bpm_snapping():
    x = render_loop(basic_pattern(4), 145.0, 4)
    x = np.concatenate([x, x[:, :1]], axis=1)  # one extra sample, like many exports
    a = analyze_loop(x, SR, name="loop.wav")
    assert a.tempo.bpm == 145.0


@pytest.mark.parametrize("bpm", [104.0, 127.0, 146.0])
def test_non_loop_grid_search(bpm):
    hits = basic_pattern(4)
    x = render_loop(hits, bpm, 4, circular=False, pre_silence_s=0.173, tail_s=0.31)
    a = analyze_loop(x, SR, name="cut.wav")
    assert a.tempo.method == "autocorr" and not a.tempo.is_loop
    assert abs(a.tempo.bpm - bpm) < 0.3
    step_s = 60 / bpm / 4
    # phase is the grid origin modulo two steps (keeps even/odd parity)
    assert abs(((a.tempo.phase_s - 0.173) / step_s + 1) % 2 - 1) < 0.05


def test_custom_range():
    x = render_loop(basic_pattern(2), 87.0, 2)
    p = AnalysisParams(bpm_min=80.0, bpm_max=115.0)
    a = analyze_loop(x, SR, name="loop.wav", params=p)
    assert abs(a.tempo.bpm - 87.0) < 0.5
