import numpy as np
import pytest

from loopforge.analysis.loop import analyze_loop
from tests.synth import SR, Hit, basic_pattern, render_loop, sparse_pattern


def test_steps_match_pattern_exactly():
    hits = basic_pattern(2)
    a = analyze_loop(render_loop(hits, 128.0, 2), SR, name="loop.wav")
    assert [s.step for s in a.slices] == [h.step for h in hits]
    assert a.steps_total == 32


def test_onset_sample_accuracy():
    bpm = 130.0
    steps = [0, 3, 6, 8, 11, 14]
    a = analyze_loop(render_loop(sparse_pattern(steps, "closed"), bpm, 1), SR, name="loop.wav")
    step_samples = 60 / bpm / 4 * SR
    for s, k in zip(a.slices, steps):
        assert abs(s.start - k * step_samples) <= 0.0015 * SR  # within 1.5 ms


def test_hit_at_sample_zero_is_found_with_wrapped_tail():
    # open hat on the last step rings across the loop point into step 0's hit
    hits = [Hit(0, "closed"), Hit(4, "closed"), Hit(8, "closed"), Hit(15, "open", 1.0)]
    a = analyze_loop(render_loop(hits, 140.0, 1), SR, name="loop.wav")
    assert [s.step for s in a.slices] == [0, 4, 8, 15]
    assert a.slices[0].start == 0


@pytest.mark.parametrize("swing", [0.0, 0.1, 0.2, 0.33])
def test_swing_estimate(swing):
    hits = sparse_pattern(list(range(16)), "closed", bars=2)
    a = analyze_loop(render_loop(hits, 124.0, 2, swing=swing), SR, name="loop.wav")
    assert abs(a.swing - swing) < 0.03
    assert [s.step for s in a.slices] == list(range(32))


def test_slices_end_at_next_onset_and_cap_at_one_beat():
    hits = [Hit(0, "open"), Hit(8, "closed")]
    bpm = 120.0
    a = analyze_loop(render_loop(hits, bpm, 1), SR, name="loop.wav")
    beat = int(round(60 / bpm * SR))
    assert len(a.slices) == 2
    assert a.slices[0].end - a.slices[0].start <= beat
    assert a.slices[1].end <= a.n_samples


def test_quiet_noise_produces_no_slices():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(SR * 2) * 1e-4).astype(np.float32)
    a = analyze_loop(x, SR, name="noise.wav")
    assert len(a.slices) <= 2
