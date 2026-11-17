import json

from loopforge.cli import main
from tests.synth import basic_pattern, render_loop, write


def test_analyze_cli_writes_png_and_json(tmp_path, capsys):
    p = write(tmp_path / "hats_128bpm.wav", render_loop(basic_pattern(2), 128.0, 2))
    out = tmp_path / "out"
    assert main(["analyze", str(p), "--out-dir", str(out), "--json"]) == 0
    txt = capsys.readouterr().out
    assert "128.000 BPM" in txt and "method=filename" in txt and "2 bars (exact loop)" in txt
    png = out / "hats_128bpm.analysis.png"
    assert png.is_file() and png.stat().st_size > 10_000
    data = json.loads((out / "hats_128bpm.analysis.json").read_text())
    assert data["tempo"]["bpm"] == 128.0 and len(data["slices"]) == 32
    assert set(data["slices"][0]["features"]) >= {"centroid_hz", "mfcc_12", "decay_ms"}


def test_analyze_missing_file_returns_error(tmp_path):
    assert main(["analyze", str(tmp_path / "nope.wav"), "--no-png"]) == 1
