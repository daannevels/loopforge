import os
import shutil
import sqlite3

import numpy as np
import pytest

from loopforge.audio_io import file_sha1
from loopforge.cli import main
from loopforge.library import FeatureSpace, NearestNeighbors, default_cache_path, load_slice_table, update_index
from loopforge.library.index import scan_audio_files
from loopforge.library.slices import load_slice_audio
from tests.synth import SR, Hit, basic_pattern, render_loop, sparse_pattern, write


def _lib(root):
    """Small synthetic library with distinct voices, mixed formats, sub-folders and junk files."""
    write(root / "hats" / "closed_128bpm.wav", render_loop(sparse_pattern(list(range(16)), "closed", 2), 128.0, 2))
    write(root / "hats" / "open_8ths.aiff", render_loop(sparse_pattern([2, 6, 10, 14], "open", 2), 128.0, 2))
    write(root / "shakers" / "shaker.flac", render_loop(sparse_pattern(list(range(0, 16, 2)), "shaker", 2), 128.0, 2))
    write(root / "perc" / "perc_44k.wav",
          render_loop(sparse_pattern([0, 3, 6, 10, 13], "perc", 2), 128.0, 2, sr=44100), sr=44100)
    (root / "notes.txt").write_text("not audio")
    (root / "hats" / "._closed_128bpm.wav").write_bytes(b"\x00\x05\x16\x07AppleDouble")
    (root / ".hidden").mkdir()
    write(root / ".hidden" / "skip.wav", render_loop(basic_pattern(), 128.0, 1))
    return root


@pytest.fixture
def lib(tmp_path):
    return _lib(tmp_path / "lib")


def test_scan_skips_hidden_and_non_audio(lib):
    names = sorted(p.name for p in scan_audio_files(lib))
    assert names == ["closed_128bpm.wav", "open_8ths.aiff", "perc_44k.wav", "shaker.flac"]


def test_index_then_incremental(lib):
    r1 = update_index(lib, jobs=1)
    assert len(r1.added) == 4 and r1.analyzed == 4 and not r1.failed
    assert default_cache_path(lib).is_file()

    r2 = update_index(lib, jobs=1)
    assert r2.analyzed == 0 and r2.unchanged == 4

    # touch without content change -> metadata refresh, no re-analysis
    p = lib / "hats" / "closed_128bpm.wav"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    r3 = update_index(lib, jobs=1)
    assert r3.touched == ["hats/closed_128bpm.wav"] and r3.analyzed == 0

    # change content of one file -> only that file re-analysed
    write(lib / "shakers" / "shaker.flac", render_loop(sparse_pattern([0, 4, 8, 12], "shaker", 2), 128.0, 2))
    # add and delete
    write(lib / "new" / "new_loop.wav", render_loop(basic_pattern(1), 128.0, 1))
    (lib / "perc" / "perc_44k.wav").unlink()
    r4 = update_index(lib, jobs=1)
    assert r4.updated == ["shakers/shaker.flac"]
    assert r4.added == ["new/new_loop.wav"]
    assert r4.removed == ["perc/perc_44k.wav"]
    assert r4.unchanged == 2

    t = load_slice_table(lib)
    assert {f["rel"] for f in t.files} == {"hats/closed_128bpm.wav", "hats/open_8ths.aiff", "shakers/shaker.flac",
                                           "new/new_loop.wav"}
    con = sqlite3.connect(default_cache_path(lib))
    n_slices = con.execute("SELECT COUNT(*) FROM slices").fetchone()[0]
    con.close()
    assert n_slices == len(t)  # no orphan slices left behind


def test_analyzer_version_bump_reindexes_everything(lib):
    update_index(lib, jobs=1)
    r = update_index(lib, jobs=1, analyzer_version=999)
    assert r.analyzed == 4
    assert update_index(lib, jobs=1, analyzer_version=999).analyzed == 0


def test_corrupt_file_is_reported_not_fatal(lib):
    (lib / "broken.wav").write_bytes(b"RIFF....WAVEjunk")
    r = update_index(lib, jobs=1)
    assert "broken.wav" in r.failed and r.analyzed == 4
    assert len(load_slice_table(lib).files) == 4


def test_parallel_matches_serial(lib, tmp_path):
    lib2 = tmp_path / "lib2"
    shutil.copytree(lib, lib2)
    update_index(lib, jobs=1)
    update_index(lib2, jobs=2)
    a, b = load_slice_table(lib), load_slice_table(lib2)
    assert len(a) == len(b) and np.allclose(a.features, b.features)


def test_slice_table_contents(lib):
    update_index(lib, jobs=1)
    t = load_slice_table(lib)
    st = t.stats()
    assert st["files"] == 4 and st["sample_rates"] == {48000: 3, 44100: 1}
    # closed: 32 hits, open: 8, shaker: 16, perc: 10
    per_file = {f["rel"]: int((t.file_idx == i).sum()) for i, f in enumerate(t.files)}
    assert per_file == {"hats/closed_128bpm.wav": 32, "hats/open_8ths.aiff": 8, "shakers/shaker.flac": 16,
                        "perc/perc_44k.wav": 10}
    i = int(np.nonzero(t.file_idx == [f["rel"] for f in t.files].index("perc/perc_44k.wav"))[0][0])
    clip, sr = load_slice_audio(t, i, target_sr=48000)
    assert sr == 48000 and clip.shape[0] == 2 and np.abs(clip).max() > 0.1
    assert abs(clip[:, -1]).max() < 1e-3  # faded out


def test_neighbours_find_same_voice(lib):
    update_index(lib, jobs=1)
    t = load_slice_table(lib)
    nn = NearestNeighbors(FeatureSpace.fit(t.features), t.features)
    voice_of = {"hats/closed_128bpm.wav": "closed", "hats/open_8ths.aiff": "open", "shakers/shaker.flac": "shaker",
                "perc/perc_44k.wav": "perc"}
    labels = np.array([voice_of[t.files[i]["rel"]] for i in t.file_idx])
    for v in voice_of.values():
        q = int(np.nonzero(labels == v)[0][0])
        excl = np.zeros(len(t), bool)
        excl[q] = True
        idx, dist = nn.query(t.features[q], k=5, exclude=excl)
        assert (labels[idx[0]] == v).all(), v
        assert np.all(np.diff(dist[0]) >= 0)


def test_gain_does_not_change_neighbours(lib):
    update_index(lib, jobs=1)
    t = load_slice_table(lib)
    nn = NearestNeighbors(FeatureSpace.fit(t.features), t.features)
    from loopforge.models import FEATURE_INDEX as FI

    q = t.features[5].copy()
    q2 = q.copy()
    q2[FI["rms_db"]] -= 12
    q2[FI["peak_db"]] -= 12
    q2[FI["mfcc_0"]] -= 30
    assert np.allclose(nn.distances(q), nn.distances(q2), atol=1e-4)


def test_query_diverse_caps_per_file(lib):
    update_index(lib, jobs=1)
    t = load_slice_table(lib)
    nn = NearestNeighbors(FeatureSpace.fit(t.features), t.features)
    idx, _ = nn.query_diverse(t.features[0], k=4, groups=t.file_idx, max_per_group=1)
    assert len(set(t.file_idx[idx].tolist())) == len(idx) == 4


def test_mask_files_by_hash(lib):
    update_index(lib, jobs=1)
    t = load_slice_table(lib)
    m = t.mask_files({file_sha1(lib / "hats" / "open_8ths.aiff")})
    assert m.sum() == 8


def test_cli_index_and_neighbors_export(lib, tmp_path, capsys):
    assert main(["index", str(lib), "--jobs", "1"]) == 0
    out = capsys.readouterr().out
    assert "4 files" in out and "analysed    4" in out
    ref = write(tmp_path / "ref_128bpm.wav", render_loop(basic_pattern(2), 128.0, 2))
    exp = tmp_path / "exp"
    assert main(["neighbors", str(ref), "--library", str(lib), "--k", "3", "--export", str(exp), "--jobs", "1"]) == 0
    out = capsys.readouterr().out
    assert out.count("query ") >= 3
    wavs = sorted(p.name for p in (exp / "q0").iterdir())
    assert wavs[0] == "00_query.wav" and "chain_query_then_neighbours.wav" in wavs and len(wavs) == 5


def test_cli_neighbors_oneshot_excludes_self(lib, tmp_path, capsys):
    from tests.synth import closed_hat

    update_index(lib, jobs=1)
    shot = write(tmp_path / "hat.wav", closed_hat()[None, :])
    assert main(["neighbors", str(shot), "--library", str(lib), "--k", "3", "--no-update"]) == 0
    out = capsys.readouterr().out
    assert "one-shot, closed" in out
    assert out.count("closed_128bpm.wav") >= 1
