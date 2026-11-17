"""Library index: recursively scan a folder, analyse every loop once, cache in SQLite.

Re-indexing is incremental:
- size + mtime unchanged (and same analyzer version/params)  -> skipped,
- mtime changed but content hash identical                   -> metadata refresh only,
- new / changed content                                       -> (re)analysed,
- file gone                                                   -> rows removed.
Slice audio is not cached; it is read from the source file when needed.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from loopforge.analysis.loop import analyze_loop
from loopforge.audio_io import file_sha1, is_audio_file, load_audio
from loopforge.config import ANALYZER_VERSION, AnalysisParams
from loopforge.models import N_FEATURES

CACHE_DIRNAME = ".loopforge"
CACHE_FILENAME = "index.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,          -- relative to the library root (posix)
    size INTEGER, mtime_ns INTEGER, sha1 TEXT,
    sr INTEGER, channels INTEGER, n_samples INTEGER,
    bpm REAL, bars INTEGER, is_loop INTEGER, tempo_method TEXT, tempo_conf REAL,
    swing REAL, analyzer_version INTEGER, error TEXT, indexed_at REAL
);
CREATE TABLE IF NOT EXISTS slices (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    idx INTEGER, start INTEGER, "end" INTEGER, step INTEGER, offset REAL, role TEXT,
    features BLOB
);
CREATE INDEX IF NOT EXISTS slices_file ON slices(file_id);
"""


def default_cache_path(root: str | Path) -> Path:
    return Path(root) / CACHE_DIRNAME / CACHE_FILENAME


def _connect(cache: Path) -> sqlite3.Connection:
    cache.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(cache))
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(_SCHEMA)
    return con


def scan_audio_files(root: str | Path) -> list[Path]:
    """Audio files under root, skipping hidden dirs/files (incl. macOS '._' AppleDouble files)."""
    root = Path(root)
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith(".") or not is_audio_file(fn):
                continue
            out.append(Path(dirpath) / fn)
    return out


@dataclass
class IndexReport:
    root: str
    cache: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)  # content changed -> re-analysed
    touched: list[str] = field(default_factory=list)  # mtime changed, same content
    unchanged: int = 0
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def analyzed(self) -> int:
        return len(self.added) + len(self.updated)


@dataclass
class _Job:
    rel: str
    path: str
    size: int
    mtime_ns: int
    sha1: str


def _analyze_job(job: _Job, params: AnalysisParams) -> tuple[_Job, dict | None, str | None]:
    """Worker: analyse one file. Returns (job, result dict, error). Top-level for multiprocessing."""
    try:
        audio, sr = load_audio(job.path)
        a = analyze_loop(audio, sr, name=Path(job.path).name, params=params)
        res = {
            "sr": a.sr, "channels": a.channels, "n_samples": a.n_samples,
            "bpm": a.tempo.bpm, "bars": a.tempo.bars, "is_loop": int(a.tempo.is_loop),
            "tempo_method": a.tempo.method, "tempo_conf": a.tempo.confidence, "swing": a.swing,
            "slices": [
                (s.index, s.start, s.end, s.step, s.offset, s.role, a.features[i].astype(np.float32).tobytes())
                for i, s in enumerate(a.slices)
            ],
        }
        return job, res, None
    except Exception as e:  # noqa: BLE001 - one bad file must not stop indexing
        return job, None, f"{type(e).__name__}: {e}"


def _params_key(params: AnalysisParams) -> str:
    return json.dumps(params.to_dict(), sort_keys=True)


def update_index(
    root: str | Path,
    cache: str | Path | None = None,
    params: AnalysisParams | None = None,
    jobs: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    analyzer_version: int = ANALYZER_VERSION,
) -> IndexReport:
    """Bring the cache for `root` up to date. Returns what changed."""
    t0 = time.time()
    root = Path(root).resolve()
    params = params or AnalysisParams()
    cache_path = Path(cache) if cache else default_cache_path(root)
    con = _connect(cache_path)
    report = IndexReport(root=str(root), cache=str(cache_path))

    meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    full_rebuild = meta.get("analyzer_version") != str(analyzer_version) or meta.get("params") != _params_key(params)

    known = {
        row[0]: row[1:]
        for row in con.execute("SELECT path, id, size, mtime_ns, sha1, analyzer_version, error FROM files")
    }
    present: set[str] = set()
    todo: list[tuple[_Job, bool]] = []  # (job, is_new)

    for path in scan_audio_files(root):
        rel = path.relative_to(root).as_posix()
        present.add(rel)
        st = path.stat()
        rec = known.get(rel)
        if rec is not None and not full_rebuild:
            _id, size, mtime_ns, sha1, ver, err = rec
            if size == st.st_size and mtime_ns == st.st_mtime_ns and err is None:
                report.unchanged += 1
                continue
            digest = file_sha1(path)
            if size == st.st_size and digest == sha1 and err is None:
                con.execute("UPDATE files SET mtime_ns=? WHERE id=?", (st.st_mtime_ns, _id))
                report.touched.append(rel)
                continue
            todo.append((_Job(rel, str(path), st.st_size, st.st_mtime_ns, digest), False))
        else:
            todo.append((_Job(rel, str(path), st.st_size, st.st_mtime_ns, file_sha1(path)), rec is None))

    for rel in sorted(set(known) - present):
        con.execute("DELETE FROM files WHERE path=?", (rel,))
        report.removed.append(rel)

    n_jobs = jobs if jobs is not None else min(8, os.cpu_count() or 1)
    jobs_only = [j for j, _ in todo]
    is_new = {j.rel: new for j, new in todo}
    if n_jobs > 1 and len(jobs_only) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
            results = ex.map(_analyze_job, jobs_only, [params] * len(jobs_only), chunksize=1)
            _store_results(con, results, is_new, report, analyzer_version, progress, len(jobs_only))
    else:
        results = (_analyze_job(j, params) for j in jobs_only)
        _store_results(con, results, is_new, report, analyzer_version, progress, len(jobs_only))

    con.execute("INSERT OR REPLACE INTO meta VALUES ('analyzer_version', ?)", (str(analyzer_version),))
    con.execute("INSERT OR REPLACE INTO meta VALUES ('params', ?)", (_params_key(params),))
    con.execute("INSERT OR REPLACE INTO meta VALUES ('root', ?)", (str(root),))
    con.commit()
    con.close()
    report.seconds = round(time.time() - t0, 2)
    return report


def _store_results(con, results, is_new, report, analyzer_version, progress, total) -> None:
    for i, (job, res, err) in enumerate(results, 1):
        con.execute("DELETE FROM files WHERE path=?", (job.rel,))
        if res is None:
            con.execute(
                "INSERT INTO files (path, size, mtime_ns, sha1, analyzer_version, error, indexed_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (job.rel, job.size, job.mtime_ns, job.sha1, analyzer_version, err, time.time()),
            )
            report.failed[job.rel] = err or "unknown error"
        else:
            cur = con.execute(
                "INSERT INTO files (path, size, mtime_ns, sha1, sr, channels, n_samples, bpm, bars, is_loop, "
                "tempo_method, tempo_conf, swing, analyzer_version, error, indexed_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)",
                (job.rel, job.size, job.mtime_ns, job.sha1, res["sr"], res["channels"], res["n_samples"],
                 res["bpm"], res["bars"], res["is_loop"], res["tempo_method"], res["tempo_conf"], res["swing"],
                 analyzer_version, time.time()),
            )
            fid = cur.lastrowid
            con.executemany(
                'INSERT INTO slices (file_id, idx, start, "end", step, offset, role, features) VALUES (?,?,?,?,?,?,?,?)',
                [(fid, *s) for s in res["slices"]],
            )
            (report.added if is_new[job.rel] else report.updated).append(job.rel)
        if progress:
            progress(i, total, job.rel)
    con.commit()


@dataclass
class SliceTable:
    """All indexed slices as flat arrays (row i = one slice)."""

    root: Path
    features: np.ndarray  # (n, N_FEATURES) float32
    file_idx: np.ndarray  # (n,) index into `files`
    slice_idx: np.ndarray  # (n,) slice index within its file
    start: np.ndarray  # (n,) samples, source sample rate
    end: np.ndarray
    step: np.ndarray
    offset: np.ndarray
    role: list[str]
    files: list[dict]  # per file: path (abs), rel, sha1, sr, channels, bpm, bars, is_loop, swing

    def __len__(self) -> int:
        return len(self.role)

    def path(self, i: int) -> Path:
        return Path(self.files[int(self.file_idx[i])]["path"])

    def sr(self, i: int) -> int:
        return int(self.files[int(self.file_idx[i])]["sr"])

    def label(self, i: int) -> str:
        f = self.files[int(self.file_idx[i])]
        return f"{f['rel']}#{int(self.slice_idx[i])}"

    def mask_files(self, sha1s: set[str]) -> np.ndarray:
        """Boolean mask of slices whose source file content hash is in `sha1s`."""
        bad = np.array([f["sha1"] in sha1s for f in self.files], dtype=bool)
        return bad[self.file_idx] if len(self.file_idx) else np.zeros(0, bool)

    def tainted_mask(self) -> np.ndarray:
        """Slices ringing under a clap/snare tail of their source loop (see analysis/instruments.py).
        Computed once and cached."""
        if getattr(self, "_tainted", None) is None:
            from loopforge.analysis.instruments import tainted_mask

            self._tainted = tainted_mask(np.asarray(self.role, dtype=object), self.features, self.step,
                                         self.file_idx)
        return self._tainted

    def instrument_counts(self) -> dict[str, int]:
        from loopforge.analysis.instruments import summarize

        return summarize(self.role)

    def mask_quiet(self, min_rms_db: float) -> np.ndarray:
        """Boolean mask of slices quieter than `min_rms_db` (near-silent / ghost slices)."""
        from loopforge.models import FEATURE_INDEX

        return self.features[:, FEATURE_INDEX["rms_db"]] < min_rms_db

    def stats(self) -> dict:
        from collections import Counter

        return {
            "files": len(self.files),
            "slices": len(self),
            "roles": dict(Counter(self.role)),
            "instruments": self.instrument_counts(),
            "instrument_loops": {c: len({int(self.file_idx[i]) for i, r in enumerate(self.role) if r == c})
                                 for c in self.instrument_counts()},
            "bpm": dict(Counter(round(f["bpm"], 2) for f in self.files)),
            "sample_rates": dict(Counter(f["sr"] for f in self.files)),
        }


class StaleIndexError(RuntimeError):
    """The cache was built by another analyzer version (e.g. after an update); re-index the library."""


def index_is_current(root: str | Path, cache: str | Path | None = None,
                     analyzer_version: int = ANALYZER_VERSION) -> bool:
    """True if a cache exists and was built by this analyzer version."""
    cache_path = Path(cache) if cache else default_cache_path(Path(root).resolve())
    if not cache_path.is_file():
        return False
    try:
        con = sqlite3.connect(str(cache_path))
        row = con.execute("SELECT value FROM meta WHERE key='analyzer_version'").fetchone()
        con.close()
    except sqlite3.Error:
        return False
    return row is not None and row[0] == str(analyzer_version)


def load_slice_table(root: str | Path, cache: str | Path | None = None) -> SliceTable:
    root = Path(root).resolve()
    cache_path = Path(cache) if cache else default_cache_path(root)
    if not cache_path.is_file():
        raise FileNotFoundError(f"no index at {cache_path}; run `loopforge index {root}` first")
    if not index_is_current(root, cache):
        raise StaleIndexError(f"the index at {cache_path} was made by another loopforge version; "
                              f"re-index with `loopforge index {root}`")
    con = sqlite3.connect(str(cache_path))
    files = []
    fid_to_idx: dict[int, int] = {}
    for fid, rel, sha1, sr, ch, bpm, bars, is_loop, swing in con.execute(
        "SELECT id, path, sha1, sr, channels, bpm, bars, is_loop, swing FROM files WHERE error IS NULL ORDER BY path"
    ):
        fid_to_idx[fid] = len(files)
        files.append({"path": str(root / rel), "rel": rel, "sha1": sha1, "sr": sr, "channels": ch, "bpm": bpm,
                      "bars": bars, "is_loop": bool(is_loop), "swing": swing})
    rows = con.execute(
        'SELECT s.file_id, s.idx, s.start, s."end", s.step, s.offset, s.role, s.features '
        "FROM slices s JOIN files f ON f.id = s.file_id WHERE f.error IS NULL ORDER BY f.path, s.idx"
    ).fetchall()
    con.close()
    n = len(rows)
    feats = np.zeros((n, N_FEATURES), np.float32)
    for i, r in enumerate(rows):
        v = np.frombuffer(r[7], dtype=np.float32)
        if v.shape[0] != N_FEATURES:
            raise StaleIndexError(f"the index at {cache_path} has a different feature layout; re-index the library")
        feats[i] = v
    col = lambda k, dt: np.array([r[k] for r in rows], dtype=dt)  # noqa: E731
    return SliceTable(
        root=root, features=feats,
        file_idx=np.array([fid_to_idx[r[0]] for r in rows], dtype=np.int64),
        slice_idx=col(1, np.int64), start=col(2, np.int64), end=col(3, np.int64),
        step=col(4, np.int64), offset=col(5, np.float64), role=[r[6] for r in rows], files=files,
    )
