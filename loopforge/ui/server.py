"""Local web UI server (stdlib only). Binds to 127.0.0.1; one generation job at a time.

The front-end (static/) talks to a small JSON API. Long work (indexing, generating)
runs in a background thread; the page polls /api/state.
"""

from __future__ import annotations

import json
import mimetypes
import os
import platform
import random
import re
import shutil
import subprocess
import threading
import time
import traceback
import webbrowser
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from loopforge import __version__
from loopforge.audio_io import is_audio_file
from loopforge.config import AnalysisParams, GenParams
from loopforge.service import GenRequest, RefLoop, load_ref, run_generation, tempo_conflict, waveform_peaks

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 300 * 1024 * 1024
MAX_REFS = 3
SETTING_KEYS = ("bars", "variations", "seed", "seed_lock", "mix_types", "mix_size", "bpm")
MIX_TYPES = ("closed", "open", "shaker", "perc", "clap")


def default_workspace() -> Path:
    return Path(os.environ.get("LOOPFORGE_HOME", Path.home() / "Loopforge"))


def _safe_name(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\- ()\[\]#&+,]", "_", name).strip() or "upload.wav"
    return name


class App:
    """All UI state. Methods are called from request threads; `lock` guards mutation."""

    def __init__(self, workspace: Path, aparams: AnalysisParams | None = None, jobs: int | None = None):
        self.ws = Path(workspace).expanduser().resolve()
        for d in ("refs", "runs", "favorites"):
            (self.ws / d).mkdir(parents=True, exist_ok=True)
        self.aparams = aparams or AnalysisParams()
        self.jobs = jobs
        self.lock = threading.RLock()
        self.refs: dict[str, RefLoop] = {}
        self.ref_order: list[str] = []
        self.settings: dict = {"bars": 2, "variations": 8, "seed": 42, "seed_lock": False, "mix_types": ["perc"],
                               "mix_size": 3, "bpm": None, "library": None}
        self.index = {"state": "none", "done": 0, "total": 0, "current": "", "stats": None, "error": None}
        self.job = {"state": "idle", "log": [], "rows": [], "run": None, "error": None, "total": 0, "started": None}
        self._stop = False
        self._load_settings()
        self._load_last_run()

    # ----- persistence -----
    @property
    def settings_path(self) -> Path:
        return self.ws / "settings.json"

    def _save_settings(self) -> None:
        data = {**self.settings, "refs": [str(self.refs[i].path) for i in self.ref_order]}
        self.settings_path.write_text(json.dumps(data, indent=1))

    def _load_settings(self) -> None:
        if not self.settings_path.is_file():
            return
        try:
            data = json.loads(self.settings_path.read_text())
        except (OSError, ValueError):
            return
        for k in self.settings:
            if k in data:
                self.settings[k] = data[k]
        for p in data.get("refs", []):
            if Path(p).is_file():
                try:
                    self.add_ref(Path(p), save=False)
                except Exception:  # noqa: BLE001 - a stale ref must not block startup
                    pass
        if self.settings.get("library") and Path(self.settings["library"]).is_dir():
            self._load_index_stats(reindex_if_stale=True)

    def _load_last_run(self) -> None:
        runs = self.list_runs()
        if runs:
            self.job["rows"] = self.load_run(runs[0]["name"])
            self.job["run"] = runs[0]["name"]

    # ----- references -----
    def add_ref(self, path: Path, save: bool = True) -> dict:
        with self.lock:
            if len(self.ref_order) >= MAX_REFS:
                raise ValueError(f"max {MAX_REFS} reference loops — remove one first")
        ref = load_ref(path, self.aparams)
        rid = ref.sha1[:12]
        with self.lock:
            if rid not in self.refs:
                self.refs[rid] = ref
                self.ref_order.append(rid)
            if save:
                self._save_settings()
        return self.ref_view(rid)

    def upload_ref(self, filename: str, data: bytes) -> dict:
        name = _safe_name(filename)
        if not is_audio_file(name):
            raise ValueError("only wav, aiff or flac files")
        dest = self.ws / "refs" / name
        if dest.exists() and dest.read_bytes() != data:
            dest = dest.with_name(f"{dest.stem}_{int(time.time())}{dest.suffix}")
        dest.write_bytes(data)
        return self.add_ref(dest)

    def remove_ref(self, rid: str) -> None:
        with self.lock:
            self.refs.pop(rid, None)
            if rid in self.ref_order:
                self.ref_order.remove(rid)
            self._save_settings()

    def ref_view(self, rid: str) -> dict:
        r = self.refs[rid]
        a = r.analysis
        n = max(1, a.n_samples)
        roles: dict[str, int] = {}
        for s in a.slices:
            roles[s.role] = roles.get(s.role, 0) + 1
        return {
            "id": rid, "name": r.name, "path": str(r.path), "url": file_url(r.path),
            "bpm": round(a.tempo.bpm, 2), "bars": a.tempo.bars, "beats": a.tempo.beats, "method": a.tempo.method,
            "is_loop": a.tempo.is_loop, "hits": len(a.slices), "swing": round(a.swing, 3),
            "duration": round(a.duration_s, 3), "sr": a.sr, "roles": roles,
            "peaks": waveform_peaks(r.audio, 400),
            "markers": [{"pos": s.start / n, "role": s.role} for s in a.slices],
        }

    # ----- library -----
    def set_library(self, path: str, force: bool = False) -> None:
        p = Path(path).expanduser()
        if not p.is_dir():
            raise ValueError(f"folder not found: {path}")
        with self.lock:
            if self.index["state"] == "indexing":
                raise ValueError("indexing is already running")
            self.settings["library"] = str(p.resolve())
            self._save_settings()
            self.index = {"state": "indexing", "done": 0, "total": 0, "current": "scanning…", "stats": None,
                          "error": None}
        threading.Thread(target=self._index_worker, args=(p, force), daemon=True).start()

    def _index_worker(self, root: Path, force: bool = False) -> None:
        from loopforge.library import update_index

        def prog(i: int, n: int, rel: str) -> None:
            with self.lock:
                self.index.update(done=i, total=n, current=rel)

        try:
            rep = update_index(root, params=self.aparams, jobs=self.jobs, progress=prog, force=force)
            with self.lock:
                self.index["report"] = {"analysed": rep.analyzed, "unchanged": rep.unchanged,
                                        "removed": len(rep.removed), "failed": rep.failed, "seconds": rep.seconds}
            self._load_index_stats()
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.index.update(state="error", error=str(e))

    def _load_index_stats(self, reindex_if_stale: bool = False) -> None:
        from loopforge.library import StaleIndexError, load_slice_table

        try:
            st = load_slice_table(self.settings["library"]).stats()
            with self.lock:
                self.index.update(state="ready", stats=st, current="")
        except StaleIndexError:
            # made by an older loopforge version (after an update): rebuild it in the background
            with self.lock:
                self.index.update(state="none", stats=None)
            if reindex_if_stale:
                self.set_library(self.settings["library"], force=True)
        except FileNotFoundError:
            with self.lock:
                self.index.update(state="none", stats=None)
        except Exception as e:  # noqa: BLE001 - a broken cache must never stop the app from starting
            with self.lock:
                self.index.update(state="error", stats=None, error=f"library index unreadable ({e}); rescan")

    # ----- generation -----
    def start_generate(self, opts: dict) -> None:
        with self.lock:
            if self.job["state"] == "running":
                raise ValueError("already generating")
            if not self.ref_order:
                raise ValueError("add at least one reference loop")
            if not self.settings.get("library"):
                raise ValueError("choose your library folder first")
            if self.index["state"] == "indexing":
                raise ValueError("wait for library indexing to finish")
            for k in SETTING_KEYS:
                if k in opts:
                    self.settings[k] = opts[k]
            s = self.settings
            refs = [self.refs[i] for i in self.ref_order]
            err = tempo_conflict(refs, s.get("bpm"))
            if err:
                raise ValueError(err)
            if not s.get("seed_lock"):
                s["seed"] = random.randrange(1_000_000)  # every run picks new sounds
            mix = tuple(t for t in (s.get("mix_types") or []) if t in MIX_TYPES)
            gp = replace(GenParams(), mode="simple", similarity=1.0, swing=0.0, bars=int(s["bars"]), palette_size=1,
                         mix_types=mix, mix_size=max(2, min(4, int(s.get("mix_size") or 3))),
                         bpm=float(s["bpm"]) if s.get("bpm") else None)
            run = time.strftime("%Y-%m-%d_%H%M%S")
            req = GenRequest(refs=refs, library=s["library"], out_dir=self.ws / "runs" / run, gp=gp,
                             aparams=self.aparams, variations=int(s["variations"]), seed=int(s["seed"]),
                             jobs=self.jobs, png=False, update_index=True,
                             command=f"ui: bars={gp.bars} mix={','.join(mix) or 'none'}x{gp.mix_size} seed={s['seed']}")
            self._save_settings()
            self._stop = False
            self.job = {"state": "running", "log": [], "rows": [], "run": run, "error": None,
                        "total": req.variations, "started": time.time()}
        threading.Thread(target=self._gen_worker, args=(req,), daemon=True).start()

    def _log(self, msg: str) -> None:
        with self.lock:
            self.job["log"].append(msg)

    def _gen_worker(self, req: GenRequest) -> None:
        def on_var(row: dict) -> None:
            with self.lock:
                self.job["rows"].append(self._row_view(row))

        try:
            run_generation(req, log=self._log, on_variation=on_var, should_stop=lambda: self._stop)
            with self.lock:
                self.job["state"] = "done"
            self._load_index_stats()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            with self.lock:
                self.job.update(state="error", error=str(e))

    def stop(self) -> None:
        self._stop = True

    def _row_view(self, row: dict) -> dict:
        m = row["metrics"]
        return {
            "stem": row["stem"], "file": row["file"], "wav": row["wav"], "json": row["json"],
            "url": file_url(row["wav"]), "seed": row["seed"], "bpm": row["bpm"], "bars": row["bars"],
            "peaks": row["peaks"], "steps": row["steps"], "sources": row["sources"],
            "favorite": (self.ws / "favorites" / row["file"]).exists(),
            "metrics": {k: m.get(k) for k in ("detected_bpm", "events", "files_used", "role_match", "role_chance",
                                              "accent_corr", "seam_ratio", "hits_per_bar", "timbre", "ref_timbre")},
            "template": row["template"],
        }

    # ----- runs / favourites -----
    def list_runs(self) -> list[dict]:
        out = []
        for d in sorted((self.ws / "runs").iterdir(), reverse=True):
            if d.is_dir() and any(d.glob("var_*.json")):
                out.append({"name": d.name, "count": len(list(d.glob("var_*.wav")))})
        return out

    def load_run(self, name: str) -> list[dict]:
        from loopforge.audio_io import load_audio
        from loopforge.generator.recipe import Recipe

        d = self.ws / "runs" / _safe_name(name)
        rows = []
        for js in sorted(d.glob("var_*.json")):
            wav = js.with_suffix(".wav")
            if not wav.is_file():
                continue
            try:
                r = Recipe.load(js)
                audio, _ = load_audio(wav)
            except Exception:  # noqa: BLE001
                continue
            rows.append(self._row_view({
                "stem": js.stem, "file": wav.name, "wav": str(wav), "json": str(js), "seed": r.seed, "bpm": r.bpm,
                "bars": r.bars, "peaks": waveform_peaks(audio),
                "steps": [{"pos": e.position / r.n_samples, "role": e.role} for e in r.events],
                "sources": sorted({e.source_rel for e in r.events}), "metrics": r.metrics, "template": r.template,
            }))
        return rows

    def open_run(self, name: str) -> None:
        rows = self.load_run(name)
        with self.lock:
            if self.job["state"] == "running":
                raise ValueError("wait for the current generation")
            self.job.update(state="idle", rows=rows, run=name, log=[], error=None)

    def toggle_favorite(self, wav: str) -> bool:
        src = Path(wav)
        self._check_allowed(src)
        fav = self.ws / "favorites"
        run = src.parent.name
        dst = fav / src.name
        if dst.exists():
            dst.unlink()
            dst.with_suffix(".json").unlink(missing_ok=True)
            state = False
        else:
            shutil.copy2(src, dst)
            if src.with_suffix(".json").exists():
                shutil.copy2(src.with_suffix(".json"), dst.with_suffix(".json"))
            (fav / "favorites.log").open("a").write(f"{time.strftime('%F %T')}\t{run}\t{src.name}\n")
            state = True
        with self.lock:
            for r in self.job["rows"]:
                if r["file"] == src.name:
                    r["favorite"] = state
        return state

    def favorites(self) -> list[dict]:
        return [{"name": p.name, "url": file_url(p), "path": str(p)}
                for p in sorted((self.ws / "favorites").glob("*.wav"))]

    # ----- files -----
    def allowed_roots(self) -> list[Path]:
        roots = [self.ws]
        if self.settings.get("library"):
            roots.append(Path(self.settings["library"]).resolve())
        roots += [self.refs[i].path.resolve().parent for i in self.ref_order]
        return roots

    def _check_allowed(self, p: Path) -> Path:
        rp = p.expanduser().resolve()
        if not any(rp == r or r in rp.parents for r in self.allowed_roots()):
            raise PermissionError("file outside the workspace/library")
        return rp

    def reveal(self, path: str) -> None:
        p = self._check_allowed(Path(path))
        target = p if p.is_dir() else p.parent
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", "-R", str(p)] if p.is_file() else ["open", str(target)])
        elif system == "Windows":
            subprocess.Popen(["explorer", "/select,", str(p)] if p.is_file() else ["explorer", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])

    def state(self) -> dict:
        with self.lock:
            return {
                "version": __version__, "workspace": str(self.ws), "settings": dict(self.settings),
                "refs": [self.ref_view(i) for i in self.ref_order], "index": dict(self.index),
                "job": {**self.job, "log": self.job["log"][-200:]},
                "favorites": self.favorites(), "runs": self.list_runs()[:30],
            }


def file_url(path: str | Path) -> str:
    return "/api/file?path=" + quote(str(Path(path).resolve()))


def browse(path: str | None) -> dict:
    p = Path(path).expanduser() if path else Path.home()
    if not p.is_dir():
        p = Path.home()
    p = p.resolve()
    dirs, n_audio = [], 0
    try:
        for c in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if c.name.startswith("."):
                continue
            if c.is_dir():
                dirs.append({"name": c.name, "path": str(c)})
            elif is_audio_file(c.name):
                n_audio += 1
    except PermissionError:
        pass
    roots = []
    if platform.system() == "Windows":
        roots = [f"{d}:\\" for d in "CDEFGHIJ" if Path(f"{d}:\\").exists()]
    elif platform.system() == "Darwin" and Path("/Volumes").is_dir():
        roots = [str(v) for v in sorted(Path("/Volumes").iterdir())]
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "dirs": dirs, "audio_files": n_audio,
            "home": str(Path.home()), "roots": roots}


class Handler(BaseHTTPRequestHandler):
    app: App  # set by make_server
    server_version = "loopforge"

    def log_message(self, fmt: str, *args) -> None:  # quiet
        pass

    # --- helpers ---
    def _json(self, data, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg: str, status: int = 400) -> None:
        self._json({"error": msg}, status)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("file too large")
        return self.rfile.read(n) if n else b""

    def _send_file(self, p: Path) -> None:
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if p.suffix.lower() == ".wav":
            ctype = "audio/wav"
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f"inline; filename=\"{quote(p.name)}\"")
        self.end_headers()
        self.wfile.write(data)

    # --- routing ---
    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._send_file(STATIC / "index.html")
            if u.path.startswith("/static/"):
                p = (STATIC / u.path[len("/static/"):]).resolve()
                if STATIC.resolve() not in p.parents or not p.is_file():
                    return self._err("not found", 404)
                return self._send_file(p)
            if u.path == "/api/state":
                return self._json(self.app.state())
            if u.path == "/api/browse":
                return self._json(browse(q.get("path", [None])[0]))
            if u.path == "/api/file":
                p = self.app._check_allowed(Path(q["path"][0]))
                if not p.is_file():
                    return self._err("not found", 404)
                return self._send_file(p)
            return self._err("not found", 404)
        except PermissionError as e:
            return self._err(str(e), 403)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._err(str(e), 500)

    def do_POST(self) -> None:
        u = urlparse(self.path)
        app = self.app
        try:
            if u.path == "/api/refs/upload":
                name = self.headers.get("X-Filename", "upload.wav")
                from urllib.parse import unquote

                return self._json(app.upload_ref(unquote(name), self._body()))
            body = json.loads(self._body() or b"{}")
            if u.path == "/api/refs/add":
                return self._json(app.add_ref(Path(body["path"])))
            if u.path == "/api/refs/remove":
                app.remove_ref(body["id"])
                return self._json({"ok": True})
            if u.path == "/api/library":
                app.set_library(body["path"])
                return self._json({"ok": True})
            if u.path == "/api/generate":
                app.start_generate(body)
                return self._json({"ok": True, "seed": app.settings["seed"]})
            if u.path == "/api/stop":
                app.stop()
                return self._json({"ok": True})
            if u.path == "/api/runs/open":
                app.open_run(body["name"])
                return self._json({"ok": True})
            if u.path == "/api/favorite":
                return self._json({"favorite": app.toggle_favorite(body["wav"])})
            if u.path == "/api/reveal":
                app.reveal(body["path"])
                return self._json({"ok": True})
            if u.path == "/api/settings":
                with app.lock:
                    for k in SETTING_KEYS:
                        if k in body:
                            app.settings[k] = body[k]
                    app._save_settings()
                return self._json({"ok": True})
            return self._err("not found", 404)
        except (ValueError, KeyError) as e:
            return self._err(str(e).strip("'"), 400)
        except PermissionError as e:
            return self._err(str(e), 403)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._err(str(e), 500)


def make_server(app: App, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    for p in [port, *range(port + 1, port + 20), 0]:
        try:
            return ThreadingHTTPServer((host, p), handler)
        except OSError:
            continue
    raise OSError("no free port")


def serve(workspace: Path | None = None, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          jobs: int | None = None) -> None:
    app = App(workspace or default_workspace(), jobs=jobs)
    srv = make_server(app, host, port)
    url = f"http://{host}:{srv.server_address[1]}/"
    print(f"Loopforge is running at {url}")
    print(f"Workspace: {app.ws}")
    print("Keep this window open while you use Loopforge. Close it (or press Ctrl+C) to quit.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
