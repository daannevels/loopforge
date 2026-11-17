import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from loopforge.ui.server import App, browse, make_server
from tests.synth import basic_pattern, render_loop, sparse_pattern, write


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("ui")
    lib = root / "lib"
    for i, (voice, steps) in enumerate([("closed", range(16)), ("open", [2, 6, 10, 14]), ("shaker", range(0, 16, 2)),
                                        ("perc", [0, 6, 12]), ("closed", range(0, 16, 2))]):
        write(lib / f"{voice}_{i}.wav", render_loop(sparse_pattern(list(steps), voice, 2), 124.0, 2, seed=i))
    ref = write(root / "my ref 128bpm.wav", render_loop(basic_pattern(2), 128.0, 2, seed=50))
    app = App(root / "ws", jobs=1)
    srv = make_server(app, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield {"url": f"http://127.0.0.1:{srv.server_address[1]}", "lib": lib, "ref": ref, "app": app, "root": root}
    srv.shutdown()


def call(base, path, body=None, headers=None, raw=False):
    data = None
    h = headers or {}
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(base + path, data=data, headers=h, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = r.read()
            return r.status, (out if raw else json.loads(out))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait(base, cond, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, st = call(base, "/api/state")
        if cond(st):
            return st
        time.sleep(0.2)
    raise TimeoutError


def test_full_flow(server):
    base = server["url"]
    code, html = call(base, "/", raw=True)
    assert code == 200 and b"LOOPFORGE" in html
    assert call(base, "/static/app.js", raw=True)[0] == 200
    assert call(base, "/static/../server.py", raw=True)[0] == 404

    # generate refused before setup
    code, err = call(base, "/api/generate", {})
    assert code == 400 and "reference" in err["error"]

    # upload a reference (as the browser does)
    code, ref = call(base, "/api/refs/upload", server["ref"].read_bytes(),
                     {"X-Filename": "my%20ref%20128bpm.wav", "Content-Type": "application/octet-stream"})
    # analyzer accuracy is covered elsewhere; here only the API contract
    assert code == 200 and ref["bpm"] == 128.0 and 28 <= ref["hits"] <= 33 and len(ref["peaks"]) == 400
    code, err = call(base, "/api/refs/upload", b"xx", {"X-Filename": "notes.txt"})
    assert code == 400

    # library: index in the background
    assert call(base, "/api/library", {"path": str(server["lib"])})[0] == 200
    st = wait(base, lambda s: s["index"]["state"] == "ready")
    assert st["index"]["stats"]["files"] == 5

    # generate 2 variations
    code, _ = call(base, "/api/generate", {"variations": 2, "seed": 3, "similarity": 0.6, "bars": 1})
    assert code == 200
    st = wait(base, lambda s: s["job"]["state"] in ("done", "error"))
    assert st["job"]["state"] == "done", st["job"]["error"]
    rows = st["job"]["rows"]
    assert len(rows) == 2 and rows[0]["bars"] == 1 and rows[0]["peaks"]
    assert st["settings"]["similarity"] == 0.6

    # audio served, favourites toggle, outside paths refused
    code, wav = call(base, rows[0]["url"], raw=True)
    assert code == 200 and wav[:4] == b"RIFF"
    assert call(base, "/api/favorite", {"wav": rows[0]["wav"]})[1]["favorite"] is True
    _, st = call(base, "/api/state")
    assert len(st["favorites"]) == 1 and st["job"]["rows"][0]["favorite"]
    assert call(base, "/api/favorite", {"wav": rows[0]["wav"]})[1]["favorite"] is False
    assert call(base, "/api/file?path=/etc/passwd")[0] == 403

    # settings + refs persist across restarts; last run is reloaded
    app2 = App(server["app"].ws, jobs=1)
    s2 = app2.state()
    assert len(s2["refs"]) == 1 and s2["settings"]["library"] == str(server["lib"].resolve())
    assert len(s2["job"]["rows"]) == 2 and s2["runs"]

    # remove ref
    assert call(base, "/api/refs/remove", {"id": ref["id"]})[0] == 200
    assert call(base, "/api/state")[1]["refs"] == []


def test_browse(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / ".hidden").mkdir()
    write(tmp_path / "x.wav", render_loop(basic_pattern(1), 120.0, 1))
    d = browse(str(tmp_path))
    assert [c["name"] for c in d["dirs"]] == ["a"] and d["audio_files"] == 1 and d["parent"]
