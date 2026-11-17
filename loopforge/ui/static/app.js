"use strict";
/* Loopforge UI — vanilla JS, talks to the local Python server (/api/*). */

const $ = (s, el = document) => el.querySelector(s);
const ROLE_COLORS = { clap: "#ff5d5d", closed: "#4cc9f0", open: "#f5a524", shaker: "#8be36b", perc: "#ff6ad5" };
const roleColor = (name = "") => {
  for (const k in ROLE_COLORS) if (name.startsWith(k)) return ROLE_COLORS[k];
  return "#b9bec8";
};
let S = null; // last server state
let busy = false;

/* ---------------- API ---------------- */
async function api(path, body, opts = {}) {
  const init = body === undefined ? {} : {
    method: "POST",
    headers: opts.headers || { "Content-Type": "application/json" },
    body: body instanceof Blob || body instanceof ArrayBuffer ? body : JSON.stringify(body),
  };
  const r = await fetch(path, init);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `request failed (${r.status})`);
  return data;
}

function toast(msg, kind = "error") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (kind === "info" ? " info" : "");
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), kind === "info" ? 2500 : 6000);
}

/* ---------------- audio player ---------------- */
const player = {
  ctx: null, cache: new Map(), cur: null,
  ensure() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  },
  async buffer(url) {
    if (!this.cache.has(url)) {
      const p = fetch(url).then(r => r.arrayBuffer()).then(b => this.ensure().decodeAudioData(b));
      this.cache.set(url, p);
      p.catch(() => this.cache.delete(url));
    }
    return this.cache.get(url);
  },
  async play(key, url, offset = 0) {
    const ctx = this.ensure();
    const buf = await this.buffer(url);
    this.stop(false);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true; // sample-accurate looping: you hear the real loop seam
    src.connect(ctx.destination);
    const off = offset % buf.duration;
    src.start(0, off);
    this.cur = { key, url, src, t0: ctx.currentTime - off, dur: buf.duration };
    renderPlaying();
  },
  stop(render = true) {
    if (this.cur) { try { this.cur.src.stop(); } catch (_) { /* already stopped */ } }
    this.cur = null;
    if (render) renderPlaying();
  },
  elapsed() { return this.cur ? this.ctx.currentTime - this.cur.t0 : 0; },
  frac() { return this.cur ? (this.elapsed() % this.cur.dur) / this.cur.dur : 0; },
};

/* ---------------- knobs ---------------- */
function makeKnob(el, o) {
  el.innerHTML = `
    <div class="knob" tabindex="0" role="slider" aria-label="${o.label}">
      <svg viewBox="0 0 72 72">
        <defs><radialGradient id="capgrad" cx="40%" cy="30%" r="80%"><stop offset="0" stop-color="#3a3f4a"/><stop offset="1" stop-color="#1b1e24"/></radialGradient></defs>
        <path class="track" fill="none" stroke-width="5" stroke-linecap="round"/>
        <path class="fill" fill="none" stroke-width="5" stroke-linecap="round"/>
        <circle class="cap" cx="36" cy="36" r="22" stroke-width="1"/>
        <line class="tick" x1="36" y1="36" x2="36" y2="18" stroke-width="2.5" stroke-linecap="round"/>
      </svg>
    </div>
    <div class="knob-label">${o.label}</div>
    <div class="knob-value"></div>
    <div class="knob-help"></div>`;
  const knob = $(".knob", el), track = $(".track", el), fill = $(".fill", el), tick = $(".tick", el);
  const A0 = -135, A1 = 135, R = 30, C = 36;
  const pt = a => [C + R * Math.sin(a * Math.PI / 180), C - R * Math.cos(a * Math.PI / 180)];
  const arc = (a, b) => {
    const [x0, y0] = pt(a), [x1, y1] = pt(b);
    return `M${x0} ${y0} A${R} ${R} 0 ${b - a > 180 ? 1 : 0} 1 ${x1} ${y1}`;
  };
  track.setAttribute("d", arc(A0, A1));
  let value = o.value;
  const clamp = v => Math.min(o.max, Math.max(o.min, Math.round(v / o.step) * o.step));
  function draw() {
    const f = (value - o.min) / (o.max - o.min);
    const a = A0 + f * (A1 - A0);
    fill.setAttribute("d", f > 0.001 ? arc(A0, a) : "");
    tick.setAttribute("transform", `rotate(${a} 36 36)`);
    $(".knob-value", el).textContent = o.fmt(value);
    $(".knob-help", el).textContent = o.help ? o.help(value) : "";
    knob.setAttribute("aria-valuenow", value);
    knob.setAttribute("aria-valuetext", o.fmt(value));
  }
  function set(v, notify = true) {
    const nv = clamp(v);
    if (nv === value && !notify) return draw();
    value = nv; draw();
    if (notify && o.onChange) o.onChange(value);
  }
  let drag = null;
  knob.addEventListener("pointerdown", e => {
    knob.setPointerCapture(e.pointerId);
    drag = { y: e.clientY, v: value };
  });
  knob.addEventListener("pointermove", e => {
    if (!drag) return;
    const range = o.max - o.min, fine = e.shiftKey ? 0.25 : 1;
    set(drag.v + (drag.y - e.clientY) / 160 * range * fine);
  });
  knob.addEventListener("pointerup", () => (drag = null));
  knob.addEventListener("dblclick", () => set(o.def));
  knob.addEventListener("wheel", e => { e.preventDefault(); set(value + (e.deltaY < 0 ? o.step : -o.step)); }, { passive: false });
  knob.addEventListener("keydown", e => {
    const d = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1 }[e.key];
    if (d) { e.preventDefault(); set(value + d * o.step); }
    if (e.key === "Home") set(o.min);
    if (e.key === "End") set(o.max);
  });
  draw();
  return { get: () => value, set: v => set(v, false) };
}

const knobs = {};
let saveTimer = null;
function saveSettings() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => api("/api/settings", currentSettings()).catch(() => {}), 300);
}
function currentSettings() {
  return {
    variations: knobs.variations.get(), palette_size: knobs.palette.get(), bars: barsValue,
    seed: parseInt($("#seed").value || "0", 10), seed_lock: $("#seed-lock").checked,
  };
}

let barsValue = 2;
function setBars(v) {
  barsValue = v;
  document.querySelectorAll("#seg-bars button").forEach(b => b.setAttribute("aria-checked", String(+b.dataset.v === v)));
  $("#lcd-bars").textContent = v;
}

/* ---------------- drawing ---------------- */
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [g, w, h];
}

function drawWave(cv, peaks, markers, beats, play = -1, compact = false) {
  const [g, w, h] = setupCanvas(cv);
  g.clearRect(0, 0, w, h);
  const top = compact ? 2 : 6, bottom = compact ? 4 : 12, mid = top + (h - top - bottom) / 2, amp = (h - top - bottom) / 2;
  // grid
  if (beats) {
    for (let b = 0; b <= beats * 4; b++) {
      const x = Math.round((b / (beats * 4)) * w) + 0.5;
      g.strokeStyle = b % 16 === 0 ? "rgba(255,255,255,.22)" : b % 4 === 0 ? "rgba(255,255,255,.10)" : "rgba(255,255,255,.04)";
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
    }
  }
  // waveform
  const n = peaks.length, bw = w / n;
  const grad = g.createLinearGradient(0, top, 0, h - bottom);
  grad.addColorStop(0, "#cfd5df"); grad.addColorStop(0.5, "#8d96a6"); grad.addColorStop(1, "#cfd5df");
  for (let i = 0; i < n; i++) {
    const x = i * bw, a = Math.max(0.5, peaks[i] * amp);
    const played = play >= 0 && i / n <= play;
    g.fillStyle = played ? "rgba(245,165,36,.9)" : grad;
    g.fillRect(x, mid - a, Math.max(1, bw - (bw > 2 ? 0.6 : 0)), 2 * a);
  }
  // hit markers coloured by role
  for (const m of markers || []) {
    g.fillStyle = roleColor(m.role);
    const x = m.pos * w;
    if (compact) g.fillRect(x, h - 3, 2, 3);
    else { g.fillRect(x - 1, h - 8, 3, 6); }
  }
  if (play >= 0) {
    g.strokeStyle = "#ffd27a"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(play * w, 0); g.lineTo(play * w, h); g.stroke(); g.lineWidth = 1;
  }
}

/* ---------------- rendering ---------------- */
const svgPlay = '<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9.5-5.5z"/></svg>';
const svgStop = '<svg viewBox="0 0 16 16"><rect x="3.5" y="3.5" width="9" height="9" rx="1.5"/></svg>';

function renderRefs() {
  const box = $("#ref-slots");
  const refs = S.refs;
  box.innerHTML = "";
  refs.forEach(r => {
    const el = document.createElement("div");
    el.className = "ref";
    el.dataset.key = "ref:" + r.id;
    el.innerHTML = `
      <button class="play" title="Play / stop">${svgPlay}</button>
      <div style="min-width:0">
        <div class="name" title="${esc(r.path)}">${esc(r.name)}</div>
        <div class="meta">${r.bpm} BPM · ${r.beats % 4 ? r.beats + " beats" : r.bars + " bars"} · ${r.hits} hits${r.swing > 0.08 ? " · swing " + Math.round(r.swing * 100) + "%" : ""}</div>
        <canvas></canvas>
      </div>
      <button class="remove" title="Remove reference" aria-label="Remove ${esc(r.name)}">✕</button>`;
    $(".play", el).onclick = () => togglePlay("ref:" + r.id, r.url);
    $(".remove", el).onclick = async () => {
      if (player.cur && player.cur.key === "ref:" + r.id) player.stop();
      await api("/api/refs/remove", { id: r.id }).catch(e => toast(e.message));
      refresh();
    };
    box.appendChild(el);
    requestAnimationFrame(() => drawWave($("canvas", el), r.peaks, r.markers, r.beats / 4 || r.bars, -1, true));
  });
  if (refs.length < 3) {
    const b = document.createElement("button");
    b.className = "ref-empty";
    b.innerHTML = `<span style="font-size:20px;color:var(--accent)">＋</span><span>${refs.length ? "Add another reference" : "Drop reference loops here, or click to choose"}</span>`;
    b.onclick = () => $("#ref-file").click();
    box.appendChild(b);
  }
  const bpm = refs.length ? refs[0].bpm : null;
  $("#lcd-bpm").textContent = bpm ? Number(bpm).toFixed(2) : "---.--";
}

function renderLibrary() {
  const lib = S.settings.library, ix = S.index;
  $("#lib-path-text").textContent = lib || "Choose your loop library…";
  $("#lib-path").title = lib || "Choose your sample library folder";
  $("#btn-rescan").disabled = !lib || ix.state === "indexing";
  const stats = $("#lib-stats");
  stats.innerHTML = "";
  if (ix.stats) {
    const st = ix.stats;
    const chips = [
      `<b>${st.files}</b> loops`, `<b>${st.slices}</b> hits`,
      ...Object.keys(st.bpm).slice(0, 4).map(b => `<b>${b}</b> BPM ×${st.bpm[b]}`),
    ];
    stats.innerHTML = chips.map(c => `<span class="chip">${c}</span>`).join("");
  }
  if (ix.state === "error") stats.innerHTML += `<span class="chip err">${esc(ix.error || "indexing failed")}</span>`;
  if (ix.report && ix.report.failed && Object.keys(ix.report.failed).length)
    stats.innerHTML += `<span class="chip err" title="${esc(Object.keys(ix.report.failed).join("\n"))}">${Object.keys(ix.report.failed).length} unreadable</span>`;
  const pr = $("#lib-progress");
  pr.hidden = ix.state !== "indexing";
  if (ix.state === "indexing") {
    const f = ix.total ? ix.done / ix.total : 0;
    $(".bar", pr).style.width = (f * 100).toFixed(1) + "%";
    $(".ptext", pr).textContent = ix.total ? `Analysing ${ix.done}/${ix.total} · ${ix.current}` : "Scanning folder…";
  }
}

const cards = new Map(); // stem -> {el, row}
let shownRun = null;
function renderResults() {
  const job = S.job, rows = job.rows || [];
  const grid = $("#results");
  if (shownRun !== job.run) { grid.innerHTML = ""; cards.clear(); shownRun = job.run; }
  for (const row of rows) {
    let c = cards.get(row.stem);
    if (!c) { c = { el: makeCard(row), row }; cards.set(row.stem, c); grid.appendChild(c.el); }
    c.row = row;
    $(".fav", c.el).classList.toggle("fav-on", !!row.favorite);
    $(".fav", c.el).textContent = row.favorite ? "★" : "☆";
  }
  $("#results-empty").hidden = rows.length > 0 || job.state === "running";
  const sel = $("#runs");
  const opts = S.runs.map(r => `<option value="${esc(r.name)}"${r.name === job.run ? " selected" : ""}>${esc(r.name.replace("_", " "))} · ${r.count}</option>`).join("");
  if (sel.dataset.sig !== opts) { sel.innerHTML = opts || "<option>no runs yet</option>"; sel.dataset.sig = opts; }
  sel.disabled = !S.runs.length || job.state === "running";
  renderPlaying();
}

function makeCard(row) {
  const el = document.createElement("div");
  el.className = "card";
  el.draggable = true;
  el.dataset.key = "var:" + row.stem;
  const num = row.stem.match(/var_(\d+)/)?.[1] || "";
  const m = row.metrics || {};
  const meta = [`${m.events ?? row.steps.length} hits`, `${m.files_used ?? row.sources.length} sources`];
  const tip = [`Rhythm template: ${row.template}`, m.role_match != null ? `Role match vs refs: ${Math.round(m.role_match * 100)}%` : "",
    "Sources:", ...row.sources].filter(Boolean).join("\n");
  el.innerHTML = `
    <div class="card-head">
      <span class="num">VAR ${num}</span><span class="seed">seed ${row.seed}</span><span class="spacer"></span>
      <button class="icon-btn fav" title="Favourite (copies it to your favourites folder)">☆</button>
      <button class="icon-btn reveal" title="Show file in Finder / Explorer">⤴</button>
    </div>
    <canvas title="Click to play / stop"></canvas>
    <div class="card-foot">
      <button class="play" title="Play / stop (loops seamlessly)">${svgPlay}</button>
      <button class="ab small" title="Switch between this loop and reference 1 in time">A/B</button>
      <span class="meta" title="${esc(tip)}">${meta.join(" · ")}</span>
      <span class="drag">⠿ drag to DAW</span>
    </div>`;
  const key = "var:" + row.stem;
  $(".play", el).onclick = () => togglePlay(key, row.url);
  $("canvas", el).onclick = () => togglePlay(key, row.url);
  $(".ab", el).onclick = () => toggleAB(row);
  $(".fav", el).onclick = async () => {
    try { const r = await api("/api/favorite", { wav: row.wav }); toast(r.favorite ? "Saved to favourites" : "Removed from favourites", "info"); refresh(); }
    catch (e) { toast(e.message); }
  };
  $(".reveal", el).onclick = () => api("/api/reveal", { path: row.wav }).catch(e => toast(e.message));
  el.addEventListener("dragstart", e => {
    const abs = location.origin + row.url;
    e.dataTransfer.setData("DownloadURL", `audio/wav:${row.file}:${abs}`); // Chrome/Edge: drop into DAW/Finder
    e.dataTransfer.setData("text/uri-list", abs);
    e.dataTransfer.setData("text/plain", row.wav);
    e.dataTransfer.effectAllowed = "copy";
  });
  requestAnimationFrame(() => drawCard(el, row, -1));
  return el;
}

function drawCard(el, row, play) {
  drawWave($("canvas", el), row.peaks, row.steps, row.bars, play);
}

let abState = null; // {row, onRef}
async function togglePlay(key, url) {
  try {
    if (player.cur && player.cur.key === key) { player.stop(); abState = null; return; }
    abState = null;
    await player.play(key, url);
  } catch (e) { toast("Could not play audio: " + e.message); }
}
async function toggleAB(row) {
  if (!S.refs.length) return toast("Add a reference loop to compare against");
  const ref = S.refs[0];
  const varKey = "var:" + row.stem;
  const t = player.elapsed();
  try {
    if (abState && abState.row.stem === row.stem && abState.onRef) {
      abState.onRef = false;
      await player.play(varKey, row.url, t);
    } else {
      abState = { row, onRef: true };
      await player.play(varKey + "#ref", ref.url, player.cur && player.cur.key.startsWith(varKey) ? t : 0);
    }
  } catch (e) { toast(e.message); }
  renderPlaying();
}

function renderPlaying() {
  const key = player.cur ? player.cur.key : null;
  document.querySelectorAll("[data-key]").forEach(el => {
    const k = el.dataset.key;
    const on = key === k || (key && key === k + "#ref");
    el.classList.toggle("playing", !!on);
    const p = $(".play", el);
    if (p) { p.classList.toggle("on", !!on); p.innerHTML = on ? svgStop : svgPlay; }
    const ab = $(".ab", el);
    if (ab) { const r = key === k + "#ref"; ab.classList.toggle("on", r); ab.textContent = r ? "REF" : "A/B"; }
    if (!on && el.classList.contains("card") && el._drawn !== -1) {
      const c = cards.get(k.slice(4)); if (c) drawCard(el, c.row, -1); el._drawn = -1;
    }
  });
}

function animate() {
  if (player.cur) {
    const k = player.cur.key.replace("#ref", "");
    const el = document.querySelector(`[data-key="${CSS.escape(k)}"]`);
    if (el && el.classList.contains("card")) {
      const c = cards.get(k.slice(4));
      if (c) { drawCard(el, c.row, player.frac()); el._drawn = 1; }
    }
  }
  requestAnimationFrame(animate);
}

function renderStatus() {
  const ix = S.index, job = S.job;
  const led = $("#led");
  let status = "READY", cls = S.refs.length && S.settings.library ? "ok" : "";
  if (ix.state === "indexing") { status = `INDEX ${ix.done}/${ix.total || "?"}`; cls = "busy"; }
  if (job.state === "running") { status = `GEN ${job.rows.length}/${job.total}`; cls = "busy"; }
  if (job.state === "error") { status = "ERROR"; cls = "err"; }
  $("#lcd-status").textContent = status;
  led.className = "led " + cls;
  busy = ix.state === "indexing" || job.state === "running";

  const g = $("#btn-generate");
  const running = job.state === "running";
  g.classList.toggle("running", running);
  $(".gtext", g).textContent = running ? `Generating ${job.rows.length}/${job.total} · click to stop` : "Generate";
  $(".gprog", g).style.width = running && job.total ? (job.rows.length / job.total * 100) + "%" : "0";
  const why = !S.refs.length ? "Add at least one reference loop" : !S.settings.library ? "Choose your library folder"
    : ix.state === "indexing" ? "Wait for library indexing" : "";
  g.disabled = !running && !!why;
  g.title = why || (running ? "Stop after the current loop" : "Generate new loops");
  $("#fav-count").textContent = S.favorites.length;
  $("#ver").textContent = "v" + S.version;
  const log = $("#log");
  const lines = job.log.join("\n") + (job.error ? "\nERROR: " + job.error : "");
  if (log.textContent !== lines) { log.textContent = lines; log.scrollTop = log.scrollHeight; }
  if (job.state === "error" && job.error && refresh._lastErr !== job.error) { toast(job.error); refresh._lastErr = job.error; }
}

function applySettings(s) {
  knobs.variations.set(s.variations); knobs.palette.set(s.palette_size); setBars(+s.bars || 2);
  $("#seed").value = s.seed; $("#seed-lock").checked = !!s.seed_lock;
}

async function refresh() {
  try {
    const first = !S;
    S = await api("/api/state");
    if (first) applySettings(S.settings);
    renderRefs(); renderLibrary(); renderResults(); renderStatus();
  } catch (e) {
    $("#lcd-status").textContent = "OFFLINE";
    $("#led").className = "led err";
  }
  clearTimeout(refresh._t);
  refresh._t = setTimeout(refresh, busy ? 600 : 4000);
}

/* ---------------- uploads ---------------- */
async function uploadFiles(files) {
  const audio = [...files].filter(f => /\.(wav|aiff?|flac)$/i.test(f.name));
  if (!audio.length) return toast("Drop wav, aiff or flac files");
  for (const f of audio) {
    $("#lcd-status").textContent = "ANALYSING";
    try {
      await api("/api/refs/upload", await f.arrayBuffer(), { headers: { "X-Filename": encodeURIComponent(f.name), "Content-Type": "application/octet-stream" } });
    } catch (e) { toast(`${f.name}: ${e.message}`); }
  }
  refresh();
}

/* ---------------- folder browser ---------------- */
let brPath = null;
async function browseTo(path) {
  try {
    const d = await api("/api/browse?path=" + encodeURIComponent(path || ""));
    brPath = d.path;
    $("#br-path").value = d.path;
    $("#br-up").disabled = !d.parent;
    $("#br-up").onclick = () => browseTo(d.parent);
    $("#br-count").textContent = d.audio_files ? `${d.audio_files} audio files here (sub-folders are included too)` : "Sub-folders are included too";
    $("#br-list").innerHTML = d.dirs.length ? "" : '<li class="hint" style="padding:10px">No sub-folders</li>';
    for (const c of d.dirs) {
      const li = document.createElement("li");
      li.innerHTML = `<button type="button">${esc(c.name)}</button>`;
      $("button", li).onclick = () => browseTo(c.path);
      $("#br-list").appendChild(li);
    }
    $("#br-roots").innerHTML = "";
    for (const r of [d.home, ...d.roots]) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "ghost small"; b.textContent = r === d.home ? "Home" : r;
      b.onclick = () => browseTo(r);
      $("#br-roots").appendChild(b);
    }
  } catch (e) { toast(e.message); }
}

/* ---------------- utils ---------------- */
function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

/* ---------------- init ---------------- */
function init() {
  knobs.variations = makeKnob($("#k-variations"), {
    label: "Variations", min: 1, max: 16, step: 1, value: 8, def: 8, fmt: v => String(v), help: () => "loops per run, all different sounds", onChange: saveSettings,
  });
  knobs.palette = makeKnob($("#k-palette"), {
    label: "Palette", min: 1, max: 4, step: 1, value: 1, def: 1, fmt: v => v + (v === 1 ? " sample" : " samples"),
    help: v => v === 1 ? "one sample per sound type" : "per sound type, mixed", onChange: saveSettings,
  });
  document.querySelectorAll("#seg-bars button").forEach(b => (b.onclick = () => { setBars(+b.dataset.v); saveSettings(); }));
  $("#seed-lock").addEventListener("change", saveSettings);
  $("#seed").addEventListener("change", saveSettings);
  $("#btn-dice").onclick = () => { $("#seed").value = Math.floor(Math.random() * 100000); saveSettings(); };

  $("#btn-generate").onclick = async () => {
    if (S && S.job.state === "running") { await api("/api/stop", {}); return; }
    player.stop();
    try {
      const r = await api("/api/generate", currentSettings());
      if (r.seed != null) $("#seed").value = r.seed;
      shownRun = null; refresh();
    }
    catch (e) { toast(e.message); }
  };
  $("#runs").onchange = async e => {
    try { await api("/api/runs/open", { name: e.target.value }); refresh(); } catch (err) { toast(err.message); }
  };

  // references: click + drag & drop (panel and whole window)
  $("#ref-file").onchange = e => { uploadFiles(e.target.files); e.target.value = ""; };
  const panel = $("#refs-panel");
  ["dragenter", "dragover"].forEach(t => window.addEventListener(t, e => {
    if ([...(e.dataTransfer?.types || [])].includes("Files")) { e.preventDefault(); panel.classList.add("drop"); }
  }));
  ["dragleave", "drop"].forEach(t => window.addEventListener(t, e => {
    if (t === "dragleave" && e.relatedTarget) return;
    panel.classList.remove("drop");
  }));
  window.addEventListener("drop", e => {
    if (e.dataTransfer?.files?.length) { e.preventDefault(); uploadFiles(e.dataTransfer.files); }
  });

  // library
  const dlg = $("#browser");
  $("#lib-path").onclick = () => { browseTo(S?.settings.library || ""); dlg.showModal(); };
  $("#br-go").onclick = () => browseTo($("#br-path").value);
  $("#br-path").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); browseTo(e.target.value); } });
  $("#br-use").onclick = async () => {
    try { await api("/api/library", { path: brPath }); dlg.close(); refresh(); } catch (e) { toast(e.message); }
  };
  $("#btn-rescan").onclick = async () => {
    try { await api("/api/library", { path: S.settings.library }); refresh(); } catch (e) { toast(e.message); }
  };

  $("#btn-workspace").onclick = () => api("/api/reveal", { path: S.workspace }).catch(e => toast(e.message));
  $("#btn-favs").onclick = () => api("/api/reveal", { path: S.workspace + "/favorites" }).catch(e => toast(e.message));
  $("#btn-log").onclick = () => { const l = $("#log"); l.hidden = !l.hidden; $("#btn-log").textContent = l.hidden ? "Log ▾" : "Log ▴"; };

  window.addEventListener("keydown", e => {
    if (e.code !== "Space" || e.target.closest("input, select, textarea, button, .knob")) return;
    e.preventDefault();
    if (player.cur) player.stop();
    else { const first = [...cards.values()][0]; if (first) togglePlay("var:" + first.row.stem, first.row.url); }
  });
  window.addEventListener("resize", () => { if (S) { renderRefs(); cards.forEach((c, k) => drawCard(c.el, c.row, -1)); } });

  refresh();
  requestAnimationFrame(animate);
}

init();
