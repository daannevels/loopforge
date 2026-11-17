"""Application services shared by the CLI and the UI (file I/O + orchestration).

The engine (analysis/, library/search, generator/) stays pure; this layer loads
files, keeps caches and writes results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from loopforge import __version__
from loopforge.config import AnalysisParams, GenParams
from loopforge.models import LoopAnalysis

Log = Callable[[str], None]


def _noop(_: str) -> None:
    pass


@dataclass
class RefLoop:
    path: Path
    sha1: str
    audio: np.ndarray
    sr: int
    analysis: LoopAnalysis
    rms: float

    @property
    def name(self) -> str:
        return self.path.name

    def info(self) -> dict:
        a = self.analysis
        return {"path": str(self.path.resolve()), "sha1": self.sha1, "bpm": a.tempo.bpm, "bars": a.tempo.bars,
                "beats": a.tempo.beats, "swing": round(a.swing, 4), "slices": len(a.slices)}


def load_ref(path: str | Path, aparams: AnalysisParams, bpm: float | None = None) -> RefLoop:
    from loopforge.analysis.loop import analyze_loop
    from loopforge.audio_io import file_sha1, load_audio

    path = Path(path)
    audio, sr = load_audio(path)
    a = analyze_loop(audio, sr, name=path.name, params=aparams, bpm=bpm)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return RefLoop(path, file_sha1(path), audio, sr, a, rms)


def tempo_conflict(refs: list[RefLoop], bpm: float | None) -> str | None:
    if bpm or not refs:
        return None
    bpms = [r.analysis.tempo.bpm for r in refs]
    if max(bpms) / min(bpms) > 1.03:
        return f"references disagree on tempo ({', '.join(f'{b:g}' for b in bpms)} BPM); set a BPM"
    return None


def waveform_peaks(audio: np.ndarray, n: int = 600) -> list[float]:
    """Max |amplitude| per bucket of the mono mix, for drawing waveforms."""
    from loopforge.audio_io import to_mono

    m = np.abs(to_mono(audio))
    if len(m) == 0:
        return []
    edges = np.linspace(0, len(m), n + 1).astype(int)
    return [round(float(m[a:b].max()) if b > a else 0.0, 4) for a, b in zip(edges[:-1], edges[1:])]


@dataclass
class GenRequest:
    refs: list[RefLoop]
    library: str
    out_dir: Path
    gp: GenParams = field(default_factory=GenParams)
    aparams: AnalysisParams = field(default_factory=AnalysisParams)
    variations: int = 8
    seed: int = 42
    cache: str | None = None
    jobs: int | None = None
    allow_ref_files: bool = False
    update_index: bool = True
    png: bool = True
    audition: bool = False
    command: str = ""


def write_audition(folder: Path, stem: str, audio: np.ndarray, recipe, ref: RefLoop) -> None:
    """<stem>_x4.wav (loop repeated: seam check) and <stem>_AB.wav (ref x2, gap, variation x2)."""
    from loopforge.audio_io import resample, to_stereo, write_wav

    write_wav(folder / f"{stem}_x4.wav", np.tile(audio, (1, 4)), recipe.sr)
    ra = to_stereo(resample(ref.audio, ref.sr, recipe.sr))[:, : recipe.n_samples]
    if ra.shape[1] < recipe.n_samples:
        ra = np.pad(ra, ((0, 0), (0, recipe.n_samples - ra.shape[1])))
    gap = np.zeros((2, recipe.sr // 2), np.float32)
    write_wav(folder / f"{stem}_AB.wav", np.concatenate([np.tile(ra, (1, 2)), gap, np.tile(audio, (1, 2))], 1),
              recipe.sr)


def run_generation(
    req: GenRequest,
    log: Log = _noop,
    on_variation: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Generate `req.variations` loops into `req.out_dir`. Returns {rows, profile, report}."""
    from loopforge.analysis.loop import analyze_loop
    from loopforge.audio_io import write_wav
    from loopforge.diagnostics import evaluate_output, generation_report, plot_analysis, timbre_stats
    from loopforge.generator.profile import build_profile
    from loopforge.generator.run import generate_one, selection_mask
    from loopforge.library import FeatureSpace, NearestNeighbors, load_slice_table, update_index
    from loopforge.library.slices import recipe_loader

    if not req.refs:
        raise ValueError("add at least one reference loop")
    err = tempo_conflict(req.refs, req.gp.bpm)
    if err:
        raise ValueError(err)
    out = Path(req.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gp = req.gp

    if req.update_index:
        log("updating library index…")
        update_index(req.library, cache=req.cache, params=req.aparams, jobs=req.jobs)
    table = load_slice_table(req.library, req.cache)
    if len(table) == 0:
        raise ValueError("the library index is empty — pick a folder with wav/aiff/flac loops")
    space = FeatureSpace.fit(table.features, group_weights=dict(gp.timbre_group_weights))
    nn = NearestNeighbors(space, table.features)
    analyses = [r.analysis for r in req.refs]
    profile = build_profile(analyses, [r.rms for r in req.refs], space, gp, bpm=gp.bpm)
    refs_info = [r.info() for r in req.refs]
    exclude = selection_mask(table, gp, set() if req.allow_ref_files else {r.sha1 for r in req.refs})
    log(f"library: {len(table.files)} files, {len(table)} slices ({int((~exclude).sum())} usable)")
    log("roles: " + ", ".join(f"{r.name} ({r.share:.0%})" for r in profile.roles))
    (out / "profile.json").write_text(json.dumps(profile.summary(), indent=1))

    loader = recipe_loader()
    ref_t = timbre_stats(analyses)
    rows: list[dict] = []
    from loopforge.generator.select import RunMemory

    memory = RunMemory.new()  # simple mode: each variation prefers samples/files not used before
    for v in range(req.variations):
        if should_stop and should_stop():
            log("stopped")
            break
        seed = req.seed + v
        recipe, audio = generate_one(profile, table, nn, gp, seed, exclude, loader, refs_info, __version__,
                                     memory=memory)
        stem = f"var_{v + 1:02d}_seed{seed}"
        wav = write_wav(out / f"{stem}.wav", audio, recipe.sr)
        oa = analyze_loop(audio, recipe.sr, name=wav.name, params=req.aparams)
        recipe.metrics = evaluate_output(audio, oa, recipe, profile, ref_t, space)
        js = recipe.save(out / f"{stem}.json")
        png = plot_analysis(audio, oa, out / f"{stem}.png") if req.png else None
        if req.audition:
            write_audition(out / "listen", stem, audio, recipe, req.refs[0])
        m = recipe.metrics
        log(f"{wav.name}: {len(recipe.events)} hits, {m['detected_bpm']:g} BPM, {m['files_used']} source files")
        row = {
            "file": wav.name, "stem": stem, "wav": str(wav), "json": str(js), "png": str(png) if png else None,
            "seed": seed, "metrics": m, "template": recipe.template, "bpm": recipe.bpm, "bars": recipe.bars,
            "sr": recipe.sr, "n_samples": recipe.n_samples, "peaks": waveform_peaks(audio),
            "steps": [{"pos": e.position / recipe.n_samples, "role": e.role} for e in recipe.events],
            "sources": sorted({e.source_rel for e in recipe.events}),
        }
        rows.append(row)
        if on_variation:
            on_variation(row)
    report = generation_report(rows, profile, [r.name for r in req.refs], req.command or "loopforge generate")
    (out / "report.md").write_text(report)
    return {"rows": rows, "profile": profile.summary(), "report": str(out / "report.md")}
