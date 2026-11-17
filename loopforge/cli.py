"""Command-line interface. Thin layer: parse args, do I/O, call the engine."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from loopforge import __version__
from loopforge.config import AnalysisParams


def _analysis_params(args: argparse.Namespace) -> AnalysisParams:
    p = AnalysisParams()
    kw = {}
    if getattr(args, "bpm_min", None) is not None:
        kw["bpm_min"] = args.bpm_min
    if getattr(args, "bpm_max", None) is not None:
        kw["bpm_max"] = args.bpm_max
    if getattr(args, "onset_delta", None) is not None:
        kw["onset_delta"] = args.onset_delta
    return replace(p, **kw) if kw else p


def cmd_analyze(args: argparse.Namespace) -> int:
    from loopforge.analysis.loop import analyze_loop
    from loopforge.audio_io import load_audio
    from loopforge.diagnostics import analysis_summary, analysis_table, plot_analysis

    params = _analysis_params(args)
    out_dir = Path(args.out_dir)
    status = 0
    for path in args.loops:
        path = Path(path)
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            status = 1
            continue
        audio, sr = load_audio(path)
        a = analyze_loop(audio, sr, name=path.name, params=params, bpm=args.bpm)
        print(analysis_summary(a))
        if not args.no_table:
            print()
            print(analysis_table(a))
        if not args.no_png:
            png = Path(args.png) if (args.png and len(args.loops) == 1) else out_dir / f"{path.stem}.analysis.png"
            plot_analysis(audio, a, png, params.steps_per_beat, params.beats_per_bar)
            print(f"\nspectrogram {png}")
        if args.json:
            js = out_dir / f"{path.stem}.analysis.json"
            js.parent.mkdir(parents=True, exist_ok=True)
            js.write_text(json.dumps(a.to_dict(), indent=1))
            print(f"json        {js}")
        print()
    return status


def _progress(i: int, n: int, rel: str) -> None:
    print(f"  [{i}/{n}] {rel}", file=sys.stderr)


def _ensure_index(library: str, cache: str | None, params: AnalysisParams, jobs: int | None, quiet: bool = False):
    from loopforge.library import update_index

    rep = update_index(library, cache=cache, params=params, jobs=jobs, progress=None if quiet else _progress)
    return rep


def cmd_index(args: argparse.Namespace) -> int:
    from loopforge.library import load_slice_table

    params = _analysis_params(args)
    rep = _ensure_index(args.library, args.cache, params, args.jobs)
    print(f"library     {rep.root}")
    print(f"cache       {rep.cache}")
    print(f"analysed    {rep.analyzed} (new {len(rep.added)}, changed {len(rep.updated)})  "
          f"unchanged {rep.unchanged}  touched {len(rep.touched)}  removed {len(rep.removed)}  "
          f"failed {len(rep.failed)}  in {rep.seconds:.1f}s")
    for rel, err in rep.failed.items():
        print(f"  FAILED {rel}: {err}")
    st = load_slice_table(args.library, args.cache).stats()
    print(f"contents    {st['files']} files, {st['slices']} slices")
    print("roles       " + ", ".join(f"{k}={v}" for k, v in sorted(st["roles"].items())))
    print("tempos      " + ", ".join(f"{k:g} BPM x{v}" for k, v in sorted(st["bpm"].items())))
    print("rates       " + ", ".join(f"{k} Hz x{v}" for k, v in sorted(st["sample_rates"].items())))
    return 1 if rep.failed and not st["files"] else 0


def _medoid_per_role(analysis, space, max_roles: int = 4) -> list[int]:
    """One representative slice per role: the one closest to the role's mean embedding."""
    import numpy as np

    out: list[int] = []
    roles = sorted({s.role for s in analysis.slices}, key=lambda r: -sum(s.role == r for s in analysis.slices))
    E = space.embed(analysis.features)
    for r in roles[:max_roles]:
        idx = [i for i, s in enumerate(analysis.slices) if s.role == r]
        c = E[idx].mean(axis=0)
        out.append(idx[int(np.argmin(((E[idx] - c) ** 2).sum(axis=1)))])
    return out


def cmd_neighbors(args: argparse.Namespace) -> int:
    import numpy as np

    from loopforge.analysis.features import describe_role, slice_features
    from loopforge.analysis.loop import analyze_loop
    from loopforge.audio_io import apply_fades, file_sha1, load_audio, resample, to_mono, to_stereo, write_wav
    from loopforge.diagnostics import step_label
    from loopforge.library import FeatureSpace, NearestNeighbors, load_slice_table
    from loopforge.library.slices import audition_chain, load_slice_audio
    from loopforge.models import FEATURE_INDEX as FI

    params = _analysis_params(args)
    if not args.no_update:
        _ensure_index(args.library, args.cache, params, args.jobs, quiet=True)
    table = load_slice_table(args.library, args.cache)
    if len(table) == 0:
        print("error: library index is empty", file=sys.stderr)
        return 1
    space = FeatureSpace.fit(table.features)
    nn = NearestNeighbors(space, table.features)

    qpath = Path(args.query)
    audio, sr = load_audio(qpath)
    mono = to_mono(audio)
    oneshot = args.oneshot or len(mono) / sr < 0.35
    queries: list[tuple[str, np.ndarray, np.ndarray]] = []  # (label, features, stereo audio)
    if oneshot:
        f = slice_features(mono, sr, params)
        queries.append((f"{qpath.name} (one-shot, {describe_role(f)})", f, to_stereo(audio)))
    else:
        a = analyze_loop(audio, sr, name=qpath.name, params=params, bpm=args.bpm)
        if not a.slices:
            print("error: no slices found in query", file=sys.stderr)
            return 1
        picks = args.slice if args.slice else _medoid_per_role(a, space)
        if args.role:
            picks = [i for i in range(len(a.slices)) if a.slices[i].role == args.role][: max(1, len(picks))]
        for i in picks:
            if not 0 <= i < len(a.slices):
                print(f"error: slice {i} out of range (0..{len(a.slices) - 1})", file=sys.stderr)
                return 1
            s = a.slices[i]
            queries.append((f"{qpath.name} slice #{i} step {s.step} ({step_label(s.step)}) role={s.role}",
                            a.features[i], to_stereo(audio[..., s.start : s.end])))

    exclude = table.mask_quiet(args.min_rms_db)
    if not args.include_same_file:
        exclude |= table.mask_files({file_sha1(qpath)})
    results = [
        nn.query_diverse(q[1], args.k, table.file_idx, max_per_group=args.per_file, exclude=exclude,
                         min_separation=args.min_sep)
        for q in queries
    ]

    def row(f: np.ndarray) -> str:
        return (f"cent {f[FI['centroid_hz']]:6.0f} Hz  atk {f[FI['attack_ms']]:5.1f} ms  "
                f"dec {f[FI['decay_ms']]:5.0f} ms  high {f[FI['band_high']]:.2f}  rms {f[FI['rms_db']]:6.1f} dB")

    out_dir = Path(args.export) if args.export else None
    for qi, (label, qf, qaudio) in enumerate(queries):
        print(f"query {qi}: {label}")
        print(f"   query                                    {row(qf)}")
        clips = [apply_fades(qaudio, sr, 0.0, 0.005)]
        for rank, (j, d) in enumerate(zip(*results[qi]), 1):
            name = table.label(int(j))
            print(f"  {rank:2d}. d={d:5.2f}  {name[-44:]:44s} {table.role[int(j)]:6s} {row(table.features[int(j)])}")
            if out_dir:
                clip, csr = load_slice_audio(table, int(j))
                q_dir = out_dir / f"q{qi}"
                stem = Path(table.files[int(table.file_idx[j])]["rel"]).stem
                write_wav(q_dir / f"{rank:02d}_d{d:.2f}_{stem}_s{int(table.slice_idx[j])}.wav", clip, csr)
                clips.append(resample(clip, csr, sr))
        if out_dir:
            q_dir = out_dir / f"q{qi}"
            write_wav(q_dir / "00_query.wav", clips[0], sr)
            write_wav(q_dir / "chain_query_then_neighbours.wav", audition_chain(clips, sr), sr)
            print(f"   exported -> {q_dir}")
        print()
    return 0


def _gen_params(args: argparse.Namespace):
    from loopforge.config import GenParams

    kw = {k: getattr(args, k) for k in ("bars", "similarity", "swing", "bpm", "palette_size", "max_file_share", "mode")
          if getattr(args, k, None) is not None}
    return GenParams(**kw)


def cmd_generate(args: argparse.Namespace) -> int:
    from loopforge.service import GenRequest, load_ref, run_generation

    import random

    aparams = _analysis_params(args)
    gp = _gen_params(args)
    if args.seed is None:
        args.seed = random.randrange(1_000_000)
    if gp.mode == "simple":
        gp = replace(gp, similarity=1.0, swing=0.0)
    print(f"seed {args.seed}")
    refs = []
    for r in args.refs:
        ref = load_ref(r, aparams, bpm=args.bpm)
        a = ref.analysis
        print(f"ref  {ref.name}: {a.tempo.bpm:g} BPM ({a.tempo.method}), {a.steps_total} steps, "
              f"{len(a.slices)} hits, swing {a.swing:.3f}")
        refs.append(ref)
    req = GenRequest(
        refs=refs, library=args.library, out_dir=Path(args.out), gp=gp, aparams=aparams,
        variations=args.variations, seed=args.seed, cache=args.cache, jobs=args.jobs,
        allow_ref_files=args.allow_ref_files, update_index=not args.no_update, png=not args.no_png,
        audition=args.audition, command="loopforge " + " ".join(sys.argv[1:]),
    )
    try:
        res = run_generation(req, log=lambda m: print(m))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"\nreport {res['report']}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from loopforge.audio_io import write_wav
    from loopforge.generator.recipe import Recipe, render_recipe
    from loopforge.library.slices import recipe_loader

    recipe = Recipe.load(args.recipe)
    audio = render_recipe(recipe, recipe_loader(args.library))
    out = Path(args.out) if args.out else Path(args.recipe).with_suffix(".render.wav")
    write_wav(out, audio, recipe.sr)
    print(f"rendered {out} ({audio.shape[1]} samples @ {recipe.sr} Hz)")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from loopforge.ui.server import serve

    serve(Path(args.workspace) if args.workspace else None, host=args.host, port=args.port,
          open_browser=not args.no_browser, jobs=args.jobs)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="loopforge", description="Generate new top loops from your own samples.")
    ap.add_argument("--version", action="version", version=f"loopforge {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyse loop(s): tempo, slices, features, spectrogram PNG")
    a.add_argument("loops", nargs="+", help="wav/aiff/flac file(s)")
    a.add_argument("--bpm", type=float, default=None, help="manual tempo override")
    a.add_argument("--bpm-min", type=float, default=None, help="lowest tempo to consider (default 100)")
    a.add_argument("--bpm-max", type=float, default=None, help="highest tempo to consider (default 150)")
    a.add_argument("--onset-delta", type=float, default=None,
                   help="onset sensitivity (lower = more hits, default 0.06)")
    a.add_argument("--out-dir", default=".", help="where to write PNG/JSON (default: current dir)")
    a.add_argument("--png", default=None, help="explicit PNG path (single input only)")
    a.add_argument("--no-png", action="store_true", help="skip the spectrogram")
    a.add_argument("--no-table", action="store_true", help="only print the summary")
    a.add_argument("--json", action="store_true", help="also write the full analysis as JSON")
    a.set_defaults(func=cmd_analyze)

    ix = sub.add_parser("index", help="scan a library folder (recursively) and update its slice cache")
    ix.add_argument("library", help="folder with wav/aiff/flac loops")
    ix.add_argument("--cache", default=None, help="cache file (default: <library>/.loopforge/index.sqlite)")
    ix.add_argument("--jobs", type=int, default=None, help="parallel workers (default: min(8, CPUs))")
    ix.add_argument("--bpm-min", type=float, default=None)
    ix.add_argument("--bpm-max", type=float, default=None)
    ix.set_defaults(func=cmd_index)

    nb = sub.add_parser("neighbors", help="find library slices that sound like a slice / one-shot / loop")
    nb.add_argument("query", help="a loop (slices are analysed) or a one-shot sample")
    nb.add_argument("--library", required=True, help="library folder (indexed incrementally first)")
    nb.add_argument("--cache", default=None)
    nb.add_argument("--k", type=int, default=10)
    nb.add_argument("--slice", type=int, nargs="+", default=None,
                    help="slice index/indices of the query loop (see `analyze`); default: one per role")
    nb.add_argument("--role", default=None, help="query slices with this role label (closed/open/shaker/perc)")
    nb.add_argument("--oneshot", action="store_true", help="treat the whole query file as one hit")
    nb.add_argument("--bpm", type=float, default=None)
    nb.add_argument("--include-same-file", action="store_true",
                    help="allow results from the query file itself (excluded by content hash by default)")
    nb.add_argument("--min-rms-db", type=float, default=-50.0, help="ignore library slices quieter than this")
    nb.add_argument("--per-file", type=int, default=2, help="max results from one source file (default 2)")
    nb.add_argument("--min-sep", type=float, default=0.05,
                    help="skip near-duplicate results closer than this to an earlier result (default 0.05)")
    nb.add_argument("--export", default=None, help="write query + neighbour slices as WAVs into this folder")
    nb.add_argument("--no-update", action="store_true", help="skip the incremental index update")
    nb.add_argument("--jobs", type=int, default=None)
    nb.set_defaults(func=cmd_neighbors)

    g = sub.add_parser("generate", help="generate new loops from refs + library")
    g.add_argument("--refs", nargs="+", required=True, help="1-3 reference loops")
    g.add_argument("--library", required=True)
    g.add_argument("--cache", default=None)
    # creative mode (mutated rhythm, swing) is kept in the engine but hidden: the product is "simple"
    g.add_argument("--mode", choices=["simple", "creative"], default=None, help=argparse.SUPPRESS)
    g.add_argument("--bars", type=int, default=2)
    g.add_argument("--variations", type=int, default=8)
    g.add_argument("--similarity", type=float, default=None, help=argparse.SUPPRESS)
    g.add_argument("--seed", type=int, default=None, help="base seed (variation i uses seed+i); default: random")
    g.add_argument("--swing", type=float, default=None, help=argparse.SUPPRESS)
    g.add_argument("--bpm", type=float, default=None, help="override tempo of refs/output")
    g.add_argument("--palette-size", type=int, default=None,
                   help="library samples per sound type (default 1 = one open hat, one closed hat, ...)")
    g.add_argument("--max-file-share", type=float, default=None, help="max share of hits from one file (0.4)")
    g.add_argument("--allow-ref-files", action="store_true", help="allow slices from the refs' own files")
    g.add_argument("--out", default="./out")
    g.add_argument("--no-png", action="store_true")
    g.add_argument("--audition", action="store_true",
                   help="also write listen/<var>_x4.wav (seam check) and <var>_AB.wav (ref x2, then variation x2)")
    g.add_argument("--no-update", action="store_true")
    g.add_argument("--jobs", type=int, default=None)
    g.set_defaults(func=cmd_generate)

    rd = sub.add_parser("render", help="re-render a loop from its JSON recipe")
    rd.add_argument("recipe")
    rd.add_argument("--out", default=None, help="output wav (default: <recipe>.render.wav)")
    rd.add_argument("--library", default=None, help="library root if it moved (sources resolved by relative path)")
    rd.set_defaults(func=cmd_render)

    u = sub.add_parser("ui", help="open the Loopforge app in your browser")
    u.add_argument("--workspace", default=None, help="where refs/runs/favourites live (default ~/Loopforge)")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--no-browser", action="store_true")
    u.add_argument("--jobs", type=int, default=None)
    u.set_defaults(func=cmd_ui)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
