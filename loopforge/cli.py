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
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
