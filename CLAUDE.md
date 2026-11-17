# loopforge — project notes for Claude

Generates new top loops (hats/shakers/percussion) by slicing and recombining the
user's own sample library, guided by 1–3 reference loops. Phase 1 = Python engine +
CLI (+ small Gradio UI in M5). A JUCE/VST3 port comes later, so the engine must stay portable.

## Commands
- `uv sync` — install (Python ≥3.11; dev runs on 3.13)
- `uv run pytest` — tests (synthetic audio only)
- `uv run loopforge analyze <loop.wav> [--bpm X] [--out-dir D] [--json]`
- `uv run loopforge index <library> [--jobs N] [--cache PATH]`
- `uv run loopforge neighbors <loop|one-shot> --library <dir> [--k 10] [--slice i ...] [--export DIR]`

## Conventions
- Audio inside the engine is float32 shaped `(channels, samples)`; mono analysis via `audio_io.to_mono`.
- Engine = pure functions over dataclasses (`models.py`) + explicit params (`config.py`).
  No globals, no UI, no file I/O in `analysis/` or `generator/`. File I/O lives in `audio_io`, `library`, `cli`.
- Every librosa call goes through `analysis/dsp.py` (onset flux, mel/MFCC) so a C++ port swaps one module.
- Bump `config.ANALYZER_VERSION` whenever slicing/features/tempo output changes (invalidates library caches).
- Feature vector layout is defined once in `models.FEATURE_SPEC` (name, group). Never index features by number.
- Type hints everywhere. No copyrighted audio in the repo or tests: fixtures come from `tests/synth.py`.
  `.gitignore` excludes all audio files.
- Commit after each milestone; wait for user confirmation before starting the next one.

## Key decisions
- **Loop length**: a file is a loop when it is a whole number of *beats* (≥2), not only whole bars
  (e.g. a 10-beat loop). Odd-beat lengths need ≤5 ms error, a good grid score at phase 0 and
  |latency| < 0.15 step, otherwise they are treated as non-loops.
- **Tempo** precedence: manual → filename (`145 BPM`, `124bpm`, validated: must fit the length as a loop or
  align with the grid) → duration fit (N∈{1,2,4,8,16} bars giving a BPM inside 100–150, checked by grid
  alignment score) → BPM×phase grid search for non-loops. BPM is snapped to a round value when that changes
  the loop length by < 2 ms. Phase is searched over two steps (keeps even/odd parity, needed for swing);
  ties prefer the straighter (less swing) reading.
- **Onsets**: librosa log-mel flux, then `log1p(10·env/max)` compression (quiet hats next to loud percs
  must still peak), delta 0.1. Loops are padded circularly (tail before, head after) so a hit at sample 0
  is found. Each onset is refined to sample accuracy: walk back from the local amplitude peak to
  15% above the local floor, minus 0.5 ms pre-roll. Gates: −45 dB rel. loop peak and −60 dBFS absolute.
- **Slicing**: slice = onset → next onset, max 1 beat. Step = nearest 16th where odd steps are expected
  `swing` late; `offset` keeps micro-timing (steps). Swing = median delay of hits in the first half of
  odd steps; grid offset = median deviation of even steps. Flams/rolls stay as extra slices on the same step.
- **Features** (`models.FEATURE_SPEC`): rms/peak dB, attack (start→peak), decay (peak→−20 dB, linearly
  extrapolated from the dB envelope when the slice is cut by the next hit), duration, centroid/rolloff/
  flatness (energy-weighted, ≤20 kHz), band ratios (<250, 250–4k, >4k Hz), 13 MFCC means (40 mels,
  fmax 16 kHz, energy-weighted). Fixed fmax values make 44.1k and 48k material comparable.
- **Library index** (`library/index.py`): SQLite at `<library>/.loopforge/index.sqlite`. Tables
  `files` (rel path, size, mtime_ns, sha1, tempo info, error) + `slices` (positions, step, role, float32
  feature blob). Skip if size+mtime unchanged; if mtime changed but sha1 equal → metadata refresh only;
  else re-analyse. Analyzer version or analysis params change → full rebuild. Hidden dirs/files and macOS
  `._*` files are skipped; unreadable files are recorded with `error` and skipped. Parallel workers use the
  `spawn` start method. Slice audio is re-read from the source (`library/slices.py`), never cached.
- **Timbre space** (`library/search.py`): log2 for centroid/rolloff, log1p for times, log for flatness,
  z-score over the library, then group weights (each group's weight shared by its members so 12 MFCCs don't
  dominate): loudness 0 (gain matched later; `mfcc_0` counts as loudness), envelope 1, spectral 1,
  bands 0.7, mfcc 1; `duration_ms` ×0.3 (it mostly reflects pattern density). Brute-force Euclidean kNN.
  `query_diverse` caps results per source file and skips near-duplicates (repeated identical hits).
  Library slices below −50 dB RMS are excluded from search; refs' own files are excluded by sha1.
- **Role** labels (`closed/open/shaker/perc`) in analysis are a heuristic for diagnostics only; the
  generator will derive roles by clustering reference slices (M3).
- **Stereo**: slices keep their original stereo image; mono sources are centred. Features use mono sum.
- **Swing in output**: straight unless `--swing` is given (detected swing is reported, and used to
  snap swung source material to the grid).
- **Selection** (M3, planned): per-role palettes of 2–4 slices instead of a new sample on every step.

## Known limitations
- Non-loop files with flams ~0.3 step after the beat can be read with flipped even/odd parity
  (shows up as swing ≈ 0.3). Trimmed loops are unaffected (phase fixed at file start).
- Downbeat of non-loop files is not detected (grid phase only).

## Milestones
M1 analysis + `analyze` ✅ · M2 library index + `neighbors` ✅ · M3 generator v1 · M4 similarity/novelty/
variations · M5 Gradio listening UI.
