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
- `uv run loopforge generate --refs a.wav b.wav --library <dir> --bars 2 --variations 8 --similarity 0.7 --seed 42 --out ./out [--audition]`
- `uv run loopforge render <var.json> [--out x.wav] [--library <moved root>]`

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
- **Generator pipeline** (`generator/`): `profile.build_profile` → `pattern.generate_pattern` →
  `select.build_palettes` + `assign_slices` → `run.generate_one` builds a `Recipe` → `recipe.render_recipe`.
  The renderer takes an injected `SliceLoader` (no I/O in the engine); `generate` and `render` use the same
  code path, so a recipe re-renders bit-identically.
- **Roles**: k-means over all ref hits in the *generation* timbre space (envelope ×3, MFCC ×0.5 —
  `GenParams.timbre_group_weights`; closed vs. open hats is mostly decay). Pick the LARGEST k (≤6) with
  silhouette ≥ 0.6×best; clusters < 3% of hits fold into the nearest. Role `long` = median decay > 120 ms.
- **Profile**: refs are cut (or tiled) into templates of the output length; per step: onset prob, role
  distribution, accent (dB vs. role level). Flams (2nd hit ≥0.1 step later on the same step) are kept as
  sub-hits.
- **Pattern**: pick a random template; per-step mutation rate 0.6·(1−sim) resamples hit/role from
  sim·profile + (1−sim)·metric prior; shifts ±1 step; density kept within ±(5%+30%·(1−sim)); velocity =
  role level + step accent + noise. At sim=1 the rhythm equals a template (novelty guard comes in M4).
- **Selection**: per role, diverse top-k (k = 4 + (1−sim)·28) around the role centroid with a fit penalty
  (gain needed > 6 dB; ringing roles prefer slices at least as long as the role's decay). Palette of 3
  sampled with weights exp(−Δd/τ); events cycle the palette (weights .55/.3/.15), never the same slice
  back-to-back, ≤ floor(40%) of events per source file; backups used when constraints block the palette.
- **Assembly**: position = round((step + swing·odd + sub_offset)·step_samples) — no drift. Short roles
  play until the next hit + 3 ms (4 ms fade-out underneath it); long roles ring up to 1 beat. 1 ms fade-in,
  gain = target − slice peak (±12 dB). Tails past the end wrap to the start (seamless). Normalise to the
  refs' mean RMS, then tanh soft-limit above 0.7×(−1 dBFS). Output = refs' (most common) sample rate.
- **Evaluation** (`diagnostics.evaluate_output`): the rendered WAV is re-analysed with the same analyzer:
  detected BPM, exact length, seam ratio, hits/bar, step agreement, role match vs. chance, accent corr,
  role-mix L1, timbre stats, file usage → `report.md` + recipe `metrics`.

## Known limitations
- Non-loop files with flams ~0.3 step after the beat can be read with flipped even/odd parity
  (shows up as swing ≈ 0.3). Trimmed loops are unaffected (phase fixed at file start).
- Downbeat of non-loop files is not detected (grid phase only).
- Library slices end at the next onset of their source loop, so long sounds (open hats) from dense loops are
  pre-truncated; selection prefers longer candidates for ringing roles but cannot recover cut tails.
- Re-analysing outputs that contain flams reports spurious swing (~0.1–0.15); the output grid is straight.

## Milestones
M1 analysis + `analyze` ✅ · M2 library index + `neighbors` ✅ · M3 generator v1 ✅ · M4 similarity/novelty/
variations · M5 Gradio listening UI.
