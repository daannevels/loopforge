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
- **Instrument scan** (`analysis/instruments.py`, ANALYZER_VERSION 3): every hit of every loop (refs and
  library) gets a class stored as the slice `role`: closed_hat, open_hat, shaker, clap, snare, perc, other,
  or `tail`. Rules: band_low ≥ 0.35 → other; midrange-heavy (band_mid ≥ 0.5, or centroid < 4 kHz and
  band_high < 0.5) and tonal (`tonal_db` ≥ 25: biggest 100 Hz–3 kHz spectral peak vs. a ~300 Hz moving
  median, weighted by band energy) → perc; midrange-heavy noise → clap or snare, decided per loop from its
  backbeat hits that stand out ≥5 dB in midrange level: median attack ≥ 8 ms = clap (several hand bursts,
  late peak), else snare (one stick transient); ghost notes take the loop's choice; noisy-mid hits 1–3 steps
  after one and ≥8 dB quieter = `tail`. Bright hits: attack ≥ 15 ms & decay < 130 ms → shaker, decay ≥ 130 →
  open_hat, else closed_hat. Snare "body" (150–500 Hz) is useless: top loops are high-passed.
  Verified on the user's refs: 01 = clap + tonal percs + hats, 04 = hats + snare, 05 = hats + shaker +
  snare pattern (backbeat with flam + ghosts on 7/15). Synthetic claps get split by the onset detector, so
  clap/shaker rules are unit-tested on measured feature values instead of synthetic audio.
- **Roles = instruments found in the refs** (no k-means any more; `cluster_roles` is unused). A ref with only
  hats gets only hats. `tail` ref hits keep their place and take the instrument clean hits have on that step
  position. Clap/snare roles are `long` (tail rings on). Library candidates must have the SAME instrument
  label, are never `tainted` (`instruments.tainted_mask`: ≤3 steps after a clap/snare and mid-heavy — that
  carried the pack clap into hats), need ≤12 dB gain, and (non-drums) decay 0.6–2× the ref's; constraints
  relax step by step if too few. Library claps/snares whose body sounds like a ref clap/snare (log-spectrum
  corr ≥ 0.85, `service.exclude_ref_clap_copies`) are skipped.
  Real data (8 variations per ref): 0 hits with a different instrument; ref 05's snares from 7 loops
  (max similarity to the original 0.80), ref 01's claps from 4 loops, ref 04's snares from 6 loops.
- **Profile**: refs are cut (or tiled) into templates of the output length; per step: onset prob, role
  distribution, accent (dB vs. role level). Flams (2nd hit ≥0.1 step later on the same step) are kept as
  sub-hits.
- **Product decision (user, final)**: only "simple" is exposed. Rhythm = an exact copy of a reference
  section (same steps, sound type and level per hit, the ref's own flams; empty steps stay empty). No swing,
  no similarity knob (fixed 1.0). UI/CLI expose Variations, Palette, Length, Seed (random per run unless
  locked). `creative` (M3 mutation/palette/swing engine) stays in code, hidden (`--mode creative`).
- **Simple-mode sound choice** (`select.same_type_candidates` + `choose_role_samples`): same instrument
  label (see Instrument scan), pool = closest 150, max 4 per source file (loops repeat the same hit). Each variation picks `palette_size` (default 1) per sound type,
  preferring samples AND files unused by earlier variations of the run (`RunMemory`), never sharing a
  sample between roles. Variation 1 picks uniformly at random; later ones pick at random among the 20% of
  candidates that are MOST different (max-min timbre distance) from what earlier variations used for that
  sound type (`_pick_most_different`). 10% was tested: samples differ more but whole loops less.
- **Seeds**: the run seed (random per run unless locked) derives one independent random seed per variation
  (`rng(run_seed).choice(1e6, n, replace=False)`); recipes store both `seed` and `run_seed`. Adjacent seed
  numbers were never "similar" for the RNG — the user's "too similar" was the narrow pool + pure random picks.
  Measured over 3 runs: sample spread per sound type 1.40 → 1.73, loop-to-loop timbre distance 1.53 → 1.93.
  Palette > 1 mixes the extra samples in by weights .55/.3/.15/.1, repeats allowed.
  Measured on the user's refs/library (3 seeds × 8): 32–33 distinct samples per run, none reused,
  22 of 24 usable files; level dip between 16ths 15.0–15.6 dB median / 21.8–23.5 dB p90 (refs 16.4 / 27.7).
- **Claps/snares history**: user heard the original clap in new loops: 80 of 117 library claps were (near)
  copies of the pack clap, and hats taken right after a clap carried its tail. Fixed by the scan above.
- History: "choppy" feedback at creative 70% = ~9 of 32 hits rewritten per loop + hard 4 ms cuts of
  slices already truncated in dense source loops.
- **Tail extension** (`generator/tail.py`): if a slice is shorter than the slot it must fill (it was cut by
  the next hit in its source loop), it rings on naturally: overlap-added random (sometimes reversed) 20 ms
  grains from its own last 30 ms before the cut, RMS-matched, shaped with its measured decay (−20 dB per
  `decay_ms`) up to ~−40 dB, equal-power 6 ms crossfade. Stored per event (`tail_ms`, `tail_decay_ms`,
  `tail_seed`) so renders stay bit-identical. Fades are 10 ms out with 10 ms overlap into the next hit.
- **Pattern** (creative, hidden): pick a random template; per-step mutation rate 0.6·(1−sim) resamples hit/role from
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

## UI (`loopforge ui`, `loopforge/ui/`)
- Stdlib `ThreadingHTTPServer` on 127.0.0.1 (port 8765, next free if taken) + static vanilla HTML/CSS/JS
  (no build step, no CDN — works offline). Chosen over Gradio so it looks like a plugin and so the front-end
  can be reused in JUCE 8's WebView UI later.
- `service.py` holds the orchestration shared by CLI and UI (`load_ref`, `run_generation`, `GenRequest`).
- `ui/server.py` `App` keeps state; indexing/generation run in background threads, the page polls `/api/state`.
  Workspace `~/Loopforge` (or `$LOOPFORGE_HOME`): `refs/` (uploaded refs), `runs/<timestamp>/`, `favorites/`,
  `settings.json` (knobs, library, ref paths — restored on start). `/api/file` only serves files under the
  workspace, the library or a ref's folder.
- Front-end: knobs (drag/wheel/arrows, double-click reset), WebAudio looping playback (`loop=true`, so the seam
  is audible), A/B vs ref 1 in time, favourites, drag-out to DAW via `DownloadURL` (Chrome/Edge).
- Launchers: `Start Loopforge.command` (macOS) / `Start Loopforge.bat` (Windows) install uv if needed,
  `uv sync`, `uv run loopforge ui`.

## Known limitations
- Non-loop files with flams ~0.3 step after the beat can be read with flipped even/odd parity
  (shows up as swing ≈ 0.3). Trimmed loops are unaffected (phase fixed at file start).
- Downbeat of non-loop files is not detected (grid phase only).
- Library slices end at the next onset of their source loop, so long sounds (open hats) from dense loops are
  pre-truncated; selection prefers longer candidates for ringing roles but cannot recover cut tails.
- Onsets: in a synthetic stress test ~4% of quiet closed hats right after a loud open hat (same HP-noise
  spectrum) are missed at delta 0.1. More sensitivity (delta 0.07–0.08) fixes most but adds 1–3 extra
  hits to the user's refs — unverified by ear, so the default stays until listening feedback.
- Re-analysing outputs that contain flams reports spurious swing (~0.1–0.15); the output grid is straight.

## Milestones
M1 analysis + `analyze` ✅ · M2 library index + `neighbors` ✅ · M3 generator v1 ✅ · M4 reshaped by the user's
decision: rhythm is copied on purpose (no novelty guard on rhythm); variation = different sounds per variation ✅ ·
M5 listening UI ✅ (local web UI instead of Gradio).
