# Loopforge: project summary

**What it is:** a local tool for music producers that makes new, unique **top loops** (hi-hats, shakers,
claps, percussion) from **your own sample library**. You give it 1–3 reference loops whose groove you like.
Every new loop keeps that reference's **exact rhythm**, but each sound is swapped for a different sample of
the same kind from your library. The result is a "new" loop in the same family that is not a copy.
It runs fully offline. Right now it is a Python engine plus a browser-based app that looks like a plugin.
A JUCE VST3/AU plugin for Ableton Live / FL Studio is the planned next phase.

## How it works
1. **Analysis of each loop:**
   - tempo and beat grid, taken from the filename, the loop length or the grid alignment;
   - onset detection with sample accuracy, then slicing into hits mapped to 16th-note steps;
   - per hit: loudness, attack/decay time, brightness (spectral centroid/rolloff/flatness), low/mid/high
     band energy and MFCC timbre.
2. **Library index:** every loop in your library folder is analysed once and cached in SQLite. Later scans
   only re-analyse new or changed files.
3. **Reference profile:**
   - the hits of the references are grouped into sound types (open hat, closed hat, shaker, percs) by
     clustering;
   - the clap on beats 2 and 4 is detected separately and becomes its own type.
4. **Generation:**
   - **Rhythm:** a section of a reference is copied hit for hit, including its accents and flams.
   - **Sound choice:** each sound type gets one library sample per loop. Types switched on under "Mix"
     (default: perc) get 2–4 samples that alternate per hit. Every variation deliberately picks the samples
     that sound most different from those used by earlier variations.
   - **Claps:** a clap is only replaced by a library clap. Claps that sound like the reference's own clap
     are skipped. Hats are never taken from a spot where a clap tail rings through.
   - **Assembly:** hits are placed sample-accurately, with short fades and overlaps. Hits that were cut short
     in their source loop get a synthesized natural tail. The loop is seamless: the exact number of bars,
     with tails wrapped to the start. Loudness is matched to the references and a soft limiter is applied.
5. **Output:** a 24-bit WAV plus a JSON "recipe" that rebuilds the loop bit for bit.

## The app
- **Inputs:** drop in reference loops and choose your library folder.
- **Controls:** a knob for Variations (loops per run); "Mix" switches per sound type (closed hat, open hat,
  shaker, perc, clap) that make that sound alternate between several samples within a loop, with a
  "Mix size" knob (2–4 samples); all other sound types use one sample. Also Length (1/2/4 bars) and a Seed
  that is random each run, with a Lock to repeat a run exactly.
- **Results:** each loop gets a card with a waveform and hit markers coloured by sound type. You can play it
  seamlessly looped, A/B it against the reference in time, star it as a favourite, or drag it into your DAW.
- **Run history:** earlier runs and favourites are kept in `~/Loopforge`.
- **Starting it:** a double-click launcher on Mac and Windows.

## Decisions made with the producer (me)
- **Rhythm and similarity:** keep the reference's rhythm exactly, with no swing and no similarity knob.
  Variety comes from the sounds only.
- **Variations must sound clearly different from each other,** not just use different seed numbers.
- **The original clap must not come back;** each loop gets a different clap.
- **Instrument scan reverted:** a per-hit scan (snare vs clap vs perc vs hats) was tried and reverted. It
  labelled darker hats as claps, and the simpler clap-only version sounded better.

## Current limitations
- **Library variety sets the ceiling.** My test library is 27 loops from 2 packs by one maker: only about
  37 claps differ from the reference clap, and percs are scarce.
- **Long sounds are pre-cut.** In dense library loops every hit is cut at the next 16th, so long open hats
  need the synthesized tail.
- **Sound types are decided by rules and clustering, not by a trained model,** so some edge cases get
  mislabelled.
- **It only uses hits from loops,** not one-shots.

## Planned
- A JUCE VST3/AU plugin. The engine was built to be portable (pure functions, no UI or file I/O in the
  core), and the web front-end could be reused through JUCE 8's WebView.
