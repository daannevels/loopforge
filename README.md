# Loopforge

Generate new, unique top loops (hats, shakers, percussion) from **your own** sample library,
in the feel of 1–3 reference loops you like. Everything runs locally on your computer.

![Loopforge](docs/loopforge_ui.png)

## Start the app (no terminal needed)

1. On GitHub, click **Code → Download ZIP** and unzip it (or `git clone` it).
2. Open the folder and double-click:
   - **macOS:** `Start Loopforge.command`
     (if macOS says it can't be opened: System Settings → Privacy & Security → scroll down → **Open Anyway**;
     on older macOS: right-click → **Open**. Only needed once.)
   - **Windows:** `Start Loopforge.bat`
3. The first start installs what it needs (about a minute). Your browser then opens Loopforge.
   Keep the small black window open while you use it; close it to quit.

## Using it

1. **References** – drop 1–3 loops whose groove/sound you like (wav, aiff, flac).
2. **Library** – click the folder field and pick your loop library (sub-folders are included).
   The first scan analyses every loop once; later scans only look at new or changed files.
3. **Engine** – every loop keeps the exact rhythm of your reference (same hits, same accents) and swaps
   each sound for one from your library: open hat → another open hat, closed hat → another closed hat, …
   - *Variations*: how many loops per run – each one uses different samples (and source loops)
   - *Palette*: samples per sound type (1 = one open hat, one closed hat, …; 2–4 mixes in alternates)
   - *Length*: 1, 2 or 4 bars of your reference
   - *Seed*: a new random seed every run, so every run brings new sounds. Tick *Lock* to repeat a run.
4. **Generate**, then click a waveform to loop it. **A/B** flips between the loop and your first
   reference in time. **★** saves it to your favourites. Drag a card straight into Ableton / FL Studio
   (Chrome or Edge), or use ⤴ to show the file.

Your files live in `~/Loopforge` (refs, every run, favourites). Each loop has a `.json` recipe next
to it that rebuilds it exactly.

## Command line (optional)

```bash
uv run loopforge index "/path/to/library"
uv run loopforge generate --refs a.wav b.wav --library "/path/to/library" --variations 8 --palette-size 1 --out ./out
uv run loopforge render out/var_01_seed42.json
uv run loopforge ui
```

See `CLAUDE.md` for design notes.
