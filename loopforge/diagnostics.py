"""Human-facing diagnostics: spectrogram PNGs with grid + slice markers, text tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from loopforge.audio_io import to_mono  # noqa: E402
from loopforge.models import LoopAnalysis  # noqa: E402

ROLE_COLORS = {"closed": "#00e5ff", "open": "#ffb000", "shaker": "#7CFC00", "perc": "#ff4fd8", "": "#ffffff"}


def plot_analysis(audio: np.ndarray, analysis: LoopAnalysis, out_png: str | Path, steps_per_beat: int = 4,
                  beats_per_bar: int = 4) -> Path:
    """Waveform + log-frequency spectrogram with 16th grid, beats, bars and slice markers."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    mono = to_mono(audio)
    sr = analysis.sr
    t_end = len(mono) / sr

    n_fft, hop = 2048, 256
    pad = np.pad(mono, (n_fft // 2, n_fft // 2))
    n_frames = 1 + (len(pad) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    S = np.abs(np.fft.rfft(pad[idx] * np.hanning(n_fft)[None, :], axis=1)).T
    S_db = 20 * np.log10(S / (S.max() + 1e-12) + 1e-6)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    times = np.arange(n_frames) * hop / sr

    width = min(40.0, max(12.0, 3.0 * analysis.tempo.bars * 2))
    fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(width, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 3]}, constrained_layout=True)
    tw = np.arange(len(mono)) / sr
    decim = max(1, len(mono) // 20000)
    ax_w.plot(tw[::decim], mono[::decim], lw=0.4, color="#333")
    ax_w.set_ylim(-1.05, 1.05)
    ax_w.set_ylabel("amp")

    ax_s.pcolormesh(times, freqs[1:], S_db[1:], shading="auto", cmap="magma", vmin=-80, vmax=0)
    ax_s.set_yscale("log")
    ax_s.set_ylim(40, sr / 2)
    ax_s.set_ylabel("Hz")
    ax_s.set_xlabel("time (s)")
    ax_s.set_xlim(0, t_end)

    tempo = analysis.tempo
    step_s = tempo.step_len_s(steps_per_beat)
    n_steps = int(np.ceil((t_end - tempo.phase_s) / step_s)) + 1
    for k in range(n_steps):
        t = tempo.phase_s + k * step_s
        if t > t_end:
            break
        if k % (steps_per_beat * beats_per_bar) == 0:
            style = dict(color="white", lw=1.6, alpha=0.9)
        elif k % steps_per_beat == 0:
            style = dict(color="white", lw=0.8, alpha=0.6)
        else:
            style = dict(color="white", lw=0.4, alpha=0.25, ls=":")
        for ax in (ax_w, ax_s):
            ax.axvline(t, **style)

    for s in analysis.slices:
        t = s.start / sr
        c = ROLE_COLORS.get(s.role, "#fff")
        ax_s.axvline(t, color=c, lw=1.0, alpha=0.95)
        ax_w.axvline(t, color=c, lw=1.0, alpha=0.95)
        ax_s.text(t, sr / 2 * 0.8, str(s.step), color=c, fontsize=6, rotation=90, va="top", ha="right")

    handles = [plt.Line2D([0], [0], color=c, lw=2) for r, c in ROLE_COLORS.items() if r]
    ax_s.legend(handles, [r for r in ROLE_COLORS if r], loc="lower right", fontsize=7, framealpha=0.5)
    ax_w.set_title(
        f"{analysis.name} | {tempo.bpm:.2f} BPM ({tempo.method}, conf {tempo.confidence:.2f}) | "
        f"{tempo.bars} bars{'' if tempo.is_loop else ' (not a loop)'} | swing {analysis.swing:.3f} | "
        f"{len(analysis.slices)} slices  — markers: slice start (colour = role), number = 16th step",
        fontsize=9,
    )
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def step_label(step: int, steps_per_beat: int = 4, beats_per_bar: int = 4) -> str:
    """0-based step -> 'bar.beat.16th' (1-based)."""
    spb = steps_per_beat * beats_per_bar
    return f"{step // spb + 1}.{(step % spb) // steps_per_beat + 1}.{step % steps_per_beat + 1}"


def analysis_table(a: LoopAnalysis) -> str:
    cols = ["#", "time_s", "step", "pos", "offset", "role", "rms_db", "atk_ms", "dec_ms", "cent_hz",
            "flat", "low", "mid", "high", "dur_ms"]
    rows = [cols]
    for i, s in enumerate(a.slices):
        f = {n: float(a.feature(n)[i]) for n in ("rms_db", "attack_ms", "decay_ms", "centroid_hz", "flatness",
                                                 "band_low", "band_mid", "band_high", "duration_ms")}
        rows.append([
            str(i), f"{s.start / a.sr:.4f}", str(s.step), step_label(s.step), f"{s.offset:+.3f}", s.role,
            f"{f['rms_db']:.1f}", f"{f['attack_ms']:.1f}", f"{f['decay_ms']:.0f}", f"{f['centroid_hz']:.0f}",
            f"{f['flatness']:.3f}", f"{f['band_low']:.2f}", f"{f['band_mid']:.2f}", f"{f['band_high']:.2f}",
            f"{f['duration_ms']:.0f}",
        ])
    widths = [max(len(r[c]) for r in rows) for c in range(len(cols))]
    return "\n".join("  ".join(v.rjust(w) for v, w in zip(r, widths)) for r in rows)


def analysis_summary(a: LoopAnalysis) -> str:
    t = a.tempo
    from collections import Counter

    roles = Counter(s.role for s in a.slices)
    density = len(a.slices) / max(1, t.bars)
    lines = [
        f"file        {a.name}",
        f"format      {a.sr} Hz, {a.channels} ch, {a.duration_s:.4f} s ({a.n_samples} samples)",
        f"tempo       {t.bpm:.3f} BPM  [method={t.method}, grid confidence={t.confidence:.2f}]",
        f"length      {t.bars} bars{' (exact loop)' if t.is_loop else ' (NOT a whole number of bars)'}, "
        f"{a.steps_total} steps",
        f"swing       {a.swing:.3f} steps (0 = straight, 0.33 = triplet feel)",
        f"grid offset {a.grid_offset:+.3f} steps",
        f"slices      {len(a.slices)} ({density:.1f} per bar)  roles: "
        + ", ".join(f"{k}={v}" for k, v in sorted(roles.items())),
    ]
    lines += [f"note        {n}" for n in a.notes]
    return "\n".join(lines)
