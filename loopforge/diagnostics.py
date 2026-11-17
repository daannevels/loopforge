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


def _len_label(t) -> str:
    if not t.is_loop:
        return f"{t.beats} beats (NOT a whole number of beats; treated as non-looping)"
    if t.beats and t.beats % 4:
        return f"{t.beats} beats = {t.beats / 4:g} bars (exact loop)"
    return f"{t.bars} bars (exact loop)"


def analysis_summary(a: LoopAnalysis) -> str:
    t = a.tempo
    from collections import Counter

    roles = Counter(s.role for s in a.slices)
    density = len(a.slices) / max(1, t.bars)
    lines = [
        f"file        {a.name}",
        f"format      {a.sr} Hz, {a.channels} ch, {a.duration_s:.4f} s ({a.n_samples} samples)",
        f"tempo       {t.bpm:.3f} BPM  [method={t.method}, grid confidence={t.confidence:.2f}]",
        f"length      {_len_label(t)}, {a.steps_total} steps",
        f"swing       {a.swing:.3f} steps (0 = straight, 0.33 = triplet feel)",
        f"grid offset {a.grid_offset:+.3f} steps",
        f"slices      {len(a.slices)} ({density:.1f} per bar)  roles: "
        + ", ".join(f"{k}={v}" for k, v in sorted(roles.items())),
    ]
    lines += [f"note        {n}" for n in a.notes]
    return "\n".join(lines)


# --- generator evaluation -------------------------------------------------------

def timbre_stats(analyses: list[LoopAnalysis]) -> dict[str, float]:
    """Mean slice timbre over one or more analyses (hits weighted equally)."""
    rows = [a.features for a in analyses if len(a.features)]
    if not rows:
        return {}
    F = np.concatenate(rows)
    from loopforge.models import FEATURE_INDEX as FI

    return {
        "centroid_hz": float(np.median(F[:, FI["centroid_hz"]])),
        "band_high": float(np.mean(F[:, FI["band_high"]])),
        "band_mid": float(np.mean(F[:, FI["band_mid"]])),
        "decay_ms": float(np.median(F[:, FI["decay_ms"]])),
        "flatness": float(np.median(F[:, FI["flatness"]])),
    }


def seam_ratio(x: np.ndarray) -> float:
    """Jump across the loop point relative to the loop's own 99th-percentile sample-to-sample jump.

    <= ~1 means the wrap is no rougher than the audio itself (no click)."""
    x = np.atleast_2d(x)
    d = np.abs(np.diff(x, axis=1))
    typical = float(np.percentile(d, 99)) + 1e-9
    return float(np.max(np.abs(x[:, 0] - x[:, -1])) / typical)


def rhythm_metrics(out_analysis: LoopAnalysis, profile, space) -> dict:
    """How the output's detected hits relate to the refs' rhythm profile.

    - step_agreement: fraction of steps where hit/no-hit agrees with the refs' majority (onset prob >= 0.5)
    - role_match: mean probability (from the refs) of the role each detected hit is closest to, at its step;
      `role_chance` is what random roles would score
    - accent_corr: correlation of per-step output level with the refs' expected level (role level + accent)
    - role_mix_l1: L1 distance between output and reference role shares (0 = same mix, 2 = disjoint)
    """
    from loopforge.models import FEATURE_INDEX as FI

    n = profile.n_steps
    hit = np.zeros(n, bool)
    level = np.full(n, np.nan)
    match: list[float] = []
    counts = np.zeros(len(profile.roles))
    expected = np.array([
        sum(profile.step_role_prob[o, r.id] * r.level_db for r in profile.roles) + profile.accent_db[o]
        for o in range(n)
    ])
    if len(out_analysis.slices):
        C = np.stack([r.centroid for r in profile.roles])
        E = space.embed(out_analysis.features)
        roles = np.argmin(((E[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for i, s in enumerate(out_analysis.slices):
            o = s.step % n
            r = int(roles[i])
            counts[r] += 1
            match.append(float(profile.step_role_prob[o, r]))
            pk = float(out_analysis.features[i, FI["peak_db"]])
            level[o] = pk if np.isnan(level[o]) else max(level[o], pk)
            hit[o] = True
    agree = float(np.mean(hit == (profile.onset_prob >= 0.5)))
    ok = ~np.isnan(level)
    acc = float(np.corrcoef(level[ok], expected[ok])[0, 1]) if ok.sum() > 2 and level[ok].std() > 0 else 0.0
    shares = profile.role_shares
    mix = counts / max(1.0, counts.sum())
    return {
        "step_agreement": round(agree, 3),
        "role_match": round(float(np.mean(match)) if match else 0.0, 3),
        "role_chance": round(float((shares**2).sum()), 3),
        "accent_corr": round(acc, 3),
        "role_mix_l1": round(float(np.abs(mix - shares).sum()), 3),
    }


def evaluate_output(audio: np.ndarray, out_analysis: LoopAnalysis, recipe, profile, ref_timbre: dict,
                    space=None) -> dict:
    from collections import Counter

    files = Counter(e.source_rel for e in recipe.events)
    t = out_analysis.tempo
    out_t = timbre_stats([out_analysis])
    return {
        "detected_bpm": round(t.bpm, 3),
        "detected_method": t.method,
        "detected_loop": t.is_loop,
        "bpm_error": round(abs(t.bpm - recipe.bpm), 3),
        "length_samples": int(audio.shape[1]),
        "length_exact": int(audio.shape[1]) == recipe.n_samples,
        "seam_ratio": round(seam_ratio(audio), 3),
        "peak_dbfs": round(20 * np.log10(float(np.abs(audio).max()) + 1e-12), 2),
        "rms_dbfs": round(20 * np.log10(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) + 1e-12), 2),
        "events": len(recipe.events),
        "hits_per_bar": round(len(out_analysis.slices) / max(1e-9, recipe.bars), 2),
        "ref_hits_per_bar": round(profile.hits_per_template / recipe.bars, 2),
        **(rhythm_metrics(out_analysis, profile, space) if space is not None else {}),
        "timbre": {k: round(v, 3) for k, v in out_t.items()},
        "ref_timbre": {k: round(v, 3) for k, v in ref_timbre.items()},
        "files_used": len(files),
        "max_file_share": round(max(files.values()) / max(1, len(recipe.events)), 3),
        "slices_used": len({(e.source_rel, e.slice_index) for e in recipe.events}),
    }


def generation_report(rows: list[dict], profile, refs: list[str], args_desc: str) -> str:
    rt = rows[0]["metrics"]["ref_timbre"] if rows else {}
    lines = [
        "# loopforge generation report",
        "",
        f"- command: `{args_desc}`",
        f"- refs: {', '.join(refs)}",
        f"- tempo {profile.bpm:g} BPM, {profile.n_steps} steps, sr {profile.sr}",
        f"- ref hits/bar {profile.hits_per_template / (profile.n_steps / 16):.1f}; ref timbre: centroid "
        f"{rt.get('centroid_hz', 0):.0f} Hz, high-band {rt.get('band_high', 0):.2f}, decay {rt.get('decay_ms', 0):.0f} ms",
        "",
        "## Roles found in the refs",
        "",
        "| id | name | share | centroid Hz | decay ms | attack ms | high band | level dB | rings |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in profile.roles:
        d = r.to_dict()
        lines.append(f"| {d['id']} | {d['name']} | {d['share']:.2f} | {d['centroid_hz']:.0f} | {d['decay_ms']:.0f} | "
                     f"{d['attack_ms']:.1f} | {d['band_high']:.2f} | {d['level_db']:.1f} | {'yes' if d['long'] else ''} |")
    lines += [
        "",
        "## Variations",
        "",
        "| file | seed | detected BPM | exact length | seam | hits/bar | step agree | role match | accent corr | "
        "role mix L1 | centroid Hz | high band | decay ms | files | max file share | template |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r["metrics"]
        t = m["timbre"]
        lines.append(
            f"| {r['file']} | {r['seed']} | {m['detected_bpm']:g} ({m['detected_method']}) | "
            f"{'yes' if m['length_exact'] else 'NO'} | {m['seam_ratio']:.2f} | {m['hits_per_bar']:.1f} | "
            f"{m.get('step_agreement', 0):.2f} | {m.get('role_match', 0):.2f} (chance {m.get('role_chance', 0):.2f}) | "
            f"{m.get('accent_corr', 0):.2f} | {m.get('role_mix_l1', 0):.2f} | "
            f"{t.get('centroid_hz', 0):.0f} | {t.get('band_high', 0):.2f} | "
            f"{t.get('decay_ms', 0):.0f} | {m['files_used']} | {m['max_file_share']:.2f} | {r['template']} |"
        )
    lines += [
        "",
        "seam = jump across the loop point / 99th-percentile sample-to-sample jump inside the loop (<= 1: no click).",
        "step agree = fraction of 16th steps where hit / no-hit matches the refs' majority.",
        "role match = mean ref probability of each detected hit's role at its step (chance = random roles).",
        "accent corr = correlation of per-step output level with the refs' expected level.",
        "role mix L1 = distance between output and ref role shares (0 = same mix, 2 = disjoint).",
        "All metrics come from re-analysing the rendered WAV with the same analyzer used on the refs.",
    ]
    return "\n".join(lines) + "\n"
