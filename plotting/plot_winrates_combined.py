import re
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d

OPPONENT_KEYS = ["vs_exploiter", "vs_scripted", "vs_random"]
OPPONENT_LABELS = {
    "vs_exploiter": "Main Agent Exploitability (vs. Trained Exploiter)",
    "vs_scripted":  "Main Agent vs. Scripted Agent",
    "vs_random":    "Main Agent vs. Random Agent",
}

SMOOTH_WINDOW = 10

# Distinct colors for different runs
RUN_COLORS = [
    "#4C9BE8",
    "#E8834C",
    "#5DBE7A",
    "#C872D8",
    "#E8C84C",
    "#E84C6A",
    "#4CE8D8",
    "#A0A0A0",
]


def smooth(y, window):
    if len(y) < window:
        return np.array(y, dtype=float)
    return uniform_filter1d(y, size=window, mode="nearest").astype(float)


def load_series(json_path, opponent_key):
    with open(json_path) as f:
        data = json.load(f)
    xs, ys = [], []
    for entry in data.values():
        x = entry.get("actor_steps") or entry.get("actor_step") or entry.get("step")
        if x is None:
            continue
        v = entry.get(opponent_key)
        if isinstance(v, dict) and "win_rate" in v:
            xs.append(x)
            ys.append(v["win_rate"])
    if not xs:
        return None, None
    pairs = sorted(zip(xs, ys))
    return (
        np.array([p[0] for p in pairs], dtype=float),
        np.array([p[1] for p in pairs], dtype=float),
    )


def make_run_label(json_path):
    stem = Path(json_path).stem
    m = re.search(r'exploiter_(\d+(?:\.\d+)?)', stem)
    if m:
        pct = int(float(m.group(1)) * 100)
        return f"{pct}% Exploiter Sampling"
    if "no_exploiter" in stem:
        return "0% Exploiter Sampling"
    return stem.replace("_", " ").title()


def _setup_axes(ax, opponent_key, x_max, fmt_x, wrap_title=False):
    ax.set_xlim(0, x_max * 1.02)
    ax.set_ylim(-0.02, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_x))
    ax.set_xlabel("Actor Steps", color="black", fontsize=13, labelpad=8)
    ax.set_ylabel("Win Rate",    color="black", fontsize=13, labelpad=8)
    ax.tick_params(colors="black", labelsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3245")
    ax.grid(axis="y", color="#2A3245", linewidth=0.7, linestyle="--", zorder=1)
    ax.grid(axis="x", color="#2A3245", linewidth=0.4, linestyle=":",  zorder=1)
    title = OPPONENT_LABELS[opponent_key]
    if wrap_title:
        title = title.replace(" (", "\n(").replace(" vs. ", "\nvs. ")
    ax.set_title(title, color="black", fontsize=16, fontweight="semibold", pad=12)


def _load_all_series(json_files, opponent_key):
    series = []
    for i, json_path in enumerate(json_files):
        color = RUN_COLORS[i % len(RUN_COLORS)]
        xs, ys = load_series(json_path, opponent_key)
        if xs is not None:
            series.append((xs, ys, color, make_run_label(json_path)))
    return series


def _sorted_json_files(eval_dir):
    return sorted(Path(eval_dir).glob("*.json"),
                  key=lambda p: (0 if "no_exploiter" in p.stem else 1, p.stem))


def _fmt_x(val, _):
    if val >= 1e6:  return f"{val/1e6:.1f}M"
    if val >= 1e3:  return f"{val/1e3:.0f}K"
    return str(int(val))


def plot_combined(eval_dir, ratio="2:1", save_dir=None):
    json_files = _sorted_json_files(eval_dir)
    if not json_files:
        print(f"No JSON files found in {eval_dir}")
        return

    w, h = (9, 4.5) if ratio == "2:1" else (6, 6)
    save_dir = Path(save_dir) if save_dir else Path("output/plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    for opponent_key in OPPONENT_KEYS:
        all_series = _load_all_series(json_files, opponent_key)
        if not all_series:
            continue

        fig, ax = plt.subplots(figsize=(w, h), dpi=130)
        x_max = max(xs[-1] for xs, *_ in all_series)
        _setup_axes(ax, opponent_key, x_max, _fmt_x, wrap_title=(ratio == "1:1"))

        handles = []
        for xs, ys, color, label in all_series:
            ys_smooth = smooth(ys, SMOOTH_WINDOW)
            ax.plot(xs, ys, color=color, alpha=0.5, linewidth=0.8, zorder=2)
            line, = ax.plot(xs, ys_smooth, color=color, linewidth=2.2, alpha=0.95,
                            solid_capstyle="round", zorder=3, label=label)
            handles.append(line)

        ax.legend(handles=handles, frameon=True, framealpha=0.85, labelcolor="black",
                  fontsize=11, loc="lower right", handlelength=1.8,
                  handletextpad=0.6, borderpad=0.7)
        fig.tight_layout(pad=1.2)

        out = save_dir / f"combined_{opponent_key}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"Saved {out}")
        plt.show()


# Max frames to keep GIF file size manageable; data is subsampled if needed.
_MAX_GIF_FRAMES = 300


def animate_combined(eval_dir, ratio="2:1", save_dir=None, fps=40):
    from matplotlib.animation import FuncAnimation, PillowWriter

    json_files = _sorted_json_files(eval_dir)
    if not json_files:
        print(f"No JSON files found in {eval_dir}")
        return

    w, h = (9, 4.5) if ratio == "2:1" else (6, 6)
    save_dir = Path(save_dir) if save_dir else Path("output/plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    for opponent_key in OPPONENT_KEYS:
        all_series = _load_all_series(json_files, opponent_key)
        if not all_series:
            continue

        # Union of all x values across runs, sorted — defines the animation timeline.
        all_xs = np.unique(np.concatenate([xs for xs, *_ in all_series]))
        all_xs.sort()

        # Drop early frames where any run hasn't accumulated enough points for
        # the smoother to kick in (SMOOTH_WINDOW points required per series).
        start_x = max(
            xs[SMOOTH_WINDOW - 1] if len(xs) >= SMOOTH_WINDOW else xs[-1]
            for xs, *_ in all_series
        )
        trimmed_xs = all_xs[all_xs >= start_x]

        # Subsample to at most _MAX_GIF_FRAMES frames.
        if len(trimmed_xs) > _MAX_GIF_FRAMES:
            idx = np.round(np.linspace(0, len(trimmed_xs) - 1, _MAX_GIF_FRAMES)).astype(int)
            frame_xs = trimmed_xs[idx]
        else:
            frame_xs = trimmed_xs

        x_max = all_xs[-1]

        fig, ax = plt.subplots(figsize=(w, h), dpi=130)
        _setup_axes(ax, opponent_key, x_max, _fmt_x, wrap_title=(ratio == "1:1"))

        # Pre-create one raw + one smooth line per run; all start empty.
        raw_lines, smooth_lines, handles = [], [], []
        for xs, ys, color, label in all_series:
            raw, = ax.plot([], [], color=color, alpha=0.5, linewidth=0.8, zorder=2)
            smo, = ax.plot([], [], color=color, linewidth=2.2, alpha=0.95,
                           solid_capstyle="round", zorder=3, label=label)
            raw_lines.append(raw)
            smooth_lines.append(smo)
            handles.append(smo)

        ax.legend(handles=handles, frameon=True, framealpha=0.85, labelcolor="black",
                  fontsize=11, loc="lower right", handlelength=1.8,
                  handletextpad=0.6, borderpad=0.7)
        fig.tight_layout(pad=1.2)

        def update(frame, _frame_xs=frame_xs, _series=all_series,
                   _raw=raw_lines, _smo=smooth_lines):
            current_x = _frame_xs[frame]
            for (xs, ys, *_), raw_line, smo_line in zip(_series, _raw, _smo):
                mask = xs <= current_x
                xs_v, ys_v = xs[mask], ys[mask]
                if len(xs_v) == 0:
                    raw_line.set_data([], [])
                    smo_line.set_data([], [])
                else:
                    raw_line.set_data(xs_v, ys_v)
                    smo_line.set_data(xs_v, smooth(ys_v, SMOOTH_WINDOW))
            return _raw + _smo

        anim = FuncAnimation(fig, update, frames=len(frame_xs), blit=True, interval=1000 / fps)

        out = save_dir / f"combined_{opponent_key}.gif"
        print(f"Rendering {len(frame_xs)} frames → {out} …")
        anim.save(str(out), writer=PillowWriter(fps=fps))
        print(f"Saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", type=str, default="output/evaluation",
                   help="Directory containing evaluation JSON files")
    p.add_argument("--ratio", choices=["1:1", "2:1"], default="2:1")
    p.add_argument("--out_dir", type=str, default=None, help="Override output directory")
    p.add_argument("--gif", action="store_true",
                   help="Render animated GIFs instead of static PNGs")
    p.add_argument("--fps", type=int, default=40,
                   help="Frames per second for GIF output (ignored without --gif)")
    args = p.parse_args()

    if args.gif:
        animate_combined(args.eval_dir, args.ratio, args.out_dir, args.fps)
    else:
        plot_combined(args.eval_dir, args.ratio, args.out_dir)
