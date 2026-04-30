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


def plot_combined(eval_dir, ratio="2:1", save_dir=None):
    eval_dir = Path(eval_dir)
    json_files = sorted(eval_dir.glob("*.json"),
                        key=lambda p: (0 if "no_exploiter" in p.stem else 1, p.stem))
    if not json_files:
        print(f"No JSON files found in {eval_dir}")
        return

    w, h = (9, 4.5) if ratio == "2:1" else (6, 6)
    save_dir = Path(save_dir) if save_dir else Path("output/plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    def fmt_x(val, _):
        if val >= 1e6:  return f"{val/1e6:.1f}M"
        if val >= 1e3:  return f"{val/1e3:.0f}K"
        return str(int(val))

    for opponent_key in OPPONENT_KEYS:
        fig, ax = plt.subplots(figsize=(w, h), dpi=130)

        handles = []
        for i, json_path in enumerate(json_files):
            color = RUN_COLORS[i % len(RUN_COLORS)]
            xs, ys = load_series(json_path, opponent_key)
            if xs is None:
                continue
            ys_smooth = smooth(ys, SMOOTH_WINDOW)
            ax.plot(xs, ys, color=color, alpha=0.5, linewidth=0.8, zorder=2)
            line, = ax.plot(xs, ys_smooth, color=color, linewidth=2.2, alpha=0.95,
                            solid_capstyle="round", zorder=3, label=make_run_label(json_path))
            handles.append(line)

        if not handles:
            plt.close(fig)
            continue

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

        ax.set_title(OPPONENT_LABELS[opponent_key], color="black", fontsize=16,
                     fontweight="semibold", pad=12)

        ax.legend(handles=handles, frameon=True, framealpha=0.85, labelcolor="black",
                  fontsize=11, loc="lower right", handlelength=1.8,
                  handletextpad=0.6, borderpad=0.7)

        fig.tight_layout(pad=1.2)

        out = save_dir / f"combined_{opponent_key}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"Saved to {out}")
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", type=str, default="output/evaluation",
                   help="Directory containing evaluation JSON files")
    p.add_argument("--ratio", choices=["1:1", "2:1"], default="2:1")
    p.add_argument("--out_dir", type=str, default=None, help="Override output directory")
    args = p.parse_args()
    plot_combined(args.eval_dir, args.ratio, args.out_dir)
