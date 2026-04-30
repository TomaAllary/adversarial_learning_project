import re
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d

SMOOTH_WINDOW = 10
WIN_RATE_THRESHOLD = 0.80
OPPONENT_KEY = "vs_exploiter"


def smooth(y, window):
    if len(y) < window:
        return np.array(y, dtype=float)
    return uniform_filter1d(y, size=window, mode="nearest").astype(float)


def extract_exploiter_rate(path):
    """Return the exploiter sampling rate encoded in the filename, or 0.0 for no-exploiter runs."""
    m = re.search(r'exploiter_(\d+(?:\.\d+)?)', path.stem)
    if m:
        return float(m.group(1))
    if "no_exploiter" in path.stem:
        return 0.0
    return None


def steps_to_threshold(json_path, threshold):
    with open(json_path) as f:
        data = json.load(f)

    xs, ys = [], []
    for entry in data.values():
        x = entry.get("actor_steps") or entry.get("actor_step") or entry.get("step")
        if x is None:
            continue
        v = entry.get(OPPONENT_KEY)
        if isinstance(v, dict) and "win_rate" in v:
            xs.append(x)
            ys.append(v["win_rate"])

    if not xs:
        return None

    pairs = sorted(zip(xs, ys))
    steps_arr = np.array([p[0] for p in pairs], dtype=float)
    wr_arr    = np.array([p[1] for p in pairs], dtype=float)
    wr_smooth = smooth(wr_arr, SMOOTH_WINDOW)

    idx = np.argmax(wr_smooth >= threshold)
    if wr_smooth[idx] < threshold:
        return None  # never reached threshold
    return steps_arr[idx]


def plot_threshold(eval_dir, threshold=WIN_RATE_THRESHOLD, ratio="2:1", save_path=None):
    eval_dir = Path(eval_dir)
    json_files = sorted(eval_dir.glob("*.json"))

    points = []
    for jp in json_files:
        rate = extract_exploiter_rate(jp)
        if rate is None:
            continue
        steps = steps_to_threshold(jp, threshold)
        if steps is None:
            print(f"  {jp.name}: never reached {threshold:.0%} — skipped")
            continue
        points.append((rate, steps, jp.stem))
        print(f"  {jp.name}: reached {threshold:.0%} at step {steps:,.0f}")

    if not points:
        print("No data points to plot.")
        return

    points.sort(key=lambda p: p[0])
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    w, h = (9, 4.5) if ratio == "2:1" else (6, 6)
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)

    LINE_COLOR   = "#4C9BE8"
    FILL_COLOR   = "#4C9BE8"
    MARKER_COLOR = "#4C9BE8"

    # Shaded area under the curve
    ax.fill_between(xs, ys, alpha=0.12, color=FILL_COLOR, zorder=1)

    # Connecting line
    ax.plot(xs, ys, color=LINE_COLOR, linewidth=2.2, alpha=0.85,
            solid_capstyle="round", zorder=2)

    # Markers — outer glow ring + filled centre
    ax.scatter(xs, ys, color="white",       s=220, zorder=3)
    ax.scatter(xs, ys, color=MARKER_COLOR,  s=100, zorder=4, edgecolors="white", linewidths=1.5)

    def fmt_step(v):
        if v >= 1e6:  return f"{v/1e6:.2f}M"
        if v >= 1e3:  return f"{v/1e3:.0f}K"
        return str(int(v))

    # Annotate each point with its step count
    for x, y in zip(xs, ys):
        ax.annotate(
            fmt_step(y),
            xy=(x, y), xytext=(0, 10),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, fontweight="semibold", color=LINE_COLOR,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
        )

    def fmt_y(val, _):
        if val >= 1e6:  return f"{val/1e6:.1f}M"
        if val >= 1e3:  return f"{val/1e3:.0f}K"
        return str(int(val))

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    ax.set_xlabel("Exploiter Sampling Rate", color="black", fontsize=13, labelpad=8)
    ax.set_ylabel("Actor Steps to Reach Threshold", color="black", fontsize=13, labelpad=8)
    ax.tick_params(colors="black", labelsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3245")

    ax.grid(axis="y", color="#2A3245", linewidth=0.7, linestyle="--", zorder=0)
    ax.grid(axis="x", color="#2A3245", linewidth=0.4, linestyle=":",  zorder=0)

    title = ("Steps to Near-Nash Convergence\n(by Exploiter Sampling Rate)"
             if ratio == "1:1" else
             "Steps to Near-Nash Convergence (by Exploiter Sampling Rate)")
    ax.set_title(title, color="black", fontsize=16, fontweight="semibold", pad=12)

    x_pad = (xs.max() - xs.min()) * 0.12 if len(xs) > 1 else 0.05
    ax.set_xlim(xs.min() - x_pad, xs.max() + x_pad)
    ax.set_ylim(bottom=0, top=ys.max() * 1.22)

    fig.tight_layout(pad=1.2)

    out = Path(save_path) if save_path else Path("output/plots/steps_to_threshold.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved to {out}")

    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir",  type=str,   default="output/evaluation")
    p.add_argument("--threshold", type=float, default=WIN_RATE_THRESHOLD,
                   help="Win-rate threshold (default: 0.80)")
    p.add_argument("--ratio",     choices=["1:1", "2:1"], default="2:1")
    p.add_argument("--out",       type=str,   default=None)
    args = p.parse_args()
    plot_threshold(args.eval_dir, args.threshold, args.ratio, args.out)
