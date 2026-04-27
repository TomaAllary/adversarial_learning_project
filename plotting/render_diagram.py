import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 6.0))
ax.axis("off")
fig.patch.set_facecolor("white")

C_POOL   = "#aed6f1"
C_MAIN   = "#a9dfbf"
C_FROZEN = "#f9e79f"
C_EXPL   = "#f1948a"
C_BORDER = "#444444"
C_ARROW  = "#333333"

BW, BH = 2.7, 0.62

# 2x2 grid
POOL_X,   POOL_Y   = 3.2,  4.1
MAIN_X,   MAIN_Y   = 9.8,  4.1
EXPL_X,   EXPL_Y   = 3.2,  2.6
FROZEN_X, FROZEN_Y = 9.8,  2.6

def L(x): return x - BW / 2
def R(x): return x + BW / 2
def T(y): return y + BH / 2
def B(y): return y - BH / 2


def stacked_box(ax, x, y, label, sublabel, color, stacked=False):
    n_cards = 3
    card_shift = 0.13
    if stacked:
        for i in range(n_cards - 1, 0, -1):
            shift = i * card_shift
            shadow = FancyBboxPatch(
                (L(x) + shift, B(y) - shift), BW, BH,
                boxstyle="round,pad=0.12",
                linewidth=1.5, edgecolor=C_BORDER,
                facecolor=color, alpha=0.55, zorder=3 - i,
            )
            ax.add_patch(shadow)

    rect = FancyBboxPatch(
        (L(x), B(y)), BW, BH,
        boxstyle="round,pad=0.12",
        linewidth=2.2, edgecolor=C_BORDER, facecolor=color, zorder=4,
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.10, label, ha="center", va="center",
            fontsize=15, fontweight="bold", color="#1a1a1a", zorder=5)
    if sublabel:
        ax.text(x, y - 0.12, sublabel, ha="center", va="center",
                fontsize=12, color="#444444", style="italic", zorder=5)


def arrow(ax, x1, y1, x2, y2, rad=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=C_ARROW, lw=2.4, mutation_scale=24,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=6,
    )


def label(ax, x, y, text, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va,
            fontsize=12, color="#222222", zorder=7)


# ── Outer dashed border ──────────────────────────────────────────────────────
_BP = 0.2  # uniform border padding
_bx = L(POOL_X) - _BP
_by = B(FROZEN_Y) - _BP
_bw = R(MAIN_X) - L(POOL_X) + 2 * _BP
_bh = T(POOL_Y)  - B(FROZEN_Y) + 2 * _BP

_SP = 0.15  # extra margin so the dashed stroke isn't clipped
ax.set_xlim(_bx - _SP, _bx + _bw + _SP)
ax.set_ylim(_by - _SP, _by + _bh + _SP)

outer = FancyBboxPatch(
    (_bx, _by), _bw, _bh,
    boxstyle="round,pad=0.1",
    linewidth=2, edgecolor="#888888", facecolor="#fafafa",
    linestyle="--", zorder=0,
)
ax.add_patch(outer)

# ── Boxes ────────────────────────────────────────────────────────────────────
stacked_box(ax, POOL_X, POOL_Y, "Population Pool",
            r"{ $\pi_M^{t-1}$, $\pi_M^{t-2}$, $\pi_M^{t-3}$, $\pi_E$ }",
            C_POOL, stacked=True)

stacked_box(ax, MAIN_X, MAIN_Y, r"Main Agent  $\pi_M^t$",
            "(R-NaD, trains continuously)",
            C_MAIN, stacked=False)

stacked_box(ax, EXPL_X, EXPL_Y, r"Exploiter  $\pi_E^{t*}$",
            "(PPO)",
            C_EXPL, stacked=False)

stacked_box(ax, FROZEN_X, FROZEN_Y, r"Frozen Snapshot  $\pi_M^{t*}$",
            "(weights fixed at launch)",
            C_FROZEN, stacked=False)

# ── Arrows ───────────────────────────────────────────────────────────────────
PAD  = 0.18   # gap between arrowhead/tail and box edge
LPAD = 0.13   # label offset perpendicular to arrow (~20 px at 180 dpi)

# Arrow y/x coordinates (for midpoint calculation)
_hy_upper = MAIN_Y + 0.22   # y of upper horizontal pair
_hy_lower = MAIN_Y - 0.22   # y of lower horizontal pair
_hx_mid   = (L(MAIN_X) + R(POOL_X)) / 2   # x midpoint between the two column centres
_vy_mid   = (B(MAIN_Y) + T(FROZEN_Y)) / 2 # y midpoint of right vertical

# Main → Pool  (upper horizontal, right-to-left)
arrow(ax, L(MAIN_X) - PAD, _hy_upper, R(POOL_X) + PAD, _hy_upper)
label(ax, _hx_mid, _hy_upper + LPAD, "inject previous versions", va="bottom")

# Pool → Main  (lower horizontal, left-to-right)
arrow(ax, R(POOL_X) + PAD, _hy_lower, L(MAIN_X) - PAD, _hy_lower)
label(ax, _hx_mid, _hy_lower - LPAD, "self-play", va="top")

# Main → Frozen  (right column, vertical, centered)
arrow(ax, MAIN_X, B(MAIN_Y) - PAD, FROZEN_X, T(FROZEN_Y) + PAD)
label(ax, MAIN_X + LPAD, _vy_mid, "freeze weights\nevery N steps", ha="left")

# Frozen → Exploiter  (bottom horizontal, centered, right-to-left)
arrow(ax, L(FROZEN_X) - PAD, FROZEN_Y, R(EXPL_X) + PAD, EXPL_Y)
label(ax, _hx_mid, FROZEN_Y + LPAD, "trains on frozen snapshot", va="bottom")

# Exploiter → Pool  (left column, vertical, centered, bottom-to-top)
_vy_left_mid = (T(EXPL_Y) + B(POOL_Y)) / 2
arrow(ax, EXPL_X, T(EXPL_Y) + PAD, POOL_X, B(POOL_Y) - PAD)
label(ax, EXPL_X - LPAD, _vy_left_mid, "inject exploiter\ninto pool", ha="right")

plt.tight_layout(pad=0.3)
plt.savefig("league_diagram.png", dpi=180, bbox_inches="tight",
            facecolor="white")
print("Saved league_diagram.png")
