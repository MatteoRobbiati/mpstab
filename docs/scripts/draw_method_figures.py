"""Draw the diagrams of the HQSMPO page and of the error-mitigation tutorial.

The palette is the one of the pipeline figure on the landing page: pastel
yellow for Clifford gates and Pauli faults, pastel blue for magic (non-Clifford)
rotations, pastel red for noise beyond Pauli and for the quantities a method
propagates backwards. Text inside a coloured box is always dark.

Run from the repository root::

    python docs/scripts/draw_method_figures.py

Every figure is written to ``docs/_static/figures/`` three times:

- ``name.svg`` on white, for the light theme of the Methods pages;
- ``name_dark.svg`` on a transparent background with light strokes, for the
  dark theme;
- ``name_neutral.svg`` on a transparent background with mid-grey strokes,
  readable on both themes, which the tutorial notebooks embed as attachments
  (a notebook cannot switch images with the page theme).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "_static" / "figures"

YELLOW = "#FDFFAE"  # Clifford gates, Pauli faults
BLUE = "#AEC0FF"  # magic rotations
RED = "#FFAEB0"  # non-Pauli noise, coherences, costate
BOX_INK = "#211715"  # text and edges of the coloured boxes, in every theme

THEMES = {
    "light": dict(
        ink="#211715",
        soft="#6b6b6b",
        bg="white",
        empty="white",
        tint_y="#FFFEF2",
        tint_b="#F2F5FF",
        tint_r="#FFF3F2",
        suffix="",
    ),
    "dark": dict(
        ink="#E6E6E6",
        soft="#A8A8A8",
        bg="none",
        empty="#3A3A3D",
        tint_y="#2B2A1F",
        tint_b="#20253A",
        tint_r="#36242A",
        suffix="_dark",
    ),
    "neutral": dict(
        ink="#7A7A7A",
        soft="#8C8C8C",
        bg="none",
        empty="#FFFFFF",
        tint_y="none",
        tint_b="none",
        tint_r="none",
        suffix="_neutral",
    ),
}
T = dict(THEMES["light"])
FONT = {"mathtext.fontset": "dejavusans", "font.family": "DejaVu Sans"}


# --------------------------------------------------------------------- helpers


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def box(ax, x, y, w, h, fill, text="", size=11, edge=None, round_=0.08, lw=1.0, ls="-"):
    """A rounded box centred at ``(x, y)``."""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={round_}",
            fc=fill,
            ec=edge or (BOX_INK if fill not in ("none",) else T["ink"]),
            lw=lw,
            ls=ls,
            zorder=3,
        )
    )
    if text:
        color = BOX_INK if fill not in ("none",) else T["ink"]
        ax.text(
            x, y, text, ha="center", va="center", fontsize=size, color=color, zorder=4
        )


def panel(ax, x0, y0, x1, y1, tint, title=None):
    """A tinted background panel with an optional title at its top left."""
    if tint != "none":
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle="round,pad=0,rounding_size=0.12",
                fc=tint,
                ec="none",
                zorder=0,
            )
        )
    else:
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle="round,pad=0,rounding_size=0.12",
                fc="none",
                ec=T["soft"],
                lw=0.8,
                ls=(0, (3, 3)),
                zorder=0,
            )
        )
    if title:
        ax.text(
            x0 + 0.15,
            y1 - 0.12,
            title,
            ha="left",
            va="top",
            fontsize=11,
            color=T["ink"],
            fontweight="bold",
            zorder=4,
        )


def text(ax, x, y, s, size=10.5, color=None, **kw):
    ax.text(x, y, s, fontsize=size, color=color or T["ink"], zorder=4, **kw)


def wire(ax, x0, x1, y, color=None, lw=1.2):
    ax.plot([x0, x1], [y, y], color=color or T["ink"], lw=lw, zorder=1)


def arrow(ax, x0, y0, x1, y1, color=None, rad=0.0, ls="-", lw=1.3, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            mutation_scale=13,
            connectionstyle=f"arc3,rad={rad}",
            color=color or T["ink"],
            lw=lw,
            ls=ls,
            zorder=2,
        )
    )


def save(fig, name):
    fig.savefig(
        OUT / f"{name}{T['suffix']}.svg",
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor=T["bg"],
        transparent=T["bg"] == "none",
    )
    plt.close(fig)


def chain(ax, x0, y, items, step=1.15, h=0.62):
    """Boxes on a wire: ``items`` are ``(fill, text, width)``; returns the centres."""
    xs, x = [], x0
    for fill, label, w in items:
        xs.append(x + w / 2)
        x += w + (step - 1.0) * 0.6 + 0.12
    wire(ax, x0 - 0.2, x - 0.05, y)
    for (fill, label, w), xc in zip(items, xs):
        box(ax, xc, y, w, h, fill, label, size=11)
    return xs


# --------------------------------------------------------------------- figures


def hqsmpo_split():
    """Head on the device, tail folded classically into the observable."""
    fig, ax = canvas(10.6, 4.4)
    xs = chain(
        ax,
        0.6,
        3.6,
        [("none", r"$|0\rangle$", 0.62)]
        + [(BLUE, f"$R'_{j}$", 0.62) for j in range(1, 7)]
        + [(YELLOW, r"$C$", 0.62), ("white", r"$O$", 0.62)],
        step=1.45,
    )
    cut = (xs[3] + xs[4]) / 2
    ax.plot([cut, cut], [3.05, 4.2], color=T["ink"], lw=1.6, ls=(0, (5, 3)), zorder=5)
    text(ax, cut, 4.28, "cut_index", size=10, ha="center", family="monospace")
    text(ax, (xs[1] + xs[3]) / 2, 3.05, "head", size=11, ha="center", fontweight="bold")
    text(ax, (xs[4] + xs[8]) / 2, 3.05, "tail", size=11, ha="center", fontweight="bold")
    panel(ax, 0.1, 0.1, 5.1, 2.55, T["tint_b"], "quantum: run the head")
    box(
        ax,
        2.6,
        1.6,
        4.4,
        0.62,
        BLUE,
        "synthesis: native gates + Clifford residual",
        size=10,
    )
    box(
        ax,
        2.6,
        0.7,
        4.4,
        0.62,
        "white",
        "run on the device with a shot budget",
        size=10,
    )
    arrow(ax, 2.6, 1.28, 2.6, 1.02)
    panel(ax, 5.4, 0.1, 10.5, 2.55, T["tint_r"], "classical: fold the tail")
    box(
        ax,
        7.95,
        1.6,
        4.6,
        0.62,
        RED,
        r"$O_{\rm tail} = T^\dagger C^\dagger O\, C\, T$ as an MPO",
        size=10,
    )
    box(
        ax,
        7.95,
        0.7,
        4.6,
        0.62,
        "white",
        "Pauli sampling or classical shadows",
        size=10,
    )
    arrow(ax, 7.95, 1.28, 7.95, 1.02)
    arrow(ax, (xs[1] + xs[3]) / 2, 2.85, 2.6, 1.95)
    arrow(ax, (xs[4] + xs[8]) / 2, 2.85, 7.95, 1.95)
    return fig


def data_driven():
    """Learn the noise on near-Clifford circuits that mpstab simulates exactly."""
    fig, ax = canvas(10.4, 3.9)
    panel(ax, 0.1, 0.1, 3.9, 3.8, T["tint_y"], "training circuits")
    for i, magic in enumerate(([1, 4], [2], [0, 5])):
        y = 3.0 - i * 0.95
        wire(ax, 0.4, 3.7, y)
        for j in range(6):
            box(
                ax,
                0.75 + 0.55 * j,
                y,
                0.4,
                0.4,
                BLUE if j in magic else YELLOW,
                "",
                round_=0.05,
            )
    text(
        ax,
        2.0,
        0.28,
        "the target, most magic gates → Clifford",
        size=9.5,
        ha="center",
        color=T["soft"],
    )
    box(
        ax,
        5.6,
        2.85,
        2.6,
        0.75,
        BLUE,
        "mpstab: noiseless value\ncheap, little magic",
        size=10,
    )
    box(ax, 5.6, 1.35, 2.6, 0.75, RED, "device or noise model:\nnoisy value", size=10)
    arrow(ax, 3.95, 2.6, 4.3, 2.8)
    arrow(ax, 3.95, 1.8, 4.3, 1.45)
    panel(ax, 7.2, 0.1, 10.3, 3.8, T["tint_b"], "fit, then correct")
    rng = np.random.default_rng(3)
    xs = rng.uniform(0.15, 0.85, 9)
    ys = 0.05 + 1.1 * xs + rng.normal(0, 0.04, 9)
    ox, oy, s = 7.6, 0.55, 2.4
    ax.plot([ox, ox + s], [oy, oy], color=T["soft"], lw=1.0)
    ax.plot([ox, ox], [oy, oy + s], color=T["soft"], lw=1.0)
    ax.plot(
        [ox, ox + s],
        [oy + 0.05 * s, oy + (0.05 + 1.1) * s * 0.92],
        color=T["ink"],
        lw=1.2,
        zorder=2,
    )
    for x, y in zip(xs, ys):
        ax.add_patch(
            Circle(
                (ox + s * x, oy + s * y * 0.92),
                0.07,
                fc=YELLOW,
                ec=BOX_INK,
                lw=0.8,
                zorder=3,
            )
        )
    text(ax, ox + s / 2, oy - 0.3, "noisy", size=9.5, ha="center", color=T["soft"])
    text(
        ax,
        ox - 0.12,
        oy + s / 2,
        "exact",
        size=9.5,
        ha="center",
        va="center",
        rotation=90,
        color=T["soft"],
    )
    arrow(ax, 6.9, 2.85, 7.5, 2.4)
    arrow(ax, 6.9, 1.35, 7.5, 1.6)
    return fig


FIGURES = {
    "hqsmpo_split": hqsmpo_split,
    "data_driven_mitigation": data_driven,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(FONT):
        for theme, values in THEMES.items():
            T.clear()
            T.update(values)
            for name, draw in FIGURES.items():
                save(draw(), name)
    print(f"{len(FIGURES)} figures x {len(THEMES)} themes in {OUT}")


if __name__ == "__main__":
    main()
