"""Draw the diagrams of the Methods pages and of the noise tutorials.

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
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

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


def hsmpo_frame():
    """The circuit, then the HSMPO frame: Cliffords moved to the end."""
    fig, ax = canvas(10.6, 4.1)
    panel(ax, 0.1, 2.15, 10.5, 4.0, T["tint_y"], "Original circuit")
    xs = chain(
        ax,
        0.55,
        2.95,
        [
            ("none", r"$|0\rangle$", 0.62),
            (YELLOW, r"$C_0$", 0.62),
            (BLUE, r"$R_1$", 0.62),
            (YELLOW, r"$C_1$", 0.62),
            (RED, r"$\mathcal{E}$", 0.62),
            (BLUE, r"$R_2$", 0.62),
            (YELLOW, r"$C_2$", 0.62),
            (BLUE, r"$R_3$", 0.62),
            ("white", r"$O$", 0.62),
        ],
        step=1.55,
    )
    text(
        ax,
        xs[4],
        2.42,
        r"noise at location $\ell$",
        size=9.5,
        ha="center",
        color=T["soft"],
    )
    panel(
        ax,
        0.1,
        0.1,
        10.5,
        1.95,
        T["tint_b"],
        "HSMPO frame: every Clifford moved to the end",
    )
    xs2 = chain(
        ax,
        0.55,
        0.9,
        [
            ("none", r"$|0\rangle$", 0.62),
            (BLUE, r"$R'_1$", 0.62),
            (RED, r"$A'\,\cdot\,B'$", 1.05),
            (BLUE, r"$R'_2$", 0.62),
            (BLUE, r"$R'_3$", 0.62),
            ("white", r"$O' = C^\dagger O\, C$", 1.75),
        ],
        step=1.9,
    )
    text(
        ax,
        (xs2[1] + xs2[4]) / 2,
        0.32,
        r"an MPS sees only the dressed rotations $R'_j$; "
        r"$G'_j$, $A'$, $B'$, $O'$ come from the tableau",
        size=9.5,
        ha="center",
        color=T["soft"],
    )
    return fig


def sliding():
    """A Pauli fault slides to the observable, flipping the anticommuting angles."""
    fig, ax = canvas(8.8, 2.9)
    text(ax, 0.1, 2.25, "before", size=11, va="center", fontweight="bold")
    text(ax, 0.1, 0.75, "after", size=11, va="center", fontweight="bold")
    xb = chain(
        ax,
        1.4,
        2.25,
        [
            (YELLOW, r"$P'$", 0.62),
            (BLUE, r"$R'_2(+\theta_2)$", 1.55),
            (BLUE, r"$R'_3(+\theta_3)$", 1.55),
            ("white", r"$O'$", 0.62),
        ],
        step=1.7,
    )
    xa = chain(
        ax,
        1.4,
        0.75,
        [
            (BLUE, r"$R'_2(-\theta_2)$", 1.55),
            (BLUE, r"$R'_3(+\theta_3)$", 1.55),
            (YELLOW, r"$P'$", 0.62),
            ("white", r"$O'$", 0.62),
        ],
        step=1.7,
    )
    arrow(
        ax,
        xb[0] + 0.2,
        1.92,
        xa[2] - 0.15,
        1.08,
        color=T["soft"],
        rad=0.12,
        ls=(0, (4, 3)),
    )
    text(
        ax,
        xa[0],
        0.2,
        r"$\{P', G'_2\} = 0$: sign flipped",
        size=9.5,
        ha="center",
        color=T["soft"],
    )
    text(
        ax,
        xa[1],
        0.2,
        r"$[P', G'_3] = 0$: unchanged",
        size=9.5,
        ha="center",
        color=T["soft"],
    )
    return fig


def strata():
    """Configurations ordered by the number k of active noise locations."""
    fig, ax = canvas(9.4, 3.3)
    rows = [
        ("k = 0", [], "weight $q_1 q_2 \\cdots q_L$: the ideal circuit, exact"),
        ("k = 1", [(2, YELLOW), (5, RED)], "every single fault or coherence, exact"),
        (
            "k ≥ 2",
            [(1, YELLOW), (4, RED), (6, YELLOW)],
            "sampled: error $\\propto W_{\\geq 2} \\approx M^2/2$",
        ),
    ]
    for i, (label, active, note) in enumerate(rows):
        y = 2.75 - i * 0.95
        text(ax, 0.1, y, label, size=11, va="center", fontweight="bold")
        for j in range(8):
            fill = dict(active).get(j, T["empty"])
            ax.add_patch(
                Circle(
                    (1.25 + 0.5 * j, y),
                    0.17,
                    fc=fill,
                    ec=BOX_INK if fill != T["empty"] else T["soft"],
                    lw=1.0,
                    zorder=3,
                )
            )
        box(ax, 7.05, y, 4.5, 0.55, "none", note, size=9.5, edge=T["soft"])
    for x, fill, label in (
        (1.2, YELLOW, "Pauli fault ($A = B$)"),
        (4.0, RED, "coherence ($A \\neq B$)"),
        (6.9, T["empty"], "nothing happens"),
    ):
        ax.add_patch(
            Circle(
                (x, 0.25),
                0.13,
                fc=fill,
                ec=BOX_INK if fill != T["empty"] else T["soft"],
                zorder=3,
            )
        )
        text(ax, x + 0.25, 0.25, label, size=9.5, va="center")
    return fig


def two_roads():
    """Diagonal terms need one MPS, coherences two that separate at the location."""
    fig, ax = canvas(9.6, 3.9)
    panel(ax, 0.1, 2.35, 9.5, 3.8, T["tint_y"], "A = B: one road")
    box(ax, 1.0, 2.95, 1.1, 0.55, "white", r"$|\phi_{<\ell}\rangle$")
    wire(ax, 1.55, 6.1, 2.95)
    box(ax, 2.5, 2.95, 0.55, 0.55, YELLOW, r"$A'$")
    text(
        ax,
        4.2,
        3.12,
        "ket and bra flip the same signs",
        size=9.5,
        ha="center",
        color=T["soft"],
    )
    box(
        ax,
        7.6,
        2.95,
        2.9,
        0.6,
        YELLOW,
        r"$\langle\phi^{(A)}|\,O'\,|\phi^{(A)}\rangle$: 1 MPS",
        size=10.5,
    )
    panel(ax, 0.1, 0.1, 9.5, 2.15, T["tint_r"], "A ≠ B: two roads")
    box(ax, 1.0, 1.05, 1.1, 0.55, "white", r"$|\phi_{<\ell}\rangle$")
    ax.plot(
        [1.55, 2.1, 2.5, 5.7, 6.1],
        [1.05, 1.05, 1.6, 1.6, 1.05],
        color=T["ink"],
        lw=1.2,
        zorder=1,
    )
    ax.plot(
        [1.55, 2.1, 2.5, 5.7, 6.1],
        [1.05, 1.05, 0.5, 0.5, 1.05],
        color=T["ink"],
        lw=1.2,
        zorder=1,
    )
    box(ax, 3.0, 1.6, 0.55, 0.5, YELLOW, r"$A'$")
    box(ax, 3.0, 0.5, 0.55, 0.5, YELLOW, r"$B'$")
    text(
        ax,
        4.4,
        1.73,
        "ket: signs flipped by $A'$",
        size=9.5,
        ha="center",
        color=T["soft"],
    )
    text(
        ax,
        4.4,
        0.27,
        "bra: signs flipped by $B'$",
        size=9.5,
        ha="center",
        color=T["soft"],
    )
    box(
        ax,
        7.75,
        1.05,
        3.1,
        0.6,
        RED,
        r"$\langle\phi^{(B)}|\,B'O'A'\,|\phi^{(A)}\rangle$: 2 MPS",
        size=10.5,
    )
    return fig


def chi_amplitude_damping():
    """Process matrix of amplitude damping: faults on the diagonal, two coherences."""
    fig, ax = canvas(7.6, 3.6)
    labels = "IXYZ"
    cells = {
        (0, 0): ("$(1+s)^2/4$", YELLOW),
        (1, 1): ("$\\gamma/4$", YELLOW),
        (2, 2): ("$\\gamma/4$", YELLOW),
        (3, 3): ("$(1-s)^2/4$", YELLOW),
        (0, 3): ("$\\gamma/4$", RED),
        (3, 0): ("$\\gamma/4$", RED),
        (1, 2): ("$-i\\gamma/4$", RED),
        (2, 1): ("$+i\\gamma/4$", RED),
    }
    size, x0, y0 = 0.78, 0.65, 0.25
    for r in range(4):
        text(
            ax,
            x0 - 0.3,
            y0 + (3.5 - r) * size,
            f"${labels[r]}$",
            size=12,
            ha="center",
            va="center",
        )
        text(
            ax,
            x0 + (c := r) * size + size / 2,
            y0 + 4 * size + 0.15,
            f"${labels[r]}$",
            size=12,
            ha="center",
        )
        for c in range(4):
            label, fill = cells.get((r, c), ("0", T["empty"]))
            ax.add_patch(
                FancyBboxPatch(
                    (x0 + c * size + 0.03, y0 + (3 - r) * size + 0.03),
                    size - 0.06,
                    size - 0.06,
                    boxstyle="round,pad=0,rounding_size=0.05",
                    fc=fill,
                    ec=BOX_INK if fill != T["empty"] else T["soft"],
                    lw=0.9,
                    zorder=3,
                )
            )
            ax.text(
                x0 + c * size + size / 2,
                y0 + (3.5 - r) * size,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color=BOX_INK if fill != T["empty"] else T["soft"],
                zorder=4,
            )
    text(
        ax,
        x0 - 0.3,
        y0 + 4 * size + 0.15,
        "A \\ B",
        size=9,
        ha="center",
        color=T["soft"],
    )
    box(
        ax,
        5.6,
        2.35,
        3.4,
        0.85,
        YELLOW,
        "diagonal: Pauli twirl\nfaults, sign flips, 1 MPS",
        size=10,
    )
    box(
        ax,
        5.6,
        1.05,
        3.4,
        0.85,
        RED,
        "off-diagonal: coherences\ntransition amplitudes, 2 MPS",
        size=10,
    )
    return fig


def coherence_effects():
    """Rotation, tilt and translation of the Bloch sphere."""
    fig, ax = canvas(9.3, 3.3)
    centres = (1.55, 4.65, 7.75)
    for xc, title, sub in zip(
        centres,
        ("Rotation", "Tilt", "Translation"),
        (
            r"Im $\chi_{IP}$: unitary errors",
            r"Re $\chi_{PQ}$: anisotropic noise",
            r"$t_P = 4\,$Re $\chi_{IP}$: relaxation",
        ),
    ):
        ax.add_patch(
            Circle(
                (xc, 1.85),
                1.0,
                fc="none",
                ec=T["soft"],
                lw=1.0,
                ls=(0, (4, 3)),
                zorder=1,
            )
        )
        text(ax, xc, 0.45, title, size=11.5, ha="center", fontweight="bold")
        text(ax, xc, 0.1, sub, size=9.5, ha="center", color=T["soft"])
    xc = centres[0]
    ax.plot([xc, xc + 1.0], [1.85, 1.85], color=T["soft"], lw=1.0, ls=(0, (3, 3)))
    arrow(ax, xc, 1.85, xc + 0.72, 1.85 + 0.69, color=T["ink"])
    ax.add_patch(Circle((xc, 1.85), 0.06, fc=T["ink"], ec="none", zorder=4))
    arrow(ax, xc + 0.95, 2.05, xc + 0.62, 2.62, color=T["soft"], rad=0.25)
    ax.add_patch(
        Ellipse(
            (centres[1], 1.85),
            1.7,
            1.05,
            angle=35,
            fc=BLUE,
            ec=BOX_INK,
            lw=1.0,
            alpha=0.9,
            zorder=2,
        )
    )
    ax.add_patch(
        Circle((centres[2], 2.15), 0.7, fc=RED, ec=BOX_INK, lw=1.0, alpha=0.9, zorder=2)
    )
    arrow(ax, centres[2], 1.85, centres[2], 2.35, color=BOX_INK)
    ax.add_patch(Circle((centres[2], 1.85), 0.05, fc=BOX_INK, ec="none", zorder=4))
    return fig


def adjoint():
    """Forward pass with checkpoints, then the costate undoing the rotations."""
    fig, ax = canvas(10.6, 4.4)
    panel(
        ax,
        0.1,
        2.55,
        10.5,
        4.3,
        T["tint_b"],
        "forward: apply the rotations, keep every state",
    )
    panel(ax, 0.1, 0.1, 10.5, 1.85, T["tint_r"], "backward: undo them on the costate")
    xs = [1.0, 3.2, 5.4, 7.6]
    for j, x in enumerate(xs):
        box(ax, x, 3.25, 0.85, 0.55, "white", f"$\\psi_{j}$")
        if not j:
            continue
        box(ax, x, 1.0, 0.85, 0.55, RED if j == 3 else "white", f"$\\lambda_{j}$")
        box(ax, x - 1.1, 3.25, 0.75, 0.55, BLUE, f"$R_{j}$")
        arrow(ax, x - 2.2 + 0.43, 3.25, x - 1.1 - 0.39, 3.25)
        arrow(ax, x - 1.1 + 0.39, 3.25, x - 0.43, 3.25)
        if j < 3:
            box(ax, x + 1.1, 1.0, 0.75, 0.55, BLUE, f"$R_{j + 1}^\\dagger$")
            arrow(ax, x + 1.1 - 0.39, 1.0, x + 0.43, 1.0)
            arrow(ax, x + 2.2 - 0.43, 1.0, x + 1.1 + 0.39, 1.0)
        ax.plot([x, x], [1.28, 2.97], color=T["soft"], lw=1.0, ls=(0, (2, 2)), zorder=1)
        box(
            ax,
            x,
            2.2,
            1.15,
            0.42,
            YELLOW,
            f"$\\partial f/\\partial\\varphi_{j}$",
            size=9.5,
        )
    box(ax, 9.45, 2.2, 1.6, 0.62, RED, r"$\lambda = O\,|\psi\rangle$", size=10.5)
    arrow(ax, 8.03, 3.25, 9.3, 2.53, rad=-0.25)
    arrow(ax, 9.3, 1.87, 8.03, 1.05, rad=-0.25)
    text(
        ax,
        5.3,
        0.3,
        r"at each rotation: $\partial f / \partial \varphi_j = $ Im$\,\langle\lambda_j|\,G_j\,|\psi_j\rangle\, /\, \langle\psi|\psi\rangle$",
        size=10,
        ha="center",
        color=T["soft"],
    )
    return fig


def sign_flip_gradient():
    """The derivative as a Pauli fault G_j inserted after rotation j."""
    fig, ax = canvas(9.4, 1.6)
    xs = chain(
        ax,
        0.6,
        0.85,
        [
            ("none", r"$|0\rangle$", 0.62),
            (BLUE, r"$R_1$", 0.62),
            (BLUE, r"$R_j$", 0.62),
            (YELLOW, r"$-\frac{i}{2}G_j$", 1.05),
            (BLUE, r"$R_{j+1}$", 0.8),
            (BLUE, r"$R_R$", 0.62),
            ("white", r"$O$", 0.62),
        ],
        step=1.6,
    )
    text(
        ax,
        xs[3],
        0.2,
        r"$\partial_{\varphi_j} R_j = -\frac{i}{2}\, G_j R_j$: a Pauli inserted after $R_j$",
        size=9.5,
        ha="center",
        color=T["soft"],
    )
    return fig


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
    "hsmpo_frame": hsmpo_frame,
    "sliding_lemma": sliding,
    "noise_strata": strata,
    "one_road_two_roads": two_roads,
    "chi_amplitude_damping": chi_amplitude_damping,
    "coherence_effects": coherence_effects,
    "adjoint_passes": adjoint,
    "derivative_as_fault": sign_flip_gradient,
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
