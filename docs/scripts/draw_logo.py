"""Draw the mpstab logo: one open MPS chain folded into a dagger.

The chain runs pommel, grip, the crossguard from its lower to its upper end,
the upper edge of the blade, the tip, and back along the lower edge, where it
ends. Black (white on dark) bonds and outlines, tensors in the light blue of
the original logo mixed, irregularly, with the red and yellow of the landing
page figure, and the name in heavy capitals. Run from the repository root::

    python docs/scripts/draw_logo.py

It writes ``docs/_static/logo/mpstab_logo_{light,dark}.{svg,png}``: black on a
light background, white on a dark one. The name is set in Avenir Next
Bold when the system has it (macOS), DejaVu Sans Bold otherwise; the SVGs
store it as paths, so they need no font to display.
"""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle

OUT = Path(__file__).resolve().parents[1] / "_static" / "logo"
AVENIR = Path("/System/Library/Fonts/Avenir Next.ttc")

BLUE = "#A7C7FF"  # the tensors of the original logo
RED = "#FFA7A7"  # the stabilizer region of docs/stab_ent.svg
YELLOW = "#FFE38A"  # its statevector region, a little deeper
# One color per tensor, in chain order: mostly blue, with red and yellow
# scattered by hand rather than in a pattern.
COLORS = [
    BLUE,
    YELLOW,
    BLUE,
    RED,  # pommel, grip, crossguard
    BLUE,
    BLUE,
    YELLOW,
    BLUE,
    RED,  # upper edge of the blade
    BLUE,  # tip
    YELLOW,
    BLUE,
    BLUE,
    RED,
    YELLOW,  # lower edge, back to the guard
]
THEMES = {"light": "#000000", "dark": "#FFFFFF"}  # bonds, outlines and name

RADIUS = 0.29
LINE = 3.2  # points


def name_font() -> FontProperties:
    """Avenir Next Bold extracted from the system collection, if any."""
    if AVENIR.exists():
        from fontTools.ttLib import TTCollection

        # Extracted to a temporary folder: the font is not ours to ship.
        target = Path(tempfile.gettempdir()) / "mpstab_avenir_next_bold.ttf"
        if not target.exists():
            for font in TTCollection(str(AVENIR)).fonts:
                if font["name"].getDebugName(4) == "Avenir Next Bold":
                    font.save(str(target))
                    break
        if target.exists():
            return FontProperties(fname=str(target))
    return FontProperties(family="DejaVu Sans", weight="bold")


def chain():
    """The tensors of the folded chain, in order."""
    half, guard, spacing = 0.62, 1.32, 0.84
    pommel, grip = (0.0, 0.0), (1.15, 0.0)
    guard_low, guard_high = (1.85, -guard), (1.85, guard)
    upper = [(2.55 + spacing * i, half) for i in range(5)]
    tip = (upper[-1][0] + 0.9, 0.0)
    lower = [(x, -half) for x, _ in reversed(upper)]
    return [pommel, grip, guard_low, guard_high, *upper, tip, *lower]


def draw(ink: str, font: FontProperties):
    nodes = np.array(chain())
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(
        nodes[:, 0],
        nodes[:, 1],
        color=ink,
        lw=LINE,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=1,
    )
    for (x, y), color in zip(nodes, COLORS, strict=True):
        ax.add_patch(Circle((x, y), RADIUS, fc=color, ec=ink, lw=LINE, zorder=2))
    # The name sits above the blade, between the crossguard and the tip.
    left, right = nodes[4][0] - RADIUS, nodes[9][0] + RADIUS
    ax.text(
        (left + right) / 2,
        1.33,
        "MPSTAB",
        fontproperties=font,
        fontsize=40,
        color=ink,
        ha="center",
        va="center",
    )
    ax.set_xlim(-RADIUS - 0.1, right + 0.1)
    ax.set_ylim(-1.32 - RADIUS - 0.1, 1.32 + RADIUS + 0.1)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = name_font()
    for theme, ink in THEMES.items():
        fig = draw(ink, font)
        for ext in ("svg", "png"):
            fig.savefig(
                OUT / f"mpstab_logo_{theme}.{ext}",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.04,
                dpi=300 if ext == "png" else None,
            )
        plt.close(fig)
    print(f"logo written to {OUT}")


if __name__ == "__main__":
    main()
