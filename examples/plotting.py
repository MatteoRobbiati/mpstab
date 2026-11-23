"""The plot style shared by the mpstab tutorials, notebooks and scripts.

Call :func:`set_style` once, before plotting, and pick colors from
:data:`COLORS`::

    import matplotlib.pyplot as plt
    from plotting import COLORS, set_style

    set_style()
    plt.plot(x, y, "o-", color=COLORS["blue"], label="mpstab")

Colors are used in the order of :data:`CYCLE`: blue, red, orange, black.
Markers are large with a black edge, fonts are large enough to read in a
browser and on a slide.

Requires matplotlib, which is not a dependency of mpstab itself.
"""

import matplotlib.pyplot as plt
from cycler import cycler

#: The four series colors, plus the gray for reference lines and the grid.
COLORS = {
    "blue": "#2a78d6",
    "red": "#d63a3a",
    "orange": "#eb8a1c",
    "black": "#0b0b0b",
    "gray": "#52514e",
    "grid": "#e4e3df",
}

#: Series colors, in the order they are assigned.
CYCLE = [COLORS["blue"], COLORS["red"], COLORS["orange"], COLORS["black"]]

#: Matplotlib settings of the mpstab style.
STYLE = {
    # text
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 17,
    "text.color": COLORS["black"],
    "axes.labelcolor": COLORS["black"],
    "xtick.color": COLORS["gray"],
    "ytick.color": COLORS["gray"],
    # axes
    "axes.prop_cycle": None,  # filled in by set_style
    "axes.edgecolor": COLORS["gray"],
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.8,
    "axes.titlepad": 10,
    # marks
    "lines.linewidth": 2.2,
    "lines.markersize": 10,
    "lines.markeredgecolor": COLORS["black"],
    "lines.markeredgewidth": 1.1,
    "scatter.edgecolors": COLORS["black"],
    "errorbar.capsize": 3,
    "patch.edgecolor": COLORS["black"],
    "patch.linewidth": 1.0,
    # legend and figure
    "legend.frameon": False,
    "figure.figsize": (8, 5),
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
}


def set_style():
    """Apply the mpstab plot style to matplotlib, for every following figure."""
    style = dict(STYLE)
    style["axes.prop_cycle"] = cycler(color=CYCLE)
    plt.rcParams.update(style)
