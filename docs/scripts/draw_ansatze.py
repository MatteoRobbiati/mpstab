"""Draw the circuit snippets shown in ``docs/guides/ansatze_catalogue.rst``.

Each figure is a small instance of one ansatz (a few qubits, one step or layer)
drawn from the circuit the class actually builds. Clifford gates are yellow
and magic (non-Clifford) rotations blue; CNOT, CZ and SWAP are drawn with their
usual symbols, and brackets above the wires mark the parts of the circuit (kicks, couplings,
steps, ...). Angles that the class takes from a parameter are labelled with the
parameter's symbol.

Run from the repository root::

    python docs/scripts/draw_ansatze.py

The figures are written to ``docs/_static/ansatze/``, each as ``name.svg``
(light theme) and ``name_dark.svg`` (dark theme).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Patch

from mpstab.models import ansatze as az

OUT = Path(__file__).resolve().parents[1] / "_static" / "ansatze"

# Pastel palette.
YELLOW = "#FBE7A1"  # Clifford gate
BLUE = "#A9C5F2"  # magic gate
GATE_INK = "#2B2B2B"  # text inside the coloured boxes, in both themes

# Colours of everything drawn on the page background. Every figure is saved
# twice: ``name.svg`` on white for the light theme and ``name_dark.svg`` on a
# transparent background for the dark theme, where the dark strokes become the
# grey of the mpstab logo, with lines ``lw`` times thicker so they stay visible;
# the text next to the wires (``text``) is a lighter grey, as thin glyphs need more contrast.
LOGO_GREY = "#737373"
THEMES = {
    "light": {"edge": "#3A3A3A", "wire": "#6B6B6B", "ink": "#2B2B2B", "text": "#2B2B2B",
              "oplus": "white", "bg": "white", "suffix": "", "lw": 1.0},
    "dark": {"edge": LOGO_GREY, "wire": LOGO_GREY, "ink": LOGO_GREY, "text": "#9A9A9A",
             "oplus": "none", "bg": "none", "suffix": "_dark", "lw": 1.6},
}
C = dict(THEMES["light"])  # current theme, set by ``save_themed``

COL_W = 0.95  # column width
BOX = 0.72  # side of a single-qubit box
ROUND = 0.14
NAMES = {"rx": "R_X", "ry": "R_Y", "rz": "R_Z", "rzz": "R_{ZZ}", "cu1": "P"}
PLAIN = {"h": "H", "x": "X", "z": "Z", "s": "S", "t": "T", "sdg": "S^\\dagger"}
# Gate text: upright sans-serif (DejaVu Sans), set in the math text of each label.
FONT = {"mathtext.fontset": "dejavusans", "mathtext.default": "regular"}


def fmt_angle(theta: float, symbols: dict) -> str:
    """The label of an angle: a parameter symbol, a multiple of pi/4, or a number."""
    for value, sym in symbols.items():
        if np.isclose(theta, value):
            return sym
        if np.isclose(theta, -value):
            return "-" + sym
    quarters = theta / (np.pi / 4)
    if np.isclose(quarters, round(quarters)):
        q = int(round(quarters))
        if q == 0:
            return "0"
        sign = "-" if q < 0 else ""
        q = abs(q)
        num, den = (q // 4, 1) if q % 4 == 0 else ((q // 2, 2) if q % 2 == 0 else (q, 4))
        num_s = "" if num == 1 else str(num)
        return f"{sign}{num_s}\\pi" + ("" if den == 1 else f"/{den}")
    return f"{theta:.2f}"


def layout(queue, nqubits, breaks=()):
    """Column of each gate, scheduled as early as possible; ``breaks`` are gate
    indices where every qubit is synchronised first."""
    level = [0] * nqubits
    cols = []
    for i, g in enumerate(queue):
        if i in breaks:
            top = max(level)
            level = [top] * nqubits
        lo, hi = min(g.qubits), max(g.qubits)
        c = max(level[lo : hi + 1])
        for q in range(lo, hi + 1):
            level[q] = c + 1
        cols.append(c)
    return cols, max(level)


def _rounded(ax, x, y_top, y_bottom, fill):
    """A rounded box of width ``BOX`` from ``y_top`` down to ``y_bottom``."""
    ax.add_patch(
        FancyBboxPatch(
            (x - BOX / 2, y_bottom),
            BOX,
            y_top - y_bottom,
            boxstyle=f"round,pad=0.0,rounding_size={ROUND}",
            fc=fill,
            ec=C["edge"],
            lw=1.0 * C["lw"],
            zorder=3,
        )
    )


def _text(ax, x, y, text, size=8.5):
    """Text inside a gate box."""
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=GATE_INK, zorder=5)


def _dot(ax, x, y):
    ax.add_patch(Circle((x, y), 0.08, fc=C["ink"], ec=C["ink"], zorder=5))


def _oplus(ax, x, y):
    r = 0.19
    ax.add_patch(Circle((x, y), r, fc=C["oplus"], ec=C["ink"], lw=1.1 * C["lw"], zorder=5))
    ax.plot([x - r, x + r], [y, y], color=C["ink"], lw=1.1 * C["lw"], zorder=6)
    ax.plot([x, x], [y - r, y + r], color=C["ink"], lw=1.1 * C["lw"], zorder=6)


def _cross(ax, x, y):
    d = 0.12
    ax.plot([x - d, x + d], [y - d, y + d], color=C["ink"], lw=1.4 * C["lw"], zorder=5)
    ax.plot([x - d, x + d], [y + d, y - d], color=C["ink"], lw=1.4 * C["lw"], zorder=5)


def _label(name, theta, symbols):
    name = NAMES.get(name, name.upper())
    return f"${name}$\n${fmt_angle(theta, symbols)}$"


def _fill(g):
    return YELLOW if getattr(g, "clifford", False) else BLUE


def draw_gate(ax, g, x, ys, symbols, label=None):
    name = g.name
    theta = g.parameters[0] if g.parameters else None
    qubits = sorted(g.qubits)

    if len(qubits) == 1:
        y = ys[qubits[0]]
        _rounded(ax, x, y + BOX / 2, y - BOX / 2, _fill(g))
        if label is None:
            label = (_label(name, theta, symbols) if theta is not None
                     else f"${PLAIN.get(name, name.upper())}$")
        _text(ax, x, y, label, size=8 if "\n" in label else 10)
        return

    if name == "rzz":
        _rounded(ax, x, ys[qubits[0]] + BOX / 2, ys[qubits[-1]] - BOX / 2, _fill(g))
        _text(ax, x, (ys[qubits[0]] + ys[qubits[-1]]) / 2,
              label or _label(name, theta, symbols), size=8)
        return

    # Controlled gates and SWAP: the usual symbols joined by a vertical line.
    ax.plot([x, x], [ys[qubits[0]], ys[qubits[-1]]], color=C["ink"], lw=1.1 * C["lw"], zorder=2)
    if name in ("cx", "ccx"):
        for c in g.control_qubits:
            _dot(ax, x, ys[c])
        _oplus(ax, x, ys[g.target_qubits[0]])
    elif name == "cz":
        for q in qubits:
            _dot(ax, x, ys[q])
    elif name == "swap":
        for q in qubits:
            _cross(ax, x, ys[q])
    elif name == "cu1":
        _dot(ax, x, ys[g.control_qubits[0]])
        y = ys[g.target_qubits[0]]
        _rounded(ax, x, y + BOX / 2, y - BOX / 2, _fill(g))
        _text(ax, x, y, label or _label(name, theta, symbols), size=8)
    else:
        raise ValueError(f"no drawing for gate {name}")


def save_themed(plot, path, pad):
    """Run ``plot()`` once per theme and save ``path`` and its ``_dark`` twin."""
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES.values():
        C.update(theme)
        fig = plot()
        name = Path(path).stem + theme["suffix"] + Path(path).suffix
        fig.savefig(OUT / name, bbox_inches="tight", facecolor=theme["bg"],
                    transparent=theme["bg"] == "none", pad_inches=pad,
                    metadata={"Date": None})
        plt.close(fig)
    C.update(THEMES["light"])


def draw(queue, nqubits, path, **kwargs):
    """Draw ``queue`` on ``nqubits`` wires, in both themes (see ``_draw``)."""
    save_themed(lambda: _draw(queue, nqubits, **kwargs), path, pad=0.06)


def _draw(queue, nqubits, symbols=None, regions=(), labels=None, qubit_names=None):
    """Draw ``queue`` on ``nqubits`` wires and return the figure.

    Args:
        symbols: angle value -> LaTeX symbol (its negative gets a minus sign).
        regions: ``(first_gate, last_gate, text)`` triples, marked by a bracket
            above the wires; each region starts a new set of columns.
        labels: gate index -> box text, to override the automatic one.
        qubit_names: wire labels (default ``q_0, q_1, ...``).
    """
    symbols = symbols or {}
    labels = labels or {}
    breaks = {r[0] for r in regions}
    cols, ncols = layout(queue, nqubits, breaks)
    ys = [-q for q in range(nqubits)]
    top_pad = 0.9 if regions else 0.45
    width = ncols * COL_W
    fig, ax = plt.subplots(figsize=(0.62 * width + 1.0, 0.62 * nqubits + 0.55 * top_pad + 0.3))
    ax.set_xlim(-0.9, width + 0.15)
    ax.set_ylim(-nqubits + 0.4, 0.45 + top_pad)
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [c * COL_W + COL_W / 2 for c in cols]
    for first, last, text in regions:
        x0 = min(xs[first : last + 1]) - BOX / 2
        x1 = max(xs[first : last + 1]) + BOX / 2
        yb = 0.56
        ax.plot([x0, x0, x1, x1], [yb - 0.08, yb, yb, yb - 0.08], color=C["wire"], lw=0.9 * C["lw"])
        ax.text((x0 + x1) / 2, yb + 0.22, text, ha="center", va="center",
                fontsize=8.5, color=C["text"])

    for q, y in enumerate(ys):
        ax.plot([-0.2, width + 0.1], [y, y], color=C["wire"], lw=1.0 * C["lw"], zorder=1)
        name = qubit_names[q] if qubit_names else f"q_{{{q}}}"
        ax.text(-0.3, y, f"${name}$", ha="right", va="center", fontsize=9.5, color=C["text"])

    for i, (g, x) in enumerate(zip(queue, xs)):
        draw_gate(ax, g, x, ys, symbols, labels.get(i))
    return fig


# --------------------------------------------------------------------------- #
# Regions of the kicked Ising circuits, found from the gate sequence.
# --------------------------------------------------------------------------- #


def kicked_ising_regions(queue, coupling_name, step_labels=False):
    """Bands for kicks and couplings: a coupling is ``CNOT ... CNOT`` on a bond."""
    kinds, open_bond = [], None
    for g in queue:
        if g.name == "cx":
            bond = tuple(g.qubits)
            kinds.append(("c", coupling_name(bond)))
            open_bond = None if open_bond == bond else bond
        else:
            kinds.append(("c", kinds[-1][1]) if open_bond else ("k", "kicks"))
    regions, step, start = [], 0, 0
    for i in range(1, len(kinds) + 1):
        if i == len(kinds) or kinds[i][1] != kinds[start][1]:
            text = kinds[start][1]
            if kinds[start][0] == "k":
                step += 1
                if step_labels:
                    text = f"kicks, step {step}"
            regions.append((start, i - 1, text))
            start = i
    return regions


def chain_bonds(bond):
    return "even bonds" if min(bond) % 2 == 0 else "odd bonds"


# --------------------------------------------------------------------------- #
# The figures.
# --------------------------------------------------------------------------- #


def variational():
    np.random.seed(0)
    a = az.HardwareEfficient(nqubits=4, nlayers=1)
    q = a.circuit.queue
    labels = {i: "$R_Y$\n$\\theta$" for i, g in enumerate(q) if g.name == "ry"}
    draw(q, 4, "hardware_efficient.svg", labels=labels,
         regions=[(0, 3, "random $R_Y$"), (4, 7, "CZ ring")])

    a = az.HardwareEfficientBrickwork(nqubits=4, nlayers=1)
    q = a.circuit.queue
    labels = {i: f"${'R_Y' if g.name == 'ry' else 'R_Z'}$\n$\\theta$"
              for i, g in enumerate(q) if g.name in ("ry", "rz")}
    draw(q, 4, "hardware_efficient_brickwork.svg", labels=labels,
         regions=[(0, 7, "rotations"), (8, 9, "even CZ"), (10, 17, "rotations"),
                  (18, 19, "odd CZ + ring"), (20, 23, "final $R_Y$")])

    a = az.HammingWeightPreserving(nqubits=4, nlayers=1)
    q = a.circuit.queue[:14]  # half filling, then the first two RBS gates
    labels, k = {}, 0
    for i, g in enumerate(q):
        if g.name == "ry":
            sign = "" if k % 2 == 0 else "-"
            labels[i] = f"$R_Y$\n${sign}\\theta_{k // 2 + 1}$"
            k += 1
    draw(q, 4, "hamming_weight_preserving.svg", labels=labels,
         regions=[(0, 1, "half filling"), (2, 7, "RBS$(\\theta_1)$ on (0,1)"),
                  (8, 13, "RBS$(\\theta_2)$ on (2,3)")])


def floquet():
    b, theta = 0.4 * np.pi, 0.3
    a = az.FloquetAnsatz(nqubits=4, nlayers=1, b=b, theta=theta, target_qubit=1)
    q = a.circuit.queue
    half = len(a.half_sandwich.queue)
    draw(q, 4, "floquet_ansatz.svg", symbols={b: "b", theta: "\\theta"},
         regions=[(0, 0, "$H$"), (1, half, "$F$"), (half + 1, half + 1, "signal"),
                  (half + 2, len(q) - 1, "$F^\\dagger$")])


def kicked():
    a = az.KickedIsing(nqubits=4, n_steps=1, theta_h=0.3, theta_z=0.2, theta_j=0.45)
    q = a.circuit.queue
    draw(q, 4, "kicked_ising.svg",
         symbols={0.3: "\\theta_h", 0.2: "\\theta_z", 0.45: "\\theta_J"},
         regions=kicked_ising_regions(q, chain_bonds))

    a = az.ParametricKickedIsing(nqubits=4, n_steps=1, delta_h=0.1, delta_z=0.07, delta_j=0.05)
    q = a.circuit.queue
    draw(q, 4, "parametric_kicked_ising.svg",
         symbols={0.1: "\\delta_h", 0.07: "\\delta_z", 0.05: "\\delta_J"},
         regions=kicked_ising_regions(q, chain_bonds))

    # Two steps, ramp 0.4 -> 0.1, one coupling of step 2 dropped (bond_density).
    a = az.RampedKickedIsing(nqubits=4, n_steps=2, delta_h=0.4, delta_h_last=0.1,
                             bond_density=0.8, seed=RAMPED_SEED)
    q = a.circuit.queue
    draw(q, 4, "ramped_kicked_ising.svg",
         symbols={0.4: "\\delta_h^{(1)}", 0.1: "\\delta_h^{(2)}"},
         regions=kicked_ising_regions(q, lambda b: "couplings", step_labels=True))

    a = az.StarKickedIsing(nqubits=5, n_steps=1, delta_h=0.1)
    q = a.circuit.queue
    draw(q, 5, "star_kicked_ising.svg", symbols={0.1: "\\delta_h"},
         regions=kicked_ising_regions(q, lambda b: "hub-leaf couplings"),
         qubit_names=["\\mathrm{hub}"] + [f"q_{{{k}}}" for k in range(1, 5)])

    a = az.TrotterIsing(nqubits=4, n_steps=1, dt=0.2, J=1.0, h=0.7, transpile=False)
    q = a.circuit.queue
    n_zz = sum(g.name == "rzz" for g in q)
    draw(q, 4, "trotter_ising.svg", symbols={-0.4: "-2J\\,dt", -0.28: "-2h\\,dt"},
         regions=[(0, n_zz - 1, "$R_{ZZ}$ on every bond"), (n_zz, len(q) - 1, "$R_X$")])


def library():
    q = az.QFT(nqubits=3, transpile=False).circuit.queue
    draw(q, 3, "qft.svg")

    coeffs = [0.3, 0.5, 0.7]
    q = az.QFTPhaseKernel(nqubits=3, coeffs=np.array(coeffs), transpile=False).circuit.queue
    nq = len(q) - 3
    draw(q, 3, "qft_phase_kernel.svg",
         symbols={c: f"c_{j}" for j, c in enumerate(coeffs)},
         regions=[(0, nq - 1, "QFT"), (nq, len(q) - 1, "kernel")])

    q = az.Grover(nqubits=3, n_iterations=1, transpile=False).circuit.queue
    draw(q, 3, "grover.svg", regions=[(0, 2, "$H^{\\otimes n}$"), (3, 5, "oracle"),
                                       (6, len(q) - 1, "diffuser")])

    a = az.QPE(n_counting=3, phase=0.375, transpile=False)
    q = a.circuit.queue
    draw(q, 4, "qpe.svg",
         symbols={2 * np.pi * 0.375 * 2**k: f"{2 * 2**k}\\pi\\phi" for k in range(3)},
         regions=[(0, 3, "prep"), (4, 6, "controlled $U^{2^k}$"),
                  (7, len(q) - 1, "QFT$^\\dagger$")],
         qubit_names=["c_0", "c_1", "c_2", "\\psi"])

    a = az.QAE(n_counting=3, theta_A=0.4, transpile=False)
    q = a.circuit.queue
    draw(q, 4, "qae.svg",
         symbols={2 * 2**k * 0.4: f"{2 * 2**k}\\theta_A" for k in range(3)},
         regions=[(0, 3, "$A$, $H$"), (4, 6, "controlled powers"),
                  (7, len(q) - 1, "QFT$^\\dagger$")],
         qubit_names=["c_0", "c_1", "c_2", "w"])


def legend():
    def plot():
        fig, ax = plt.subplots(figsize=(3.4, 0.32))
        ax.axis("off")
        handles = [Patch(fc=YELLOW, ec=C["edge"], label="Clifford gate"),
                   Patch(fc=BLUE, ec=C["edge"], label="magic gate")]
        ax.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=9,
                  labelcolor=C["text"])
        return fig

    save_themed(plot, "legend.svg", pad=0.04)


def _ramped_seed() -> int:
    """First seed that drops exactly one coupling, in step 2."""
    full = len(az.RampedKickedIsing(nqubits=4, n_steps=2, delta_h=0.4,
                                    delta_h_last=0.1).circuit.queue)
    for seed in range(1000):
        a = az.RampedKickedIsing(nqubits=4, n_steps=2, delta_h=0.4, delta_h_last=0.1,
                                 bond_density=0.8, seed=seed)
        q = a.circuit.queue
        # step 1 is 8 kick gates and 3 couplings of 3 gates each
        if len(q) == full - 3 and sum(g.name == "cx" for g in q[:17]) == 6:
            return seed
    raise RuntimeError("no seed found")


RAMPED_SEED = _ramped_seed()

if __name__ == "__main__":
    plt.rcParams.update(FONT)
    plt.rcParams["svg.hashsalt"] = "mpstab"  # stable ids: re-running gives the same files
    variational()
    floquet()
    kicked()
    library()
    legend()
    print(f"figures written to {OUT}")
