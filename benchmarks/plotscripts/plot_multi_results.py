import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# ==================================================
# Configuration
# ==================================================
BASE_DIR = Path("results/hdw_efficient_ansatz")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

NQUBITS_LIST = [4, 6, 8, 10, 12]
REPLACEMENT_PROBS = [0, 0.25, 0.5, 0.75, 0.99]
BACKENDS = ["mpstab", "qibotn"]

NLAYERS = 3

STYLES = {
    "mpstab": dict(marker="o", linestyle="-", label="mpstab"),
    "qibotn": dict(marker="s", linestyle="--", label="qibotn"),
}


# ==================================================
# Helpers
# ==================================================
def platform_from_backend(backend: str) -> str:
    return "None" if backend == "mpstab" else "quimb"


def load_result(
    backend: str,
    nqubits: int,
    bond_dim: int,
    replacement_prob: float,
) -> Dict | None:

    platform = platform_from_backend(backend)

    path = (
        BASE_DIR
        / f"{nqubits}qubits_{NLAYERS}layers"
        / f"backend_{backend}_platform_{platform}"
        / f"bd_{bond_dim}_p_{replacement_prob}"
        / "results.json"
    )

    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def bond_dims_for_nqubits(nqubits: int):
    return [2**e for e in range(1, nqubits // 2 + 1)]


def filter_positive(xs, ys, errs=None):
    xs_f, ys_f, errs_f = [], [], []

    for i, (x, y) in enumerate(zip(xs, ys)):
        if y is None or not np.isfinite(y) or y <= 0:
            continue
        xs_f.append(x)
        ys_f.append(y)
        if errs is not None:
            errs_f.append(errs[i])

    if errs is None:
        return xs_f, ys_f
    return xs_f, ys_f, errs_f


# ==================================================
# Plot 1: Runtime vs bond dimension
# ==================================================
def plot_vs_bond_dimension():
    outdir = FIG_DIR / "vs_bond_dimension"
    outdir.mkdir(exist_ok=True)

    for nqubits in NQUBITS_LIST:
        for p in REPLACEMENT_PROBS:

            plt.figure(figsize=(7, 5))
            plotted = False

            for backend in BACKENDS:
                bds = bond_dims_for_nqubits(nqubits)
                times, errs = [], []

                for bd in bds:
                    res = load_result(backend, nqubits, bd, p)
                    if res is None:
                        times.append(np.nan)
                        errs.append(np.nan)
                        continue
                    times.append(res["median_time"])
                    errs.append(res["mad_time"])

                bds_f, times_f, errs_f = filter_positive(bds, times, errs)

                if not times_f:
                    continue

                plotted = True
                plt.errorbar(
                    bds_f,
                    times_f,
                    yerr=errs_f,
                    **STYLES[backend],
                    capsize=3,
                )

            if not plotted:
                plt.close()
                continue

            plt.xscale("log", base=2)
            plt.yscale("log")
            plt.xlabel("Bond dimension")
            plt.ylabel("Median runtime [s]")
            plt.title(f"Runtime vs bond dimension\n" f"{nqubits} qubits, p={p}")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()

            fname = f"runtime_vs_bd_" f"nq{nqubits}_p{p}.pdf"
            plt.savefig(outdir / fname)
            plt.close()


# ==================================================
# Plot 2: Runtime vs number of qubits
# ==================================================
def plot_vs_qubits(bond_dim: int, p: float):
    outdir = FIG_DIR / "vs_qubits"
    outdir.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 5))
    plotted = False

    for backend in BACKENDS:
        times = []

        for nqubits in NQUBITS_LIST:
            res = load_result(backend, nqubits, bond_dim, p)
            times.append(np.nan if res is None else res["median_time"])

        nq_f, times_f = filter_positive(NQUBITS_LIST, times)

        if not times_f:
            continue

        plotted = True
        plt.plot(
            nq_f,
            times_f,
            **STYLES[backend],
        )

    if not plotted:
        plt.close()
        return

    plt.yscale("log")
    plt.xlabel("Number of qubits")
    plt.ylabel("Median runtime [s]")
    plt.title(f"Runtime vs qubits\n" f"BD={bond_dim}, p={p}")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    fname = f"runtime_vs_qubits_bd{bond_dim}_p{p}.pdf"
    plt.savefig(outdir / fname)
    plt.close()


# ==================================================
# Plot 3: Runtime vs replacement probability
# ==================================================
def plot_vs_replacement_probability(
    nqubits: int,
    bond_dim: int,
):
    outdir = FIG_DIR / "vs_replacement_probability"
    outdir.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 5))
    plotted = False

    for backend in BACKENDS:
        times = []

        for p in REPLACEMENT_PROBS:
            res = load_result(backend, nqubits, bond_dim, p)
            times.append(np.nan if res is None else res["median_time"])

        p_f, times_f = filter_positive(REPLACEMENT_PROBS, times)

        if not times_f:
            continue

        plotted = True
        plt.plot(
            p_f,
            times_f,
            **STYLES[backend],
        )

    if not plotted:
        plt.close()
        return

    plt.yscale("log")
    plt.xlabel("Replacement probability")
    plt.ylabel("Median runtime [s]")
    plt.title(
        f"Runtime vs replacement probability\n" f"{nqubits} qubits, BD={bond_dim}"
    )
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    fname = f"runtime_vs_p_nq{nqubits}_bd{bond_dim}.pdf"
    plt.savefig(outdir / fname)
    plt.close()


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":

    # 1️⃣ Bond‑dimension scaling (all combinations)
    plot_vs_bond_dimension()

    # 2️⃣ Qubit scaling (representative slice)
    plot_vs_qubits(
        bond_dim=16,
        p=0.5,
    )

    # 3️⃣ Replacement‑probability scaling
    plot_vs_replacement_probability(
        nqubits=12,
        bond_dim=16,
    )
