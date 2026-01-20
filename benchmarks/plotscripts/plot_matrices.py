import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, SymLogNorm

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RESULTS_ROOT = Path("../results")
FIG_ROOT = Path("figures/heatmaps")

sns.set_context("talk")
sns.set_style("white")

FIG_ROOT.mkdir(parents=True, exist_ok=True)

SYMLINTHRESH = 1e-3  # seconds


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def save_figure(fig, path_base):
    fig.savefig(f"{path_base}.pdf", bbox_inches="tight")
    fig.savefig(
        f"{path_base}.png",
        bbox_inches="tight",
        dpi=300,
    )


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_results():
    records = []

    for path in RESULTS_ROOT.rglob("results.json"):
        with open(path) as f:
            data = json.load(f)

        args = data["input_arguments"]

        records.append(
            {
                "backend": args["backend"],
                "nqubits": int(args["nqubits"]),
                "replacement_probability": float(args["replacement_probability"]),
                "max_bond_dim": int(args["max_bond_dim"]),
                "median_time": float(data["median_time"]),
            }
        )

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No results.json files found.")

    return df


# ------------------------------------------------------------
# Matrix builders
# ------------------------------------------------------------
def build_matrix(df, backend, p):
    return (
        df[(df.backend == backend) & (df.replacement_probability == p)]
        .pivot_table(
            index="nqubits",
            columns="max_bond_dim",
            values="median_time",
            aggfunc="median",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )


def build_p_matrix(df, backend, nqubits):
    return (
        df[(df.backend == backend) & (df.nqubits == nqubits)]
        .pivot_table(
            index="max_bond_dim",
            columns="replacement_probability",
            values="median_time",
            aggfunc="median",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )


# ------------------------------------------------------------
# Annotation helper
# ------------------------------------------------------------
def annotate_heatmap(ax, data, fmt="{:.1e}"):
    vmin = np.nanmin(data.values)
    vmax = np.nanmax(data.values)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            if not np.isfinite(val):
                continue

            norm_val = (val - vmin) / (vmax - vmin + 1e-12)

            # dark cell -> white text
            # light cell -> black text
            color = "white" if norm_val < 0.5 else "black"

            ax.text(
                j + 0.5,
                i + 0.5,
                fmt.format(val),
                ha="center",
                va="center",
                fontsize=11,
                color=color,
            )


# ------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------
def plot_parallel_heatmaps(df, p):
    mat_mp = build_matrix(df, "mpstab", p)
    mat_qt = build_matrix(df, "qibotn", p)
    mat_mp, mat_qt = mat_mp.align(mat_qt)

    vmin = min(mat_mp.min().min(), mat_qt.min().min())
    vmax = max(mat_mp.max().max(), mat_qt.max().max())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    for ax, mat, title in zip(
        axes,
        [mat_mp, mat_qt],
        ["mpstab", "qibotn"],
    ):
        sns.heatmap(
            mat,
            ax=ax,
            cmap="coolwarm",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cbar=False,
        )
        annotate_heatmap(ax, mat)
        ax.set_title(title)
        ax.set_xlabel("Max bond dimension")
        ax.set_ylabel("Number of qubits")

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(
        axes[0].collections[0],
        cax=cax,
        label="Median execution time [s]",
    )

    fig.suptitle(
        f"Backend comparison (log scale)\n" f"replacement probability p={p}",
        y=0.98,
    )

    path_base = FIG_ROOT / f"parallel_heatmaps_p_{p:.2f}"
    save_figure(fig, path_base)
    plt.close(fig)


def plot_parallel_replacement_heatmaps(df, nqubits):
    mat_mp = build_p_matrix(df, "mpstab", nqubits)
    mat_qt = build_p_matrix(df, "qibotn", nqubits)
    mat_mp, mat_qt = mat_mp.align(mat_qt)

    vmin = min(mat_mp.min().min(), mat_qt.min().min())
    vmax = max(mat_mp.max().max(), mat_qt.max().max())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    for ax, mat, title in zip(
        axes,
        [mat_mp, mat_qt],
        ["mpstab", "qibotn"],
    ):
        sns.heatmap(
            mat,
            ax=ax,
            cmap="coolwarm",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cbar=False,
        )
        annotate_heatmap(ax, mat)
        ax.set_title(title)
        ax.set_xlabel("Replacement probability p")
        ax.set_ylabel("Max bond dimension")

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(
        axes[0].collections[0],
        cax=cax,
        label="Median execution time [s]",
    )

    fig.suptitle(
        f"Replacement probability comparison (log scale)\n" f"{nqubits} qubits",
        y=0.98,
    )

    path_base = FIG_ROOT / f"parallel_replacement_heatmaps_{nqubits}q"
    save_figure(fig, path_base)
    plt.close(fig)


def plot_difference_heatmap(df, p):
    mat_mp = build_matrix(df, "mpstab", p)
    mat_qt = build_matrix(df, "qibotn", p)
    mat_mp, mat_qt = mat_mp.align(mat_qt)

    diff = mat_mp - mat_qt
    vmax = np.nanmax(np.abs(diff.values))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        diff,
        ax=ax,
        cmap="coolwarm",
        norm=SymLogNorm(
            linthresh=SYMLINTHRESH,
            vmin=-vmax,
            vmax=vmax,
        ),
        cbar_kws={"label": "Δ time = mpstab − qibotn [s]"},
    )

    annotate_heatmap(ax, diff, fmt="{:.2e}")

    ax.set_xlabel("Max bond dimension")
    ax.set_ylabel("Number of qubits")
    ax.set_title(f"Execution time difference (symlog)\n" f"p={p}")

    path_base = FIG_ROOT / f"diff_heatmap_p_{p:.2f}"
    save_figure(fig, path_base)
    plt.close(fig)


def plot_speedup_heatmap(df, p):
    mat_mp = build_matrix(df, "mpstab", p)
    mat_qt = build_matrix(df, "qibotn", p)
    mat_mp, mat_qt = mat_mp.align(mat_qt)

    speedup = mat_qt / mat_mp

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        speedup,
        ax=ax,
        cmap="coolwarm",
        norm=LogNorm(),
        cbar_kws={"label": "Speedup = qibotn / mpstab"},
    )

    annotate_heatmap(ax, speedup, fmt="{:.2f}")

    ax.set_xlabel("Max bond dimension")
    ax.set_ylabel("Number of qubits")
    ax.set_title(f"Relative speedup (log scale)\n" f"p={p}")

    path_base = FIG_ROOT / f"speedup_heatmap_p_{p:.2f}"
    save_figure(fig, path_base)
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    df = load_results()

    for p in sorted(df.replacement_probability.unique()):
        plot_parallel_heatmaps(df, p)
        plot_difference_heatmap(df, p)
        plot_speedup_heatmap(df, p)

        print(f"[OK] Backend comparison for p={p}")

    for nqubits in sorted(df.nqubits.unique()):
        plot_parallel_replacement_heatmaps(df, nqubits)
        print(f"[OK] Replacement probability comparison for {nqubits} qubits")


if __name__ == "__main__":
    main()
