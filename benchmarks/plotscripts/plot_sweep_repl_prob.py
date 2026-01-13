import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt


def load_results(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_vs_replacement_probability(
    base_folder: Path,
    replacement_probabilities: List[float],
    max_bond_dim: int,
) -> Dict[str, List[float]]:
    median_times = []
    mad_times = []
    ave_magic_gates = []

    for p in replacement_probabilities:
        run_folder = f"{max_bond_dim}max_bd_{p}repl_prob"
        results_path = base_folder / run_folder / "results.json"

        if not results_path.exists():
            raise FileNotFoundError(
                f"Missing results file: {results_path}"
            )

        results = load_results(results_path)

        median_times.append(results["median_time"])
        mad_times.append(results["mad_time"])
        ave_magic_gates.append(results["ave_magic_gates"])

    return {
        "median_time": median_times,
        "mad_time": mad_times,
        "ave_magic_gates": ave_magic_gates,
    }


def plot_vs_replacement_probability(
    replacement_probabilities: List[float],
    metrics: Dict[str, List[float]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Runtime ----
    axes[0].errorbar(
        replacement_probabilities,
        metrics["median_time"],
        yerr=metrics["mad_time"],
        marker="o",
        capsize=4,
    )
    axes[0].set_xlabel("Replacement probability")
    axes[0].set_ylabel("Median runtime [s]")
    axes[0].set_title("Runtime vs replacement probability")
    axes[0].set_yscale("log")
    axes[0].grid(True)

    # ---- Magic gates ----
    axes[1].plot(
        replacement_probabilities,
        metrics["ave_magic_gates"],
        marker="o",
    )
    axes[1].set_xlabel("Replacement probability")
    axes[1].set_ylabel("Average number of magic gates")
    axes[1].set_title("Magic gates vs replacement probability")
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ---- EXACT parameters from your bash script ----
    MAX_BOND_DIM = 256
    NQUBITS = 15
    NLAYERS = 4

    REPLACEMENT_PROBABILITIES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    BASE_FOLDER = Path(
        f"../results/hdw_efficient_ansatz_"
        f"{NQUBITS}qubits_{NLAYERS}layers"
    )

    metrics = collect_vs_replacement_probability(
        base_folder=BASE_FOLDER,
        replacement_probabilities=REPLACEMENT_PROBABILITIES,
        max_bond_dim=MAX_BOND_DIM,
    )

    plot_vs_replacement_probability(
        REPLACEMENT_PROBABILITIES,
        metrics,
    )