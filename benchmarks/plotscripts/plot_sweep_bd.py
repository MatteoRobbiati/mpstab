import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def load_results(path: Path) -> Dict[str, Any]:
    """
    Load a results.json file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_vs_bd(
    base_folder: Path,
    bond_dims: List[int],
    replacement_probability: float,
) -> Dict[str, List[float]]:
    """
    Collect metrics vs bond dimension.
    """
    median_times = []
    mad_times = []
    ave_magic_gates = []

    for bd in bond_dims:
        run_folder = (
            f"{bd}max_bd_{replacement_probability}repl_prob"
        )
        results_path = base_folder / run_folder / "results.json"

        if not results_path.exists():
            raise FileNotFoundError(results_path)

        results = load_results(results_path)

        median_times.append(results["median_time"])
        mad_times.append(results["mad_time"])
        ave_magic_gates.append(results["ave_magic_gates"])

    return {
        "median_time": median_times,
        "mad_time": mad_times,
        "ave_magic_gates": ave_magic_gates,
    }


def plot_performance_vs_bd(
    bond_dims: List[int],
    metrics: Dict[str, List[float]],
) -> None:
    """
    Plot performance metrics vs bond dimension.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Runtime ----
    axes[0].errorbar(
        bond_dims,
        metrics["median_time"],
        yerr=metrics["mad_time"],
        marker="o",
        capsize=4,
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Max bond dimension")
    axes[0].set_ylabel("Median runtime [s]")
    axes[0].set_title("Runtime vs bond dimension")
    axes[0].grid(True, which="both")

    # ---- Magic gates ----
    axes[1].plot(
        bond_dims,
        metrics["ave_magic_gates"],
        marker="o",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Max bond dimension")
    axes[1].set_ylabel("Average number of magic gates")
    axes[1].set_title("Magic gates vs bond dimension")
    axes[1].grid(True, which="both")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ---- Match these to the experiment ----
    NQUBITS = 20
    NLAYERS = 4
    REPLACEMENT_PROBABILITY = 0.75
    BOND_DIMS = [2, 4, 16, 32, 64, 128, 256]

    BASE_FOLDER = Path(
        f"../results/hdw_efficient_ansatz_"
        f"{NQUBITS}qubits_{NLAYERS}layers"
    )

    metrics = collect_vs_bd(
        base_folder=BASE_FOLDER,
        bond_dims=BOND_DIMS,
        replacement_probability=REPLACEMENT_PROBABILITY,
    )

    plot_performance_vs_bd(BOND_DIMS, metrics)