import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# User configuration
# --------------------------------------------------
BASE_DIR = Path("../results/hdw_efficient_ansatz")

NQUBITS = 12
NLAYERS = 4
REPLACEMENT_PROB = 0.75

BACKENDS = ["mpstab", "qibotn"]
BOND_DIMS = [4, 8, 16, 32, 64]

# Colors / markers
STYLES = {
    "mpstab": dict(marker="o", linestyle="-", label="mpstab"),
    "qibotn": dict(marker="s", linestyle="--", label="qibotn"),
}


# --------------------------------------------------
# Data loading
# --------------------------------------------------
def load_results(
    backend: str,
) -> Dict[int, Dict[str, float]]:
    """
    Returns:
        { bond_dim : results_dict }
    """
    data = {}

    if backend == "mpstab":
        platform = "None"
    else:
        platform = "quimb"

    backend_dir = (
        BASE_DIR
        / f"{NQUBITS}qubits_{NLAYERS}layers"
        / f"backend_{backend}_platform_{platform}"
    )

    if not backend_dir.exists():
        print(f"[WARN] Missing directory: {backend_dir}")
        return data

    for bd in BOND_DIMS:
        res_file = backend_dir / f"bd_{bd}_p_{REPLACEMENT_PROB}" / "results.json"

        if not res_file.exists():
            print(f"[WARN] Missing file: {res_file}")
            continue

        with open(res_file) as f:
            data[bd] = json.load(f)

    return data


# --------------------------------------------------
# Plot
# --------------------------------------------------
def plot_runtime_vs_bd():
    plt.figure(figsize=(7, 5))

    for backend in BACKENDS:
        data = load_results(backend)

        bds = sorted(data.keys())
        times = [data[bd]["median_time"] for bd in bds]

        plt.plot(
            bds,
            times,
            **STYLES[backend],
        )

    plt.xscale("log", base=2)
    plt.yscale("log")

    plt.xlabel("Bond dimension")
    plt.ylabel("Median runtime [s]")
    plt.title(f"Runtime vs bond dimension\n" f"{NQUBITS} qubits, {NLAYERS} layers")

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_runtime_vs_bd()
