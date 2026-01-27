import json
import os

import numpy as np

from mpstab.analysis_scripts.utils import run_experiment


def load_params(folder: str, nruns: int):
    return [
        np.load(os.path.join(folder, f"params_run_{i}.npy"))
        for i in range(1, nruns + 1)
    ]


def test_run_experiment_mpstab_vs_quimb():
    seed = 123
    nqubits = 4
    nlayers = 2
    nruns = 3
    replacement_probability = 0.5
    max_bond_dim = 16

    results = {}

    for backend in ["mpstab", "quimb"]:
        # --------------------------------------------------
        # Run experiment (results go to backend-specific folder)
        # --------------------------------------------------
        run_experiment(
            backend=backend,
            max_bond_dim=max_bond_dim,
            replacement_probability=replacement_probability,
            nqubits=nqubits,
            nlayers=nlayers,
            nruns=nruns,
            rng_seed=seed,
        )

        folder = (
            f"results/hdw_efficient_ansatz/"
            f"{nqubits}qubits_{nlayers}layers/"
            f"backend_{backend}_platform_None/"
            f"bd_{max_bond_dim}_p_{replacement_probability}"
        )

        # --------------------------------------------------
        # Load results
        # --------------------------------------------------
        with open(os.path.join(folder, "results.json")) as f:
            results[backend] = json.load(f)

        results[backend]["params"] = load_params(folder, nruns)

    # --------------------------------------------------
    # PARAMETER CHECK
    # --------------------------------------------------
    for i, (p1, p2) in enumerate(
        zip(results["mpstab"]["params"], results["quimb"]["params"])
    ):
        assert np.array_equal(p1, p2), f"Parameters differ at run {i + 1}"

    # --------------------------------------------------
    # EXPVAL CHECK
    # --------------------------------------------------
    exp_mpstab = np.array(results["mpstab"]["expvals"])
    exp_quimb = np.array(results["quimb"]["expvals"])

    assert np.allclose(
        exp_mpstab, exp_quimb, atol=1e-10
    ), "Expectation values differ between mpstab and quimb"

    print(
        "[OK] run_experiment produces identical "
        "params and expvals for mpstab and quimb"
    )


if __name__ == "__main__":
    test_run_experiment_mpstab_vs_quimb()
