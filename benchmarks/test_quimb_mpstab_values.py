import random

import numpy as np
from qibo import Circuit, gates

from mpstab.analysis_scripts.utils import (
    execute_benchmark_circuit,
    generate_partitionated_circuit,
    initialize_backend,
)


def run_once(
    backend: str,
    seed: int,
    nqubits: int = 4,
    nlayers: int = 2,
    replacement_probability: float = 0.5,
    max_bond_dim: int | None = 16,
):

    backend_obj = initialize_backend(
        backend=backend,
        platform=None,
        max_bond_dim=max_bond_dim,
    )

    # Fix all RNGs
    np.random.seed(seed)
    random.seed(seed)

    # Initial state (deterministic)
    initial_state = Circuit(nqubits)
    for q in range(nqubits):
        initial_state.add(gates.RY(q, np.random.uniform(-np.pi, np.pi)))

    # Circuit generation
    circuit = generate_partitionated_circuit(
        nqubits=nqubits,
        nlayers=nlayers,
        replacement_probability=replacement_probability,
    )

    # Deterministic parameters
    params = np.random.uniform(-np.pi, np.pi, size=len(circuit.get_parameters()))
    circuit.set_parameters(params)
    print(params)

    obs_str = "ZX" * (nqubits // 2)
    if nqubits % 2:
        obs_str += "Z"

    expval, _, _ = execute_benchmark_circuit(
        circuit=circuit,
        observable=obs_str,
        backend=backend,
        max_bond_dim=max_bond_dim,
        initial_state=initial_state,
        backend_obj=backend_obj,
    )

    return {
        "expval": float(expval),
        "params": np.array(params, copy=True),
    }


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_reproducibility_same_seed():
    seed = 1234

    for backend in ["quimb", "mpstab"]:
        out1 = run_once(backend=backend, seed=seed)
        out2 = run_once(backend=backend, seed=seed)

        # Parameters must match exactly
        assert np.array_equal(
            out1["params"], out2["params"]
        ), f"{backend}: circuit parameters differ"

        # Expectation values must match
        assert np.isclose(
            out1["expval"], out2["expval"], atol=1e-10
        ), f"{backend}: expectation values differ"

        print(f"[OK] {backend} reproducibility:" f" expval={out1['expval']:.6e}")


if __name__ == "__main__":
    test_reproducibility_same_seed()
