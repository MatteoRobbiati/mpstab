import json
import os
import random
import time
from typing import Any, Dict, List

import numpy as np
import scipy.stats as sp
from qibo import Circuit, gates
from qibo.backends import get_backend, set_backend

from mpstab.evolutors.models import HybridSurrogate
from mpstab.models.ansatze import HardwareEfficient
from mpstab.models.utils import obs_string_to_qibo_hamiltonian

SUPPORTED_BACKENDS = [
    "mpstab",
    "numpy",
    "qibojit",
    "qibotn",
]


# ---------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------
def generate_partitionated_circuit(
    nqubits: int,
    nlayers: int,
    magic_fraction: float,
) -> Circuit:
    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)

    _, part_circuit = ansatz.partitionate_circuit(
        replacement_probability=1.0 - magic_fraction,
        replacement_method="closest",
    )

    return part_circuit


# ---------------------------------------------------------------------
# Backend initialization
# ---------------------------------------------------------------------
def initialize_backend(
    backend: str,
    platform: str | None,
    max_bond_dim: int | None,
):
    """
    Strategy:
    1. set_backend
    2. get_backend
    3. configure if qibotn
    """

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend {backend} not supported. " f"Choose among {SUPPORTED_BACKENDS}."
        )

    # mpstab still needs a concrete execution backend
    exec_backend = "numpy" if backend == "mpstab" else backend

    # 1. Set backend globally
    set_backend(
        backend=exec_backend,
        platform=platform,
    )

    # 2. Retrieve backend object
    backend_obj = get_backend()

    # 3. Configure tensor‑network backend
    if backend == "qibotn":
        backend_obj.setup_backend_specifics(
            quimb_backend="jax",
            contractions_optimizer="auto-hq",
        )

        if max_bond_dim is not None:
            backend_obj.configure_tn_simulation(max_bond_dimension=max_bond_dim)

    return backend_obj


# ---------------------------------------------------------------------
# Circuit execution
# ---------------------------------------------------------------------
def execute_benchmark_circuit(
    circuit: Circuit,
    observable: str,
    backend: str,
    max_bond_dim: int | None,
    initial_state: Circuit | None,
    replacement_probability: float,
    backend_obj,
) -> tuple[float, float, int | None]:

    start_time = time.time()

    # ---- mpstab surrogate backend ----
    if backend == "mpstab":
        ansatz = HardwareEfficient(
            nqubits=circuit.nqubits,
            nlayers=0,
        )
        ansatz.circuit = circuit

        evolutor = HybridSurrogate(
            ansatz=ansatz,
            max_bond_dimension=max_bond_dim,
            initial_state=initial_state,
        )

        expval, partitions = evolutor.expectation_from_partition(
            observable=observable,
            replacement_probability=replacement_probability,
            return_partitions=True,
        )

        elapsed_time = time.time() - start_time
        return expval, elapsed_time, len(partitions["magic_gates"])

    # ---- Build full circuit ----
    full_circuit = Circuit(circuit.nqubits)
    if initial_state is not None:
        full_circuit += initial_state
    full_circuit += circuit

    # ---- qibotn tensor‑network backend ----
    if backend == "qibotn":
        ham = obs_string_to_qibo_hamiltonian(
            observable,
            backend=backend_obj,
        )

        expval = ham.expectation(full_circuit)

        elapsed_time = time.time() - start_time
        return float(expval), elapsed_time, None

    # ---- Standard Qibo backends ----
    result = full_circuit()
    ham = obs_string_to_qibo_hamiltonian(
        observable,
        backend=backend_obj,
    )

    expval = ham.expectation(result.state())

    elapsed_time = time.time() - start_time
    return float(expval), elapsed_time, None


# ---------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------
def run_experiment(
    backend: str,
    max_bond_dim: int | None,
    replacement_probability: float,
    nqubits: int,
    nlayers: int,
    nruns: int = 10,
    rng_seed: int = 42,
    set_initial_state: bool = True,
    platform: str | None = None,
) -> None:

    random.seed(rng_seed)
    np.random.seed(rng_seed)

    # ✅ Backend initialized ONCE
    backend_obj = initialize_backend(
        backend=backend,
        platform=platform,
        max_bond_dim=max_bond_dim,
    )

    obs_str = "ZX" * (nqubits // 2)
    if nqubits % 2:
        obs_str += "Z"

    base_folder = (
        f"results/hdw_efficient_ansatz/"
        f"{nqubits}qubits_{nlayers}layers/"
        f"backend_{backend}_platform_{platform}/"
        f"bd_{max_bond_dim}_p_{replacement_probability}"
    )
    os.makedirs(base_folder, exist_ok=True)

    results: Dict[str, List[Any]] = {
        "measured_observable": obs_str,
        "times": [],
        "expvals": [],
        "magic_gates": [],
    }

    # --------------------------------------------------
    # DRY RUN: first iteration is discarded
    # --------------------------------------------------
    total_runs = nruns + 1

    for run_idx in range(total_runs):
        if set_initial_state:
            initial_state = Circuit(nqubits)
            for q in range(nqubits):
                initial_state.add(gates.RY(q, np.random.uniform(-np.pi, np.pi)))
        else:
            initial_state = None

        circuit = generate_partitionated_circuit(
            nqubits=nqubits,
            nlayers=nlayers,
            magic_fraction=replacement_probability,
        )

        expval, elapsed_time, n_magic = execute_benchmark_circuit(
            circuit=circuit,
            observable=obs_str,
            backend=backend,
            max_bond_dim=max_bond_dim,
            initial_state=initial_state,
            replacement_probability=replacement_probability,
            backend_obj=backend_obj,
        )

        # Skip first run (dry run)
        if run_idx == 0:
            continue

        results["times"].append(elapsed_time)
        results["expvals"].append(expval)
        if n_magic is not None:
            results["magic_gates"].append(n_magic)

    results["median_time"] = float(np.median(results["times"]))
    results["mad_time"] = float(sp.median_abs_deviation(results["times"]))

    if results["magic_gates"]:
        results["ave_magic_gates"] = float(np.mean(results["magic_gates"]))

    results["input_arguments"] = dict(
        backend=backend,
        max_bond_dim=max_bond_dim,
        replacement_probability=replacement_probability,
        nqubits=nqubits,
        nlayers=nlayers,
        nruns=nruns,
        rng_seed=rng_seed,
        set_initial_state=set_initial_state,
        platform=platform,
        dry_run_discarded=True,
    )

    with open(os.path.join(base_folder, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
