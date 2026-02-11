import json
import os
import random
import time
from typing import Any, Dict, List

import numpy as np
import scipy.stats as sp
import stim
from qibo import Circuit, gates
from qibo.backends import get_backend, set_backend
from quimb import pauli

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient
from mpstab.models.utils import obs_string_to_qibo_hamiltonian

SUPPORTED_BACKENDS = [
    "mpstab",
    "numpy",
    "qibojit",
    "quimb",
    "stim",
]


# ---------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------
def generate_partitionated_circuit(
    nqubits: int,
    nlayers: int,
    replacement_probability: float,
) -> Circuit:
    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)

    _, part_circuit = ansatz.partitionate_circuit(
        replacement_probability=replacement_probability,
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

    # custom rules
    if backend == "mpstab":
        exec_backend = "numpy"
    elif backend == "quimb":
        exec_backend = "qibotn"
        platform = "quimb"
    elif backend == "stim":
        # Stim creates its own simulation, use numpy for circuit construction
        exec_backend = "numpy"
    else:
        exec_backend = backend

    # 1. Set backend globally
    set_backend(
        backend=exec_backend,
        platform=platform,
    )

    # 2. Retrieve backend object
    backend_obj = get_backend()

    # 3. Configure tensor‑network backend
    # We will use it to do some translations, even if the actual simulation
    # will be run using Quimb
    if backend == "quimb":
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

        evolutor = HSMPO(
            ansatz=ansatz,
            max_bond_dimension=max_bond_dim,
            initial_state=initial_state,
        )

        expval = evolutor.expectation(
            observable=observable,
        )

        # Counting Clifford gates as in the original code snippet logic
        n_magic_gates = 0
        for g in circuit.queue:
            if g.clifford:
                n_magic_gates += 1

        elapsed_time = time.time() - start_time
        return expval, elapsed_time, n_magic_gates

    # ---- Build full circuit ----
    full_circuit = Circuit(circuit.nqubits)
    if initial_state is not None:
        full_circuit += initial_state
    full_circuit += circuit

    # ---- stim clifford backend ----
    if backend == "stim":
        # 1. Verify circuit is all Clifford (count magic gates)
        n_magic_gates = 0
        for g in full_circuit.queue:
            if not g.clifford:
                n_magic_gates += 1

        if n_magic_gates > 0:
            raise RuntimeError(
                f"Backend 'stim' requires an all-Clifford circuit. "
                f"Found {n_magic_gates} non-Clifford gates."
            )

        # 2. Convert to Stim circuit
        stim_circuit = stim.Circuit()
        for g in full_circuit.queue:
            q_indices = g.qubits

            # Helper to check rotation angles
            def is_approx(val, target, atol=1e-5):
                return np.isclose(
                    val % (2 * np.pi), target % (2 * np.pi), atol=atol
                ) or np.isclose(
                    val % (2 * np.pi), (target % (2 * np.pi)) + 2 * np.pi, atol=atol
                )

            if g.name == "h":
                stim_circuit.append("H", q_indices)
            elif g.name == "x":
                stim_circuit.append("X", q_indices)
            elif g.name == "y":
                stim_circuit.append("Y", q_indices)
            elif g.name == "z":
                stim_circuit.append("Z", q_indices)
            elif g.name == "cx" or g.name == "cnot":
                stim_circuit.append("CNOT", q_indices)
            elif g.name == "cz":
                stim_circuit.append("CZ", q_indices)
            elif g.name == "swap":
                stim_circuit.append("SWAP", q_indices)
            elif g.name == "s":
                stim_circuit.append("S", q_indices)
            elif g.name == "sdg":
                stim_circuit.append("S_DAG", q_indices)
            elif g.name in ["rx", "ry", "rz"]:
                # Handle Clifford rotations
                theta = g.parameters[0]
                axis = g.name[1].upper()  # X, Y, or Z

                if is_approx(theta, 0):
                    continue  # Identity
                elif is_approx(theta, np.pi) or is_approx(theta, -np.pi):
                    stim_circuit.append(axis, q_indices)
                elif is_approx(theta, np.pi / 2):
                    stim_circuit.append(f"SQRT_{axis}", q_indices)
                elif is_approx(theta, -np.pi / 2):
                    stim_circuit.append(f"SQRT_{axis}_DAG", q_indices)
                else:
                    raise ValueError(
                        f"Gate {g} with angle {theta} is not a supported Clifford gate for Stim conversion."
                    )
            elif g.name == "id":
                continue
            else:
                # Fallback for other gates or raise error
                raise NotImplementedError(
                    f"Gate {g.name} not implemented in Stim converter."
                )

        # 3. Simulate
        sim = stim.TableauSimulator()
        sim.do(stim_circuit)

        # 4. Compute Expectation
        # Stim expects Pauli string object.
        # observable is like "ZX..." corresponding to qubits 0, 1, ...
        # If the string length < nqubits, we might need to pad with I or raise error.
        # Assuming observable string length matches nqubits as per generation logic.
        pauli_string = stim.PauliString(observable)
        expval = sim.peek_observable_expectation(pauli_string)

        elapsed_time = time.time() - start_time
        return float(expval), elapsed_time, n_magic_gates

    # ---- qibotn tensor‑network backend ----
    if backend == "quimb":

        psi_ket = backend_obj._qibo_circuit_to_quimb(
            full_circuit, gate_opts={"max_bond": max_bond_dim}
        ).psi

        non_i_ops = {
            i: op.upper() for i, op in enumerate(observable) if op.upper() != "I"
        }

        psi_op = psi_ket.copy()

        for site, label in non_i_ops.items():
            psi_op.gate_(pauli(label), site)

        expectation = (psi_ket.H & psi_op).contract(all, optimize="auto-hq").real

        elapsed_time = time.time() - start_time
        return float(expectation), elapsed_time, None

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

    # ✅ Backend initialized ONCE
    backend_obj = initialize_backend(
        backend=backend,
        platform=platform,
        max_bond_dim=max_bond_dim,
    )

    # 1. --- Defining the observable
    obs_str = "ZX" * (nqubits // 2)
    if nqubits % 2:
        obs_str += "Z"

    # 2. --- Constructing the output folder
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

    # --- DRY RUN: first iteration is discarded
    total_runs = nruns + 1

    for run_idx in range(total_runs):

        np.random.seed(rng_seed + run_idx + 1)
        random.seed(rng_seed + run_idx + 1)
        #        bkd = get_backend()
        #        bkd.set_seed(rng_seed + run_idx + 1)

        if set_initial_state:
            initial_state = Circuit(nqubits)
            for q in range(nqubits):
                # NOTE: For Stim backend, this random initialization will likely cause
                # a non-Clifford error unless replacement_probability ensures Clifford gates
                # or set_initial_state is set to False by the caller.
                initial_state.add(gates.RY(q, np.random.uniform(-np.pi, np.pi)))
        else:
            initial_state = None

        # --- Generate a circuit where we replace magic gates with probablity `replacement_probability`
        circuit = generate_partitionated_circuit(
            nqubits=nqubits,
            nlayers=nlayers,
            replacement_probability=replacement_probability,
        )

        # --- Hydrate the rotations with new angles
        magic_gates_info = []
        for i, gate in enumerate(circuit.queue):
            if not gate.clifford:
                new_angle = np.random.uniform(-np.pi, np.pi)
                gate.parameters = (new_angle,)
                magic_gates_info.append([new_angle, i])

        expval, elapsed_time, n_magic = execute_benchmark_circuit(
            circuit=circuit,
            observable=obs_str,
            backend=backend,
            max_bond_dim=max_bond_dim,
            initial_state=initial_state,
            backend_obj=backend_obj,
        )

        # Skip first run (dry run)
        if run_idx == 0:
            continue

        # Save magic parameters for this run
        np.save(
            os.path.join(base_folder, f"magic_params_run_{run_idx}.npy"),
            np.asarray(magic_gates_info).T[0],
        )

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
