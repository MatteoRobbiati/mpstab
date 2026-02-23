# TODO bisogna mettere i check a quimb e qibo

import json
import os
import random
import time
from typing import Any, Dict, List

import numpy as np
import scipy.stats as sp
from qibo import Circuit, gates
from qibo.backends import get_backend, set_backend
from qibo.gates.abstract import ParametrizedGate
from quimb import pauli
from quimb.tensor import CircuitMPS

from mpstab.engines.stabilizers.native import NativeStabilizersEngine
from mpstab.engines.stabilizers.stim import StimEngine
from mpstab.engines.tensor_networks.native import NativeTensorNetworkEngine
from mpstab.engines.tensor_networks.quimb import QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient
from mpstab.models.utils import obs_string_to_qibo_hamiltonian

SUPPORTED_BACKENDS = ["mpstab", "qibo", "quimb"]

MPSTAB_PLATFORMS = ["quimb", "stim", "native_tn", "native_stab"]
QIBO_PLATFORMS = [
    "jax",
    "numpy",
    "qibojit",
]
QUIMB_PLATFORMS = [
    "jax",
    "numpy",
]

GATE_MAP = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "t": "T",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "u3": "U3",
    "cx": "CX",
    "cnot": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "iswap": "ISWAP",
    "swap": "SWAP",
    "ccx": "CCX",
    "ccy": "CCY",
    "ccz": "CCZ",
    "toffoli": "TOFFOLI",
    "cswap": "CSWAP",
    "fredkin": "FREDKIN",
    "fsim": "fsim",
    "measure": "measure",
}


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


def qibo_circuit_to_quimb(nqubits, qibo_circ, **circuit_kwargs):
    """
    Convert a Qibo Circuit to a Quimb Circuit. Measurement gates are ignored. If are given gates not supported by Quimb, an error is raised.

    Parameters
    ----------
    qibo_circ : qibo.models.circuit.Circuit
        The circuit to convert.
    quimb_circuit_type : type
        The Quimb circuit class to use (Circuit, CircuitMPS, etc).
    circuit_kwargs : dict
        Extra arguments to pass to the Quimb circuit constructor.

    Returns
    -------
    circ : quimb.tensor.circuit.Circuit
        The converted circuit.
    """

    circ = CircuitMPS(nqubits, **circuit_kwargs)

    for gate in qibo_circ.queue:
        gate_name = getattr(gate, "name", None)
        quimb_gate_name = GATE_MAP.get(gate_name, None)
        if quimb_gate_name == "measure":
            continue
        if quimb_gate_name is None:
            raise ValueError(f"Gate {gate_name} not supported in Quimb backend.")

        params = getattr(gate, "parameters", ())
        qubits = getattr(gate, "qubits", ())

        is_parametrized = isinstance(gate, ParametrizedGate) and getattr(
            gate, "trainable", True
        )
        if is_parametrized:
            circ.apply_gate(
                quimb_gate_name, *params, *qubits, parametrized=is_parametrized
            )
        else:
            circ.apply_gate(
                quimb_gate_name,
                *params,
                *qubits,
            )

    return circ


def execute_benchmark_circuit(
    circuit: Circuit,
    observable: str,
    backend: str,
    platform: str,
    max_bond_dim: int | None,
    initial_state: Circuit | None,
) -> tuple[float, float, int | None]:

    start_time = time.time()

    if backend == "mpstab":

        if platform[0] == "quimb":
            tn_engine = QuimbEngine()
        elif platform[0] == "native_tn":
            tn_engine = NativeTensorNetworkEngine()
        else:
            raise ValueError(f"Unknown TN engine: {platform[0]}")
        if platform[1] == "stim":
            stab_engine = StimEngine()
        elif platform[1] == "native_stab":
            stab_engine = NativeStabilizersEngine()
        else:
            raise ValueError(f"Unknown stabilizer engine: {platform[1]}")

        evolutor = HSMPO(
            ansatz=circuit,
            max_bond_dimension=max_bond_dim,
            initial_state=initial_state,
        )

        evolutor.set_engines(tn_engine=tn_engine, stab_engine=stab_engine)

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

    # For both Quimb and Qibo we build full circuit manually
    full_circuit = Circuit(circuit.nqubits)
    if initial_state is not None:
        full_circuit += initial_state
    full_circuit += circuit

    if backend == "quimb":

        psi_ket = qibo_circuit_to_quimb(
            nqubits=full_circuit.nqubits,
            qibo_circ=full_circuit,
            gate_opts={"max_bond": max_bond_dim},
        ).psi

        non_i_ops = {
            i: op.upper() for i, op in enumerate(observable) if op.upper() != "I"
        }

        psi_op = psi_ket.copy()

        for site, label in non_i_ops.items():
            psi_op.gate_(pauli(label), site)

        expectation = (psi_ket.H & psi_op).contract(optimize="auto-hq").real

        elapsed_time = time.time() - start_time
        return float(expectation), elapsed_time, None

    if backend == "qibo":

        set_backend(backend=platform)
        backend_obj = get_backend()
        result = full_circuit()

        ham = obs_string_to_qibo_hamiltonian(
            observable,
            backend=backend_obj,
        )

        expval = ham.expectation(result.state())

        elapsed_time = time.time() - start_time
        return float(expval), elapsed_time, None


def run_experiment(
    backend: str,
    max_bond_dim: int | None,
    replacement_probability: float,
    nqubits: int,
    nlayers: int,
    nruns: int = 10,
    rng_seed: int = 42,
    set_initial_state: bool = True,
    platform: str | tuple | None = None,
) -> None:

    # Defining the observable
    obs_str = "ZX" * (nqubits // 2)
    if nqubits % 2:
        obs_str += "Z"

    # Constructing the output folder
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

    # Running simulations
    total_runs = nruns + 1
    for run_idx in range(total_runs):

        np.random.seed(rng_seed + run_idx + 1)
        random.seed(rng_seed + run_idx + 1)

        if set_initial_state:
            initial_state = Circuit(nqubits)
            for q in range(nqubits):
                # NOTE: For Stim backend, this random initialization will likely cause
                # a non-Clifford error unless replacement_probability ensures Clifford gates
                # or set_initial_state is set to False by the caller.
                initial_state.add(gates.RY(q, np.random.uniform(-np.pi, np.pi)))
        else:
            initial_state = None

        # Generate a circuit where we replace magic gates with probablity `replacement_probability`
        circuit = generate_partitionated_circuit(
            nqubits=nqubits,
            nlayers=nlayers,
            replacement_probability=replacement_probability,
        )

        # Hydrate the rotations with new angles
        magic_gates_info = []
        for i, gate in enumerate(circuit.queue):
            if not gate.clifford:
                new_angle = np.random.uniform(-np.pi, np.pi)
                gate.parameters = (new_angle,)
                magic_gates_info.append([new_angle, i])

        # Execute benchmark from constructed circuit and parameters
        expval, elapsed_time, n_magic = execute_benchmark_circuit(
            circuit=circuit,
            observable=obs_str,
            backend=backend,
            platform=platform,
            max_bond_dim=max_bond_dim,
            initial_state=initial_state,
        )

        # Skip first run
        if run_idx == 0:
            continue

        # Save magic parameters for this run
        if magic_gates_info == []:
            np.save(
                os.path.join(base_folder, f"magic_params_run_{run_idx}.npy"),
                np.asarray(0),
            )
        else:
            np.save(
                os.path.join(base_folder, f"magic_params_run_{run_idx}.npy"),
                np.asarray(magic_gates_info).T[0],
            )

        results["times"].append(elapsed_time)
        results["expvals"].append(expval)
        if n_magic is not None:
            results["magic_gates"].append(n_magic)

    # Computing and saving statistics
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
