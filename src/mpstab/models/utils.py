import random
from copy import deepcopy
from typing import List, Optional

import networkx as nx
import numpy as np
from qibo import Circuit, gates, hamiltonians, symbols
from qibo.backends import Backend, get_backend
from qibo.noise import NoiseModel, PauliError, ReadoutError
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.placer import Random
from qibo.transpiler.router import ShortestPaths
from qibo.transpiler.unroller import NativeGates, Unroller


def hardware_compatible_circuit(
    circuit: Circuit,
    native_gates: Optional[List] = [gates.GPI2, gates.RZ, gates.Z, gates.CZ],
    connectivity: Optional[nx.Graph] = None,
):

    natives = NativeGates(0).from_gatelist(native_gates)
    unroller = Unroller(natives)

    if connectivity is None:
        circuit = unroller(circuit)
        return circuit
    else:
        custom_passes = [Preprocessing(), Random(), ShortestPaths(), unroller]
        custom_pipeline = Passes(custom_passes, connectivity=connectivity)
        transpiled_circ, final_layout = custom_pipeline(circuit)
        return transpiled_circ


def get_closest_angle(old_angle, candidates):
    """Return the closest angle to `old_angle` among some `candidates`."""
    differences = np.abs(
        np.arctan2(np.sin(candidates - old_angle), np.cos(candidates - old_angle))
    )
    closest_index = np.argmin(differences)
    return candidates[closest_index]


def replace_non_clifford_gate(gate, replacement_method, candidates=None):
    """Replace non‐Clifford RX/RY/RZ or GPI2 gate with a Clifford one."""
    # RX/RY/RZ branch
    if gate.name in ("rx", "ry", "rz"):
        if candidates is None:
            candidates = np.arange(-2, 3) * np.pi / 2.0
        old_angle = gate.parameters[0]
        if replacement_method == "random":
            new_angle = random.choice(candidates)
        elif replacement_method == "closest":
            new_angle = get_closest_angle(old_angle, candidates)
        else:
            raise ValueError(f"Unknown method {replacement_method!r}")
        new_gate = deepcopy(gate)
        new_gate.parameters = [new_angle]
        return new_gate

    # GPI2 branch
    elif isinstance(gate, gates.GPI2):
        # only multiples of π/2 are legal Clifford GPI2 angles
        if candidates is None:
            candidates = np.arange(4) * (np.pi / 2.0)  # [0, π/2, π, 3π/2]
        # extract the current angle; assuming you stored it as gate.angle
        old_angle = gate.parameters[0]
        if replacement_method == "random":
            new_angle = random.choice(candidates)
        elif replacement_method == "closest":
            new_angle = get_closest_angle(old_angle, candidates)
        else:
            raise ValueError(f"Unknown method {replacement_method!r}")
        # re-instantiate a fresh GPI2 at the chosen Clifford angle
        new_gate = deepcopy(gate)
        new_gate.parameters = [new_angle]
        return new_gate

    else:
        raise NotImplementedError(
            f"replace_non_clifford_gate does not support gate type {gate.name}"
        )


def build_noise_model(
    nqubits: int,
    local_pauli_noise_sigma: float,
    readout_bit_flip_prob: float = None,
):
    """Costruct noise model as a local Pauli noise channel + readout noise."""
    noise_model = NoiseModel()
    for q in range(nqubits):
        noise_model.add(
            PauliError(
                [
                    ("X", np.abs(np.random.normal(0, local_pauli_noise_sigma))),
                    ("Y", np.abs(np.random.normal(0, local_pauli_noise_sigma))),
                    ("Z", np.abs(np.random.normal(0, local_pauli_noise_sigma))),
                ]
            ),
            qubits=q,
        )

    if readout_bit_flip_prob is not None:
        single_readout_matrix = np.array(
            [
                [1 - readout_bit_flip_prob, readout_bit_flip_prob],
                [readout_bit_flip_prob, 1 - readout_bit_flip_prob],
            ]
        )
        readout_noise = ReadoutError(single_readout_matrix)
        noise_model.add(readout_noise, gates.M)
    return noise_model


def obs_string_to_qibo_hamiltonian(
    observable: str,
    backend: Backend = None,
) -> hamiltonians.SymbolicHamiltonian:
    """
    Convert a string representation of a Pauli observable to a Qibo symbolic Hamiltonian.

    Args:
        observable (str): A string representing the Pauli observable, e.g., "XZIY".

    Returns:
        hamiltonians.SymbolicHamiltonian: The corresponding Qibo symbolic Hamiltonian.
    """

    if backend is None:
        backend = get_backend()

    form = 1
    for i, pauli in enumerate(observable):
        form *= getattr(symbols, pauli)(i)
    ham = hamiltonians.SymbolicHamiltonian(form=form, backend=backend)
    return ham
