from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy import cos, sin
from qibo.gates.abstract import ParametrizedGate
from quimb.gates import I, X, Y, Z
from quimb.tensor import (
    CircuitMPS,
    MatrixProductOperator,
    MatrixProductState,
    MPO_identity,
    MPO_product_operator,
    TensorNetwork,
)

from mpstab.engines.tensor_networks.abstract import TensorNetworkEngine

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

pauli_map = {"X": X, "Y": Y, "Z": Z, "I": I}


def _qibo_circuit_to_quimb(nqubits, qibo_circ, **circuit_kwargs):
    circ = CircuitMPS(nqubits, **circuit_kwargs)
    for gate in qibo_circ.queue:
        gate_name = getattr(gate, "name", None)
        quimb_gate_name = GATE_MAP.get(gate_name, None)
        if quimb_gate_name == "measure" or quimb_gate_name is None:
            continue
        params = getattr(gate, "parameters", ())
        qubits = getattr(gate, "qubits", ())
        is_parametrized = isinstance(gate, ParametrizedGate) and getattr(
            gate, "trainable", True
        )
        circ.apply_gate(quimb_gate_name, *params, *qubits, parametrized=is_parametrized)
    return circ


def PauliExp(pauli_string, theta):
    L = len(pauli_string)
    pauli_matrices = [pauli_map[s.upper()] for s in pauli_string]
    id_mpo = MPO_identity(L, phys_dim=2, dtype="complex128")
    pauli_mpo = MPO_product_operator(pauli_matrices)
    return (cos(theta / 2) * id_mpo).add_MPO(-1j * sin(theta / 2) * pauli_mpo)


class QuimbEngine(TensorNetworkEngine):
    def __init__(self):
        self._path_cache = {}

    def build_circuit_mps(
        self,
        n,
        initial_state_amplitudes,
        initial_state_circuit,
        max_bond_dimension=None,
    ):
        if initial_state_circuit is not None:
            return _qibo_circuit_to_quimb(
                nqubits=n, qibo_circ=initial_state_circuit, max_bond=max_bond_dimension
            ).psi
        raise NotImplementedError("Initial state circuit required.")

    @lru_cache(maxsize=1024)
    def pauli_mpo(self, pauli_string: str):
        pauli_matrices = [pauli_map[s.upper()] for s in pauli_string]
        mpo = MPO_product_operator(pauli_matrices)
        mpo.add_tag("MPO")
        return mpo

    def expval(
        self, state_circuit: MatrixProductState, operator: MatrixProductOperator
    ):
        path_key = (state_circuit.L, state_circuit.max_bond)

        # Reindex bra to match operator/ket indices
        state_dag = state_circuit.reindex(
            {f"k{i}": f"b{i}" for i in range(state_circuit.L)}
        )

        # Convert to generic TensorNetwork to fix the 'list' attribute error
        tn = TensorNetwork((state_dag.H, operator, state_circuit))

        if path_key not in self._path_cache:
            self._path_cache[path_key] = tn.contract(optimize="auto-hq", get="path")

        return tn.contract(optimize=self._path_cache[path_key]).real

    def pauli_rot(
        self,
        state_circuit: MatrixProductState,
        generator: str,
        angle: float,
        max_bond_dimension: int,
    ):
        rotation_mpo = PauliExp(generator, angle)
        state_circuit.gate_with_mpo(
            rotation_mpo, inplace=True, max_bond=max_bond_dimension
        )
        return state_circuit
