from __future__ import annotations
from typing import Any

from mpstab.backends.tensor_networks.abstract import TensorNetworkEngine
from mpstab.evolutors.tensor_network.operators.observables import PauliMPO
from quimb.tensor import CircuitMPS


class QuimbEngine(TensorNetworkEngine):
    """
    Thin adapter that exposes the minimal API required by HybridSurrogate
    while reusing the existing evolutors.tensor_network implementation.
    """

    def build_circuit_mps(self, n: int, initial_state_amplitudes: Any, initial_state_circuit: Any, max_bond_dimension: int | None = None):
        raise NotImplementedError("Circuit MPS construction is not implemented in the QuimbEngine yet.")
    def pauli_mpo(self, pauli_string: str | object):
        raise NotImplementedError("Pauli MPO construction is not implemented in the QuimbEngine yet.")
    
    def expval(self, state: CircuitMPS, operator: PauliMPO):
        raise NotImplementedError("Expectation value computation is not implemented in the QuimbEngine yet.") 
    def pauli_rot(self, state: CircuitMPS, generator: str, angle: float):
        raise NotImplementedError("Pauli rotations are not implemented in the QuimbEngine yet.")