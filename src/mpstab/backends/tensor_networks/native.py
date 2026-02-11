from __future__ import annotations
from typing import Any

from mpstab.backends.tensor_networks.abstract import TensorNetworkEngine
from mpstab.evolutors.tensor_network.circuit_mps import CircuitMPS
from mpstab.evolutors.tensor_network.operators.observables import PauliMPO


class NativeTensorNetworkEngine(TensorNetworkEngine):
    """
    Thin adapter that exposes the minimal API required by HybridSurrogate
    while reusing the existing evolutors.tensor_network implementation.
    """

    def build_circuit_mps(self, n: int, initial_state_amplitudes: Any, initial_state_circuit: Any, max_bond_dimension: int | None = None):
        """
        Create a CircuitMPS initialised with `initial_state`.
        `initial_state` is passed as-is to CircuitMPS (the caller creates the
        array of single-qubit amplitudes as before).
        """
        return CircuitMPS(n=n, initial_state=initial_state_amplitudes, max_bond_dimension=max_bond_dimension)

    def pauli_mpo(self, pauli_string: str | object):
        """Return the existing PauliMPO for the given pauli_string."""
        return PauliMPO(pauli_string)
    
    def expval(self, state_circuit: CircuitMPS, operator: PauliMPO):
        """Compute the expectation value of `operator` on `state`."""
        return state_circuit.expval(operator)
    
    def pauli_rot(self, state_circuit: CircuitMPS, generator: str, angle: float):
        """Apply a Pauli rotation specified by `generator` and `angle` to the MPS."""
        return state_circuit.pauli_rot(generator, angle)
        