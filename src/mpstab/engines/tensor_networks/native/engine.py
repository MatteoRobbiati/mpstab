"""Pure-Python tensor-network engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from mpstab.engines.tensor_networks.abstract import TensorNetworkEngine
from mpstab.engines.tensor_networks.native.circuit_mps import CircuitMPS
from mpstab.engines.tensor_networks.native.operators.observables import PauliMPO


class NativeTensorNetworkEngine(TensorNetworkEngine):
    """
    MPS evolution and expectation values on the in-package tensor network.

    A pure-Python example of the :class:`TensorNetworkEngine` interface, written
    to be read rather than to be fast: use
    :class:`~mpstab.engines.QuimbEngine` for real workloads. Supports state-side evolution and expectation values. Operator-side
    conjugation, and so the whole head/tail split of
    :class:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO`, needs
    :class:`~mpstab.engines.QuimbEngine`.
    """

    def build_circuit_mps(
        self,
        n: int,
        initial_state_circuit: Any,
        max_bond_dimension: int | None = None,
    ):
        """
        Build a :class:`CircuitMPS` in the product state a qibo circuit of
        one-qubit gates prepares, simulating each qubit on its own.
        """
        amplitudes = []
        for qubit in range(n):
            light_circuit, _ = initial_state_circuit.light_cone(qubit)
            amplitudes.append(light_circuit().state())
        return CircuitMPS(
            n=n,
            initial_state=np.array(amplitudes),
            max_bond_dimension=max_bond_dimension,
        )

    def pauli_mpo(self, pauli_string: str | object):
        """Build the :class:`PauliMPO` for a Pauli string."""
        return PauliMPO(pauli_string)

    def expval(self, state_circuit: CircuitMPS, operator: PauliMPO):
        """Expectation value of ``operator`` on ``state_circuit``."""
        return state_circuit.expval(operator)

    def pauli_rot(
        self,
        state_circuit: CircuitMPS,
        generator: str,
        angle: float,
        max_bond_dimension: int,
    ):
        """
        Apply ``exp(-i angle/2 generator)`` to the state.

        The bond cap comes from the :class:`CircuitMPS` itself, set at
        construction, so ``max_bond_dimension`` is ignored here.
        """
        return state_circuit.pauli_rot(generator, angle)

    def squared_norm(self, state_circuit: CircuitMPS) -> float:
        """``<psi|psi>``, the expectation value of the identity."""
        identity = PauliMPO("I" * state_circuit.n_qubits)
        return float(np.real(state_circuit.expval(identity)))
