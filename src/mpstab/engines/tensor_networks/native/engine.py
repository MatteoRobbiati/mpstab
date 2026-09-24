"""Pure-Python tensor-network engine."""

from __future__ import annotations

from typing import Any

from mpstab.engines.tensor_networks.abstract import TensorNetworkEngine
from mpstab.engines.tensor_networks.native.circuit_mps import CircuitMPS
from mpstab.engines.tensor_networks.native.operators.observables import PauliMPO


class NativeTensorNetworkEngine(TensorNetworkEngine):
    """
    MPS evolution and expectation values on the in-package tensor network.

    Supports state-side evolution and expectation values. Operator-side
    conjugation, and so the whole head/tail split of
    :class:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO`, needs
    :class:`~mpstab.engines.QuimbEngine`.
    """

    def build_circuit_mps(
        self,
        n: int,
        initial_state_amplitudes: Any,
        initial_state_circuit: Any,
        max_bond_dimension: int | None = None,
    ):
        """Build a :class:`CircuitMPS` from the per-site amplitudes."""
        return CircuitMPS(
            n=n,
            initial_state=initial_state_amplitudes,
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

    def conjugate_operator(
        self,
        operator: PauliMPO,
        generator: str,
        angle: float,
        max_bond_dimension: int,
    ):
        """Not supported: operator-side conjugation requires QuimbEngine."""
        raise NotImplementedError(
            "conjugate_operator is only implemented for QuimbEngine. Call "
            "set_engines(tn_engine=QuimbEngine()) to use the head/tail split."
        )
