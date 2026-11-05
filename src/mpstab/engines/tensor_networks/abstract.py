"""The interface a tensor-network engine must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

#: What each method takes and returns is engine-specific: an engine may
#: represent states and operators however it likes, as long as the objects it
#: hands back are the ones it is later given.


class TensorNetworkEngine(ABC):
    """MPS state evolution and MPO expectation values."""

    @abstractmethod
    def build_circuit_mps(
        self,
        n: int,
        initial_state_amplitudes: Any,
        initial_state_circuit: Any,
        max_bond_dimension: int | None = None,
    ):
        """
        Build an MPS on ``n`` qubits in the given initial state.

        Args:
            n: number of qubits.
            initial_state_amplitudes: array of per-site single-qubit amplitudes.
            initial_state_circuit: the same state as a qibo circuit. Engines take
                whichever of the two forms they prefer.
            max_bond_dimension: truncation cap, or ``None`` for no cap.
        """

    @abstractmethod
    def pauli_mpo(self, pauli_string: str | Any):
        """Build the MPO for a Pauli string."""

    @abstractmethod
    def expval(self, state_circuit: Any, operator: Any):
        """Expectation value of ``operator`` on ``state_circuit``."""

    @abstractmethod
    def pauli_rot(
        self, state_circuit: Any, generator: str, angle: float, max_bond_dimension: int
    ):
        """Apply ``exp(-i angle/2 generator)`` to the state, in place."""

    @abstractmethod
    def conjugate_operator(
        self, operator: Any, generator: str, angle: float, max_bond_dimension: int
    ):
        """
        Heisenberg-conjugate ``operator`` by ``R = exp(-i angle/2 generator)``,
        returning ``R^dag . operator . R``.
        """
