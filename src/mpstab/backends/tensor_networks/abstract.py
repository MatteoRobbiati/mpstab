"""Abstract API for tensor-network backends."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class TensorNetworkBackend(ABC):
    """
    Abstract interface that a tensor-network backend must implement for mpstab.

        """

    @abstractmethod
    def create_mps(self, n: int, initial_state_amplitudes: Any, initial_state_circuit: Any, max_bond_dimension: int | None = None):
        """
        Create and return an MPS-like object initialised to `initial_state`.
        - n: number of qubits
        - initial_state_amplitudes: array-like of single-site amplitudes
        - initial_state_circuit: qibo circuit to be converted to quimb in backend
        - max_bond_dimension: optional truncation parameter
        """
        raise NotImplementedError

    @abstractmethod
    def pauli_mpo(self, pauli_string: str | Any):
        """
        Return an MPO object representing the supplied Pauli string (or equivalent).
        """
        raise NotImplementedError
    
    @abstractmethod
    def expval(state: Any, operator: Any):
        """
        Compute the expectation value of `operator` on `state`.
        The types of `state` and `operator` depend on the backend's internal representations.
        """
        raise NotImplementedError

