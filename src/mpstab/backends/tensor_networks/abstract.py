"""Abstract API for tensor-network engines."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class TensorNetworkEngine(ABC):
    """
    Abstract interface that a tensor-network engine must implement for mpstab.

        """

    @abstractmethod
    def build_circuit_mps(self, n: int, initial_state_amplitudes: Any, initial_state_circuit: Any, max_bond_dimension: int | None = None):
        """
        Create and return an MPS-like object initialised to `initial_state`.
        - n: number of qubits
        - initial_state_amplitudes: array-like of single-site amplitudes
        - initial_state_circuit: qibo circuit to be converted to quimb in engine
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
    def expval(self,state_circuit: Any, operator: Any):
        """
        Compute the expectation value of `operator` on `state_circuit`.
        The types of `state_circuit` and `operator` depend on the engine's internal representations.
        """
        raise NotImplementedError

    @abstractmethod
    def pauli_rot(self, state_circuit: Any, generator: str, angle: float):
        """
        Apply a Pauli rotation specified by `generator` and `angle` to the MPS.
        """
        raise NotImplementedError