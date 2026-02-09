from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TensorNetworkEngine(ABC):
    """Interface for Tensor Network simulation (MPS/MPO)."""

    @abstractmethod
    def create_state(
        self, nqubits: int, max_bond_dimension: int, initial_state: Any = None
    ):
        """Initialize the MPS/TN state."""
        pass

    @abstractmethod
    def apply_rotation(
        self, state: Any, generator: str, angle: float, qubits: List[int]
    ):
        """Apply exp(-i * angle/2 * generator) to the TN state."""
        pass

    @abstractmethod
    def expectation(self, state: Any, observable: str) -> float:
        """Compute expectation value <psi|O|psi>."""
        pass
