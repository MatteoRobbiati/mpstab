from abc import ABC, abstractmethod
from dataclasses import dataclass

from qibo import Circuit


@dataclass
class StabilizersEngine(ABC):
    """Interface for stabilizers backpropagation logic used in mpstab."""

    @abstractmethod
    def backpropagate(self, observable: str, clifford_circuit: Circuit) -> str:
        """Evolve `observable` applying a given `clifford_circuit`."""
        pass
