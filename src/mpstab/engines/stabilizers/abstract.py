"""The interface a stabilizers engine must implement."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from qibo import Circuit


@dataclass
class StabilizersEngine(ABC):
    """Clifford backpropagation of Pauli observables."""

    @abstractmethod
    def backpropagate(self, observable: str, clifford_circuit: Circuit):
        """
        Evolve ``observable`` back through ``clifford_circuit``.

        Returns:
            ``(pauli_string, sign)`` for ``U^dag . O . U``, with ``U`` the circuit.
        """
