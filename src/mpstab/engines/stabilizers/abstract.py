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

    def backpropagate_many(self, observables, clifford_circuit: Circuit) -> list:
        """
        :meth:`backpropagate` every observable through the same circuit.

        Engines that can share work across observables, such as building the
        circuit's tableau once, override this.

        Returns:
            One ``(pauli_string, sign)`` per observable, in order.
        """
        return [self.backpropagate(o, clifford_circuit) for o in observables]
