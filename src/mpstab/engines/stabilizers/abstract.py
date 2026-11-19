"""The interface a stabilizers engine must implement."""

from abc import ABC, abstractmethod
from typing import Sequence

from qibo import Circuit

from mpstab.engines.stabilizers import _stim_gates


class StabilizersEngine(ABC):
    """
    Clifford backpropagation of Pauli observables.

    This is the only way mpstab touches the Clifford part of a circuit: the
    evolutors and the noise compiler ask the engine for backpropagated Pauli
    strings and never simulate the Clifford circuit themselves. Methods with a
    default fall back on stim; an engine with its own Clifford backend
    overrides them to use it.
    """

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

    def backpropagate_at(
        self,
        observables: Sequence[str],
        breakpoints: Sequence[int],
        clifford_circuit: Circuit,
    ) -> list:
        """
        Backpropagate each observable through its own prefix of
        ``clifford_circuit``: ``observables[i]`` through the first
        ``breakpoints[i]`` gates.

        This is what dresses the magic gates of a circuit, and the noise events
        between them, with the Clifford gates that precede each one. The
        default runs one stim tableau pass over the circuit, a segment at a time
        in breakpoint order; an engine with its own Clifford backend overrides
        it.

        Returns:
            One ``(pauli_string, sign)`` per observable, in order, each for
            ``U_k^dag . O . U_k`` with ``U_k`` the first ``k`` gates.
        """
        return _stim_gates.backpropagate_at(observables, breakpoints, clifford_circuit)
