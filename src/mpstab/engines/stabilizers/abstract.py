"""The interface a stabilizers engine must implement."""

from abc import ABC, abstractmethod
from typing import Sequence

from qibo import Circuit


class StabilizersEngine(ABC):
    """
    Clifford backpropagation of Pauli observables.

    This is the only way mpstab touches the Clifford part of a circuit: the
    evolutors and the noise compiler ask the engine for backpropagated Pauli
    strings and never simulate the Clifford circuit themselves. Only
    :meth:`backpropagate` is required: the other methods have defaults built on
    it, which an engine overrides when its backend can do the same work faster.
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
        default replays each prefix through :meth:`backpropagate`, so its cost
        grows with the circuit length times the number of observables; an
        engine whose backend can sweep the circuit once, as
        :class:`~mpstab.engines.StimEngine` does with a tableau simulator,
        overrides it.

        Returns:
            One ``(pauli_string, sign)`` per observable, in order, each for
            ``U_k^dag . O . U_k`` with ``U_k`` the first ``k`` gates.

        Raises:
            ValueError: on a breakpoint outside ``[0, len(clifford_circuit.queue)]``.
        """
        results = []
        for observable, breakpoint_index in zip(observables, breakpoints):
            if not 0 <= breakpoint_index <= len(clifford_circuit.queue):
                raise ValueError(
                    f"Breakpoint {breakpoint_index} is outside the circuit's "
                    f"{len(clifford_circuit.queue)} gates."
                )
            prefix = Circuit(clifford_circuit.nqubits)
            for gate in clifford_circuit.queue[:breakpoint_index]:
                prefix.add(gate)
            results.append(self.backpropagate(observable, prefix))
        return results

    def clifford_images(self, clifford_circuit: Circuit) -> list:
        """
        ``U^dag P U`` for the generators ``X_0, ..., X_{n-1}, Z_0, ...,
        Z_{n-1}``, as ``(pauli_string, sign)``: they fix the Clifford ``U`` up
        to a global phase, so two circuits implement the same Clifford exactly
        when their images agree.
        """
        n = clifford_circuit.nqubits
        generators = [
            "I" * q + label + "I" * (n - q - 1) for label in "XZ" for q in range(n)
        ]
        return [
            (pauli, int(sign))
            for pauli, sign in self.backpropagate_many(generators, clifford_circuit)
        ]

    def is_identity(self, clifford_circuit: Circuit) -> bool:
        """Whether the Clifford circuit acts as the identity, up to a phase."""
        n = clifford_circuit.nqubits
        expected = [
            ("I" * q + label + "I" * (n - q - 1), 1) for label in "XZ" for q in range(n)
        ]
        return self.clifford_images(clifford_circuit) == expected
