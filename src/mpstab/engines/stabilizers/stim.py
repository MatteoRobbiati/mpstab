"""Stabilizers engine backed by stim."""

from functools import lru_cache

import stim
from qibo import Circuit

from mpstab.engines.stabilizers._stim_gates import circuit_instructions
from mpstab.engines.stabilizers.abstract import StabilizersEngine
from mpstab.pauli import conjugate


@lru_cache(maxsize=32)
def _inverse_tableau(width: int, instructions: tuple) -> stim.Tableau:
    """
    ``U^dag`` for the circuit given as stim instructions. Cached on the
    circuit's content, so in-place parameter changes can never hit a stale one.
    """
    simulator = stim.TableauSimulator()
    simulator.do(
        stim.Circuit("\n".join([f"I {width - 1}", *filter(None, instructions)]))
    )
    # current_inverse_tableau() is U^dag, so conjugating by it gives U^dag O U.
    return simulator.current_inverse_tableau()


class StimEngine(StabilizersEngine):
    """Clifford backpropagation of Pauli observables, powered by stim."""

    def backpropagate(self, observable: str, clifford_circuit: Circuit):
        """
        Evolve ``observable`` back through ``clifford_circuit`` (Heisenberg picture).

        Returns:
            ``(pauli_string, sign)`` for ``U^dag . O . U``, with ``U`` the circuit.
        """
        return self.backpropagate_many([observable], clifford_circuit)[0]

    def backpropagate_many(self, observables, clifford_circuit: Circuit) -> list:
        """
        :meth:`backpropagate` every observable through the same circuit.

        The circuit's ``U^dag`` tableau is built once and cached, so a whole
        Hamiltonian costs one stim simulation plus one tableau lookup per term,
        and repeated calls on an unchanged circuit skip the simulation too.
        """
        observables = list(observables)
        if not observables:
            return []
        # Size the tableau to at least the observables: stim sizes a circuit by
        # the highest qubit any gate touches, which may be lower.
        tableau = _inverse_tableau(
            max(map(len, observables)), tuple(self.instructions(clifford_circuit))
        )
        return [conjugate(o, tableau) for o in observables]

    def instructions(self, circuit: Circuit) -> list:
        """
        One stim instruction line per gate of ``circuit``, ``None`` for gates
        that leave the Pauli frame untouched.

        stim parses a whole program from text far faster than it appends
        instructions one call at a time, so circuits are assembled as text.

        Raises:
            ValueError: on a non-Clifford rotation angle, or a gate with no stim
                equivalent. Unsupported gates fail loudly rather than being
                dropped, which would silently give a wrong Pauli frame.
        """
        return circuit_instructions(circuit)
