"""Stabilizers engine backed by stim."""

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import stim
from qibo import Circuit

from mpstab.engines.stabilizers.abstract import StabilizersEngine
from mpstab.pauli import conjugate

#: Clifford qibo gates that map one-to-one onto a stim instruction.
_DIRECT_GATES = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "cx": "CNOT",
    "cnot": "CNOT",
    "cz": "CZ",
    "swap": "SWAP",
    "s": "S",
    "sdg": "S_DAG",
}

#: Gates that leave the Pauli frame untouched and are simply dropped.
_IGNORED_GATES = ("measure", "id", "barrier")


def _is_multiple_of(angle, target, atol: float = 1e-5) -> bool:
    """Whether ``angle`` equals ``target`` modulo ``2 pi``."""
    difference = (angle - target) % (2 * np.pi)
    return difference < atol or abs(difference - 2 * np.pi) < atol


#: How many backpropagation tableaus :class:`StimEngine` keeps.
_TABLEAU_CACHE_SIZE = 32


@dataclass
class StimEngine(StabilizersEngine):
    """Clifford backpropagation and tableau folding, powered by stim."""

    #: ``U^dag`` tableaus of recently backpropagated circuits, keyed by the
    #: circuit's content so in-place parameter changes can never hit a stale one.
    _tableaus: OrderedDict = field(
        default_factory=OrderedDict, init=False, repr=False, compare=False
    )

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
        tableau = self._inverse_tableau(
            max(map(len, observables)), self.instructions(clifford_circuit)
        )
        return [conjugate(o, tableau) for o in observables]

    def _inverse_tableau(self, width: int, instructions: list) -> stim.Tableau:
        """``U^dag`` for the circuit given as :meth:`instructions`, cached."""
        key = (width, tuple(instructions))
        tableau = self._tableaus.get(key)
        if tableau is not None:
            self._tableaus.move_to_end(key)
            return tableau
        simulator = stim.TableauSimulator()
        simulator.do(
            stim.Circuit("\n".join([f"I {width - 1}", *filter(None, instructions)]))
        )
        # current_inverse_tableau() is U^dag, so conjugating by it gives U^dag O U.
        tableau = simulator.current_inverse_tableau()
        self._tableaus[key] = tableau
        if len(self._tableaus) > _TABLEAU_CACHE_SIZE:
            self._tableaus.popitem(last=False)
        return tableau

    def fold_pauli_through_tableau(
        self, pauli_str: str, tableau: stim.Tableau, sign: float = 1.0
    ):
        """
        Reabsorb a Clifford ``tableau`` applied *after* the state into an observable.

        Since ``<psi| U^dag . P . U |psi> = <psi| M |psi>`` with
        ``M = U^dag . P . U``, folding the Clifford residual of a resynthesis into
        the observable is exact and involves no tensor-network truncation.

        Args:
            pauli_str: Pauli string (qubit-0-leftmost), e.g. ``"XZIZ"``.
            tableau: the Clifford to reabsorb.
            sign: prefactor carried on the input Pauli.

        Returns:
            ``(pauli_string, sign)`` for the folded observable.
        """
        return conjugate(pauli_str, tableau.inverse(), sign)

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
        return [gate_instruction(gate) for gate in circuit.queue]


def gate_instruction(gate):
    """
    The stim instruction line for one Clifford qibo gate, or ``None`` if it is
    a no-op on the Pauli frame.

    Raises:
        ValueError: on a non-Clifford rotation angle, or a gate with no stim
            equivalent.
    """
    name = gate.name.lower()
    if name in _DIRECT_GATES:
        instruction = _DIRECT_GATES[name]
    elif name in _IGNORED_GATES:
        return None
    elif name == "gpi2":
        instruction = _gpi2_instruction(gate)
    elif name in _ROTATION_GATES:
        instruction = _rotation_instruction(gate, axis=name[-1].upper())
        if instruction is None:  # a zero-angle rotation is a no-op
            return None
    else:
        raise ValueError(
            f"Gate {name!r} has no stim equivalent. The Clifford part of "
            "the circuit must use gates this engine supports: "
            f"{sorted(_DIRECT_GATES)}, gpi2, {', '.join(_ROTATION_GATES)}."
        )
    return " ".join([instruction, *map(str, gate.qubits)])


#: Pauli rotations ``exp(-i theta/2 P...P)``, Clifford at multiples of pi/2.
_ROTATION_GATES = ("rx", "ry", "rz", "rxx", "ryy", "rzz")


def _rotation_instruction(gate, axis: str):
    """
    The stim instruction for an ``R{axis}`` or ``R{axis}{axis}`` gate, or
    ``None`` if it is a no-op.

    At ``pi`` the rotation is the Pauli ``axis`` on every target qubit, up to a
    phase, so the single-qubit instruction is broadcast over them. At ``pi/2``
    it is stim's ``SQRT_{axis}`` or ``SQRT_{axis}{axis}``, again up to a phase.
    """
    theta = gate.parameters[0]
    sqrt = "SQRT_" + axis * len(gate.qubits)
    if _is_multiple_of(theta, 0):
        return None
    if _is_multiple_of(theta, np.pi):
        return axis
    if _is_multiple_of(theta, np.pi / 2):
        return sqrt
    if _is_multiple_of(theta, -np.pi / 2):
        return f"{sqrt}_DAG"
    raise ValueError(f"Gate {gate} is not Clifford.")


def _gpi2_instruction(gate) -> str:
    """
    The stim instruction for a ``GPI2(phi)`` gate.

    ``GPI2(phi)`` is a pi/2 rotation about the axis ``(cos phi, sin phi, 0)``, so
    it is Clifford only at multiples of pi/2. The mapping below was verified by
    comparing ``U P U^dag`` against stim's named-gate tableaus.
    """
    phi = gate.parameters[0]
    for target, instruction in (
        (0, "SQRT_X"),
        (np.pi / 2, "SQRT_Y"),
        (np.pi, "SQRT_X_DAG"),
        (-np.pi / 2, "SQRT_Y_DAG"),
    ):
        if _is_multiple_of(phi, target):
            return instruction
    raise ValueError(f"Gate {gate} is not Clifford.")
