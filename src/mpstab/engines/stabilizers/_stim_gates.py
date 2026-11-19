"""
qibo Clifford gates as stim instructions, and the stim tableau pass that
:class:`~mpstab.engines.stabilizers.abstract.StabilizersEngine` uses by
default to backpropagate Pauli strings through prefixes of a circuit.
"""

from typing import Sequence

import numpy as np
import stim
from qibo import Circuit

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


def circuit_instructions(circuit: Circuit) -> list:
    """One stim instruction line per gate of ``circuit``, ``None`` for no-ops."""
    return [gate_instruction(gate) for gate in circuit.queue]


def backpropagate_at(
    observables: Sequence[str], breakpoints: Sequence[int], clifford_circuit: Circuit
) -> list:
    """
    Backpropagate each observable through its own prefix of
    ``clifford_circuit`` with one stim tableau pass.

    The circuit is simulated once, a segment at a time in breakpoint order, and
    each observable is conjugated by the inverse tableau of its prefix, so the
    cost is one pass over the circuit plus one conjugation per observable.

    Raises:
        ValueError: on a breakpoint outside ``[0, len(clifford_circuit.queue)]``.
    """
    observables = list(observables)
    if not observables:
        return []
    instructions = circuit_instructions(clifford_circuit)
    width = max(clifford_circuit.nqubits, *map(len, observables))
    simulator = stim.TableauSimulator()
    simulator.do(stim.Circuit(f"I {width - 1}"))

    results = [None] * len(observables)
    simulated, inverse = 0, None
    for index in sorted(range(len(observables)), key=breakpoints.__getitem__):
        breakpoint_index = breakpoints[index]
        if not 0 <= breakpoint_index <= len(instructions):
            raise ValueError(
                f"Breakpoint {breakpoint_index} is outside the circuit's "
                f"{len(instructions)} gates."
            )
        if inverse is None or breakpoint_index > simulated:
            segment = filter(None, instructions[simulated:breakpoint_index])
            simulator.do(stim.Circuit("\n".join(segment)))
            simulated = breakpoint_index
            # current_inverse_tableau() is U^dag for the prefix U.
            inverse = simulator.current_inverse_tableau()
        results[index] = conjugate(observables[index], inverse)
    return results
