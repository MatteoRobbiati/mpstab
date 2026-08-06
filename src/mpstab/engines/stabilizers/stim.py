"""Stabilizers engine backed by stim."""

from dataclasses import dataclass

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


@dataclass
class StimEngine(StabilizersEngine):
    """Clifford backpropagation and tableau folding, powered by stim."""

    def backpropagate(self, observable: str, clifford_circuit: Circuit):
        """
        Evolve ``observable`` back through ``clifford_circuit`` (Heisenberg picture).

        Returns:
            ``(pauli_string, sign)`` for ``U^dag . O . U``, with ``U`` the circuit.
        """
        # Pad so the tableau is at least as wide as the observable: stim sizes a
        # circuit by the highest qubit any gate touches, which may be lower.
        padded = stim.Circuit()
        padded.append("I", [len(observable) - 1])
        padded += self.to_stim(clifford_circuit)

        simulator = stim.TableauSimulator()
        simulator.do(padded)
        # current_inverse_tableau() is U^dag, so conjugating by it gives U^dag O U.
        return conjugate(observable, simulator.current_inverse_tableau())

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

    def to_stim(self, circuit: Circuit) -> stim.Circuit:
        """
        Convert a Clifford qibo circuit into a stim circuit.

        Raises:
            ValueError: on a non-Clifford rotation angle, or a gate with no stim
                equivalent. Unsupported gates fail loudly rather than being
                dropped, which would silently give a wrong Pauli frame.
        """
        stim_circuit = stim.Circuit()
        for gate in circuit.queue:
            name = gate.name.lower()

            if name in _DIRECT_GATES:
                stim_circuit.append(_DIRECT_GATES[name], gate.qubits)
            elif name in _IGNORED_GATES:
                continue
            elif name == "gpi2":
                stim_circuit.append(_gpi2_instruction(gate), gate.qubits)
            elif name in ("rx", "ry", "rz"):
                instruction = _rotation_instruction(gate, axis=name[1].upper())
                if instruction is not None:  # a zero-angle rotation is a no-op
                    stim_circuit.append(instruction, gate.qubits)
            else:
                raise ValueError(
                    f"Gate {name!r} has no stim equivalent. The Clifford part of "
                    "the circuit must use gates this engine supports: "
                    f"{sorted(_DIRECT_GATES)}, gpi2, rx, ry, rz."
                )
        return stim_circuit


def _rotation_instruction(gate, axis: str):
    """The stim instruction for an ``R{axis}`` gate, or ``None`` if it is a no-op."""
    theta = gate.parameters[0]
    if _is_multiple_of(theta, 0):
        return None
    if _is_multiple_of(theta, np.pi):
        return axis
    if _is_multiple_of(theta, np.pi / 2):
        return f"SQRT_{axis}"
    if _is_multiple_of(theta, -np.pi / 2):
        return f"SQRT_{axis}_DAG"
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
