"""Pure-Python stabilizers engine."""

import math

from qibo import gates
from qibo.models import Circuit

from mpstab.engines.stabilizers.abstract import StabilizersEngine
from mpstab.engines.stabilizers.native import tableaus
from mpstab.engines.stabilizers.native.pauli_string import Pauli

#: qibo gate names mapped to the tableau class in
#: :mod:`~mpstab.engines.stabilizers.native.tableaus` that implements them.
GATE_TABLEAUS = {
    "cx": "CNOT",
    "h": "H",
    "s": "S",
    "sdg": "Sdg",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "swap": "SWAP",
    "cz": "CZ",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "gpi2": "GPI2",
}

#: Square roots of X, as the quarter turns of ``RX`` they equal up to a phase.
_SQRT_X_ANGLES = {"sx": math.pi / 2, "sxdg": -math.pi / 2}

#: Gates that leave the Pauli frame untouched and are simply dropped.
_IGNORED_GATES = ("measure", "id", "barrier")


def _two_qubit_rotation(gate) -> list:
    """
    ``RXX``, ``RYY`` or ``RZZ`` as one-qubit Cliffords around an ``RZZ``, and
    ``RZZ(theta) = CNOT . RZ_target(theta) . CNOT``, in circuit order. The
    identities are exact, so the Pauli frame and its sign come out right; the
    ``RZ`` raises on a non-Clifford angle as usual.
    """
    a, b = gate.qubits
    theta = gate.parameters[0]
    zz = [gates.CNOT(a, b), gates.RZ(b, theta), gates.CNOT(a, b)]
    axis = gate.name.lower()[-1]
    if axis == "z":
        return zz
    if axis == "x":  # H Z H = X
        change = [gates.H(a), gates.H(b)]
        return change + zz + change
    # V Z V^dag = Y for V = S H: RYY = V RZZ V^dag, V^dag = H S^dag.
    before = [gates.SDG(a), gates.SDG(b), gates.H(a), gates.H(b)]
    after = [gates.H(a), gates.H(b), gates.S(a), gates.S(b)]
    return before + zz + after


def _expanded(queue) -> list:
    """The circuit's gates, two-qubit Pauli rotations expanded."""
    expanded = []
    for gate in queue:
        if gate.name.lower() in ("rxx", "ryy", "rzz"):
            expanded += _two_qubit_rotation(gate)
        else:
            expanded.append(gate)
    return expanded


class NativeStabilizersEngine(StabilizersEngine):
    """
    Clifford backpropagation by composing per-gate tableaus.

    A pure-Python example of the :class:`StabilizersEngine` interface, written
    to be read rather than to be fast: use :class:`~mpstab.engines.StimEngine`
    for real workloads.
    """

    def backpropagate(self, observable: str, clifford_circuit: Circuit):
        """
        Evolve ``observable`` back through ``clifford_circuit``, applying each
        gate's inverse in reverse order.

        Returns:
            ``(pauli_string, sign)`` for ``U^dag . O . U``, with ``U`` the circuit.
        """
        propagator = Pauli(observable)

        for gate in reversed(_expanded(clifford_circuit.queue)):
            if gate.name.lower() in _IGNORED_GATES:
                continue
            inverted = gate.dagger()
            name = inverted.name.lower()
            if name in _SQRT_X_ANGLES:
                propagator.apply(
                    tableaus.RX(*inverted.qubits, angle=_SQRT_X_ANGLES[name])
                )
                continue
            tableau_name = GATE_TABLEAUS.get(name)
            if tableau_name is None:
                # Skipping it would silently give a wrong Pauli frame.
                raise ValueError(
                    f"Gate {gate.name!r} is not supported by "
                    f"{type(self).__name__}: use {sorted(GATE_TABLEAUS)}, sx, "
                    "sxdg, rxx, ryy or rzz."
                )

            tableau_class = getattr(tableaus, tableau_name)
            if inverted.parameters:
                tableau = tableau_class(*inverted.qubits, angle=inverted.parameters[0])
            else:
                tableau = tableau_class(*inverted.qubits)
            propagator.apply(tableau)

        result = propagator.to_string()
        if result[0] not in "IXYZ":  # a leading sign or phase character
            return result[1:], -1 if result.startswith("-") else 1
        return result, 1.0
