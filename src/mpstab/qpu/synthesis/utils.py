"""
The gate-list format every head synthesiser returns, and what they share.

A synthesised head is a list of two entry shapes: a placed rotation
``(axis, [qubit], angle)`` with ``axis`` in :data:`ROTATION_GATES`, or a Clifford
gate ``(name, qubits)`` with ``name`` in :data:`CLIFFORD_GATES`.
"""

from __future__ import annotations

import inspect

import numpy as np
from qibo import Circuit, gates
from qibo.backends import NumpyBackend

from mpstab.pauli import PAULI_LABELS, pauli_matrix

#: Clifford gate names appearing in a synthesised gate list, and their qibo gates.
CLIFFORD_GATES = {
    "H": gates.H,
    "S": gates.S,
    "Sd": gates.SDG,
    "SqrtX": gates.SX,
    "SqrtXd": gates.SXDG,
    "X": gates.X,
    "Y": gates.Y,
    "Z": gates.Z,
    "CNOT": gates.CNOT,
    "CZ": gates.CZ,
}


#: Rotation axis names appearing in a synthesised gate list, and their qibo gates.
ROTATION_GATES = {"RX": gates.RX, "RY": gates.RY, "RZ": gates.RZ}


#: Gate matrices are read on qibo's numpy backend, whatever the global one.
_QIBO_NUMPY = NumpyBackend()


def _conjugation_table(gate_class) -> dict:
    """
    ``U P U^dag`` for every Pauli ``P`` on the gate's qubits, as
    ``{local_label: (image_label, sign)}``, read off the gate's qibo matrix.
    The gate's width is the number of qubits its constructor requires.
    """
    width = sum(
        parameter.default is parameter.empty
        and parameter.kind is parameter.POSITIONAL_OR_KEYWORD
        for parameter in inspect.signature(gate_class).parameters.values()
    )
    unitary = np.asarray(gate_class(*range(width)).matrix(_QIBO_NUMPY))
    if unitary.shape != (2**width, 2**width):
        raise ValueError(
            f"{gate_class.__name__} takes {width} qubits but its matrix is "
            f"{unitary.shape}: give it a conjugation table by hand."
        )
    labels = [""]
    for _ in range(width):
        labels = [label + letter for label in labels for letter in PAULI_LABELS]
    table = {}
    for label in labels:
        image = unitary @ pauli_matrix(label) @ unitary.conj().T
        for candidate in labels:
            overlap = np.trace(pauli_matrix(candidate) @ image).real / 2**width
            if abs(overlap) > 0.5:
                table[label] = (candidate, 1 if overlap > 0 else -1)
                break
    return table


#: How each Clifford gate of :data:`CLIFFORD_GATES` maps the Paulis on its
#: qubits under conjugation ``P -> U P U^dag``: ``{name: {local: (image, sign)}}``.
CLIFFORD_CONJUGATION = {
    name: _conjugation_table(gate) for name, gate in CLIFFORD_GATES.items()
}


class PauliImage:
    """
    A signed Pauli string carried forward through Clifford gates,
    ``P -> U P U^dag``, one gate at a time: each update touches only the
    gate's qubits.
    """

    __slots__ = ("labels", "sign", "weight")

    def __init__(self, pauli: str, sign: int = 1):
        self.labels = list(pauli)
        self.sign = sign
        self.weight = sum(label != "I" for label in pauli)

    def after(self, name: str, qubits) -> None:
        """Conjugate by the Clifford gate ``name`` on ``qubits``, in place."""
        local = "".join(self.labels[q] for q in qubits)
        image, sign = CLIFFORD_CONJUGATION[name][local]
        for qubit, before, label in zip(qubits, local, image):
            self.weight += (label != "I") - (before != "I")
            self.labels[qubit] = label
        self.sign *= sign

    def __str__(self) -> str:
        return "".join(self.labels)


_AXIS_ROTATION = {"X": "RX", "Y": "RY", "Z": "RZ"}


def place_rotations(skeleton, paulis, angles, nqubits) -> list:
    """
    Interleave one single-qubit rotation per Pauli into a Clifford ``skeleton``.

    A rotation can be emitted once its Pauli image has collapsed onto a single
    qubit and no earlier unresolved rotation anticommutes with it. Each image is
    carried as a :class:`PauliImage` and updated only when a skeleton gate
    touches its support; the anticommutation graph is precomputed as bitmasks so
    the blocking test is a counter rather than a scan.
    """
    n_rotations = len(paulis)
    x_mask = [0] * n_rotations
    z_mask = [0] * n_rotations
    for i, pauli in enumerate(paulis):
        for qubit, label in enumerate(pauli):
            if label in ("X", "Y"):
                x_mask[i] |= 1 << qubit
            if label in ("Z", "Y"):
                z_mask[i] |= 1 << qubit

    blockers = [0] * n_rotations
    blocks = [[] for _ in range(n_rotations)]
    for i in range(n_rotations):
        for j in range(i):
            anticommutes = (
                (x_mask[i] & z_mask[j]) ^ (z_mask[i] & x_mask[j])
            ).bit_count() & 1
            if anticommutes:
                blockers[i] += 1
                blocks[j].append(i)

    images = [PauliImage(pauli) for pauli in paulis]
    resolved = [False] * n_rotations
    head: list = []

    def emit(i):
        image = images[i]
        support = [q for q in range(nqubits) if image.labels[q] != "I"]
        qubit = support[0] if support else 0
        axis = image.labels[qubit] if support else "Z"
        head.append((_AXIS_ROTATION[axis], [qubit], float(angles[i]) * image.sign))
        resolved[i] = True
        for later in blocks[i]:
            blockers[later] -= 1

    def drain():
        progress = True
        while progress:
            progress = False
            for i in range(n_rotations):
                if not resolved[i] and blockers[i] == 0 and images[i].weight <= 1:
                    emit(i)
                    progress = True

    drain()
    for name, qubits in skeleton:
        for i in range(n_rotations):
            if resolved[i]:
                continue
            image = images[i]
            if any(image.labels[q] != "I" for q in qubits):
                image.after(name, qubits)
        head.append((name, qubits))
        drain()

    if not all(resolved):
        unplaced = [i for i, done in enumerate(resolved) if not done]
        raise RuntimeError(f"failed to place rotations at indices {unplaced}")
    return head


def head_to_qibo_circuit(head: list[tuple], nqubits: int) -> Circuit:
    """Convert a synthesised gate list into a runnable qibo ``Circuit``."""
    circuit = Circuit(nqubits)
    for entry in head:
        if len(entry) == 3:
            axis, qubits, angle = entry
            circuit.add(ROTATION_GATES[axis](qubits[0], theta=angle))
        else:
            name, qubits = entry
            circuit.add(CLIFFORD_GATES[name](*qubits))
    return circuit


def count_two_qubit_gates(head: list[tuple]) -> int:
    """Number of two-qubit gates in a synthesised gate list."""
    return sum(1 for entry in head if len(entry) == 2 and len(entry[1]) == 2)
