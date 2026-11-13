"""
The gate-list format every head synthesiser returns, and what they share.

A synthesised head is a list of two entry shapes: a placed rotation
``(axis, [qubit], angle)`` with ``axis`` in :data:`ROTATION_GATES`, or a Clifford
gate ``(name, qubits)`` with ``name`` in :data:`CLIFFORD_GATES`.
"""

from __future__ import annotations

import stim
from qibo import Circuit, gates

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


#: The same Clifford names as ``stim`` tableaus, for tracking Pauli images.
CLIFFORD_TABLEAUS = {
    name: stim.Tableau.from_named_gate(stim_name)
    for name, stim_name in {
        "H": "H",
        "S": "S",
        "Sd": "S_DAG",
        "SqrtX": "SQRT_X",
        "SqrtXd": "SQRT_X_DAG",
        "CNOT": "CX",
        "CZ": "CZ",
        "X": "X",
        "Y": "Y",
        "Z": "Z",
    }.items()
}


_AXIS_ROTATION = {1: "RX", 2: "RY", 3: "RZ"}


def place_rotations(skeleton, paulis, angles, nqubits) -> list:
    """
    Interleave one single-qubit rotation per Pauli into a Clifford ``skeleton``.

    A rotation can be emitted once its Pauli image has collapsed onto a single
    qubit and no earlier unresolved rotation anticommutes with it. Each image is
    carried as a ``stim.PauliString`` and updated only when a skeleton gate
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

    images = [stim.PauliString(pauli) for pauli in paulis]
    resolved = [False] * n_rotations
    head: list = []

    def emit(i):
        image = images[i]
        support = [q for q in range(nqubits) if image[q] != 0]
        qubit = support[0] if support else 0
        axis = image[qubit] if support else 3
        sign = 1 if image.sign == 1 else -1
        head.append((_AXIS_ROTATION[axis], [qubit], float(angles[i]) * sign))
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
        tableau = CLIFFORD_TABLEAUS[name]
        for i in range(n_rotations):
            if resolved[i]:
                continue
            image = images[i]
            if any(image[q] != 0 for q in qubits):
                images[i] = image.after(tableau, targets=qubits)
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
