"""
Resynthesis of a dressed-rotation chain into a runnable circuit.

A head of dressed Pauli rotations is turned into a hardware-native gate list
plus a pure-Clifford residual applied last::

    target = U(residual) . U(head)
    <target|O|target> = <head 0| residual^dag . O . residual |head 0>

The residual is therefore reabsorbed into the observable with a Clifford
simulator (:meth:`~mpstab.engines.StimEngine.fold_pauli_through_tableau`), never
with dense matrices.

Two synthesisers produce that pair:

- :func:`build_head_and_residual` uses rustiq's Pauli-network synthesis, which
  cancels gates across rotations. ``rustiq`` is optional and not on PyPI, so it
  is imported lazily; install it with ``pip install "mpstab[rustiq]"``.
- :func:`build_naive_head_and_residual` needs no extra dependency and costs
  ``2*(w-1)`` CNOTs per weight-``w`` rotation, with an identity residual.

Both return gate lists using two entry shapes: a placed rotation
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


def _import_rustiq():
    try:
        from rustiq import Metric, pauli_network_synthesis
    except ImportError as error:
        raise ImportError(
            "rustiq is required for Pauli-network resynthesis. Install it with: "
            'pip install "mpstab[rustiq]"'
        ) from error
    return Metric, pauli_network_synthesis


def build_head_and_residual(
    paulis: list[str],
    angles: list[float],
    metric_name: str = "count",
    preserve_order: bool = True,
):
    """
    Synthesise a head circuit and its pure-Clifford residual with rustiq.

    ``rustiq.pauli_network_synthesis`` called with ``fix_clifford=False`` returns
    a skeleton that is a prefix of the one it returns with ``fix_clifford=True``.
    The prefix becomes the head; what the longer version appends is the residual.

    Args:
        paulis: dressed Pauli axis strings (qubit-0-leftmost), in circuit order.
        angles: matching rotation angles.
        metric_name: rustiq metric, ``"count"`` or ``"depth"``.
        preserve_order: keep the (non-commuting) rotation order.

    Returns:
        ``(head, residual_tableau, residual_gates)``. The residual gate list is
        non-deterministic across runs (rustiq's ``fix_clifford=True`` draws
        randomly) even though the Clifford operator it implements is unique, so
        compare tableaus rather than gate lists.
    """
    Metric, pauli_network_synthesis = _import_rustiq()

    if not paulis:
        return [], stim.Tableau(0), []

    nqubits = len(paulis[0])
    metric = Metric(metric_name)
    skeleton = pauli_network_synthesis(
        paulis, metric, preserve_order, fix_clifford=False
    )
    fixed = pauli_network_synthesis(paulis, metric, preserve_order, fix_clifford=True)
    if fixed[: len(skeleton)] != skeleton:
        raise RuntimeError(
            "rustiq returned a fix_clifford=True skeleton that does not extend "
            "the fix_clifford=False one; cannot split off a Clifford residual."
        )
    residual = fixed[len(skeleton) :]

    head = _place_rotations(skeleton, paulis, angles, nqubits)

    residual_tableau = stim.Tableau(nqubits)
    for name, qubits in residual:
        residual_tableau.append(CLIFFORD_TABLEAUS[name], qubits)
    return head, residual_tableau, residual


def _place_rotations(skeleton, paulis, angles, nqubits) -> list:
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


def build_naive_head_and_residual(paulis: list[str], angles: list[float]):
    """
    Decompose every dressed rotation into an exact CNOT-ladder circuit.

    Uses the textbook identity ``exp(-i theta P / 2) = U^dag L^dag RZ(theta) L
    U``, where ``U`` rotates every non-identity site of ``P`` into the ``Z``
    frame and ``L`` is a CNOT ladder chaining the parity of the support onto its
    last qubit. Every gate is placed exactly, so nothing is left over and the
    residual is the identity -- at the cost of ``2*(w-1)`` CNOTs per weight-``w``
    rotation and no cancellation across rotations.

    Args:
        paulis: dressed Pauli axis strings (qubit-0-leftmost), in circuit order.
        angles: matching rotation angles.

    Returns:
        ``(head, residual_tableau, residual_gates)``, the same shape as
        :func:`build_head_and_residual`, with an identity tableau and an empty
        residual gate list.
    """
    if not paulis:
        return [], stim.Tableau(0), []

    nqubits = len(paulis[0])
    head: list[tuple] = []
    for pauli, angle in zip(paulis, angles):
        support = [q for q, label in enumerate(pauli) if label != "I"]
        if not support:
            continue  # exp(-i theta/2 I) is a global phase, no gate needed.

        for qubit in support:
            if pauli[qubit] == "Y":
                head.append(("Sd", [qubit]))
            if pauli[qubit] in ("X", "Y"):
                head.append(("H", [qubit]))

        ladder = list(zip(support[:-1], support[1:]))
        head.extend(("CNOT", [q0, q1]) for q0, q1 in ladder)
        head.append(("RZ", [support[-1]], float(angle)))
        head.extend(("CNOT", [q0, q1]) for q0, q1 in reversed(ladder))

        for qubit in support:
            if pauli[qubit] in ("X", "Y"):
                head.append(("H", [qubit]))
            if pauli[qubit] == "Y":
                head.append(("S", [qubit]))

    return head, stim.Tableau(nqubits), []


def head_counts_only(
    paulis: list[str], metric_name: str = "count", preserve_order: bool = True
):
    """
    Gate counts of the rustiq head without placing the rotations.

    The head is the rustiq Clifford skeleton with one single-qubit rotation
    interleaved per Pauli, so the total is ``len(skeleton) + len(paulis)`` and
    the two-qubit count is the skeleton's. One rustiq call, much cheaper than
    :func:`build_head_and_residual` and exact for counting.

    Returns:
        ``(total_gates, two_qubit_gates)``.
    """
    Metric, pauli_network_synthesis = _import_rustiq()

    if not paulis:
        return 0, 0
    skeleton = pauli_network_synthesis(
        paulis, Metric(metric_name), preserve_order, fix_clifford=False
    )
    two_qubit = sum(1 for gate in skeleton if len(gate[1]) == 2)
    return len(skeleton) + len(paulis), two_qubit


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
