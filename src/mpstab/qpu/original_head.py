"""
The original head: the circuit's own gates up to the cut, not a resynthesis.

The synthesisers of :mod:`mpstab.qpu.synthesis` rebuild the head from its
dressed Pauli rotations alone. The original head keeps the gates of the ansatz
instead: the device runs the original circuit up to the ``cut_index``-th magic
gate, which realises ``C_prefix . R_{cut-1} ... R_0`` (``C_prefix`` being the
Clifford gates of that prefix), and the residual undoes ``C_prefix``. Its value
is that the original structure, such as nearest-neighbour couplings, often
compiles better than a resynthesis of the dressed strings.

With ``absorb`` the head is then trimmed. Going backwards, a Clifford gate with
no magic gate left in its future commutes to the end of the head: it leaves the
device and joins the residual, folded classically into the observable. Light
cone pruning of the magic gates is the evolutor's, as for the synthesised heads
(see :meth:`~mpstab.evolutors.hqsmpo.HQSMPO.relevant_rotations`).

Select it with ``HQSMPO(..., head="original")``, or build one head with
:meth:`~mpstab.evolutors.hqsmpo.HQSMPO.original_head`.
"""

from __future__ import annotations

import numpy as np
from qibo import Circuit

_HALF_PI = 0.5 * np.pi

#: Clifford rotations by number of quarter turns, as gate names of
#: :data:`~mpstab.qpu.synthesis.utils.CLIFFORD_TABLEAUS` in application order,
#: up to a global phase.
_QUARTER_TURNS = {
    "RX": {0: [], 1: ["SqrtX"], 2: ["X"], 3: ["SqrtXd"]},
    "RY": {0: [], 1: ["Z", "H"], 2: ["Y"], 3: ["H", "Z"]},
    "RZ": {0: [], 1: ["S"], 2: ["Z"], 3: ["Sd"]},
}
_FIXED = {
    "H": "H",
    "S": "S",
    "SDG": "Sd",
    "SX": "SqrtX",
    "SXDG": "SqrtXd",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "CNOT": "CNOT",
    "CZ": "CZ",
}
_INVERSE = {"S": "Sd", "Sd": "S", "SqrtX": "SqrtXd", "SqrtXd": "SqrtX"}


def clifford_gate_names(gate) -> list:
    """
    A qibo Clifford gate as ``[(name, qubits), ...]`` in the names of
    :data:`~mpstab.qpu.synthesis.utils.CLIFFORD_TABLEAUS`, up to a global phase.

    Raises:
        ValueError: on a gate that is not one of these Cliffords.
    """
    kind = type(gate).__name__
    qubits = list(gate.qubits)
    if kind in _FIXED:
        return [(_FIXED[kind], qubits)]
    if kind in _QUARTER_TURNS:
        turns = float(gate.parameters[0]) / _HALF_PI
        if np.isclose(turns, round(turns)):
            return [(name, qubits) for name in _QUARTER_TURNS[kind][round(turns) % 4]]
    raise ValueError(f"{gate!r} is not a supported Clifford gate.")


def _inverse(names: list) -> list:
    return [(_INVERSE.get(name, name), qubits) for name, qubits in reversed(names)]


def build_original_head(
    clifford_gates, magic_gates, cut_index: int, keep, nqubits: int, absorb: bool
):
    """
    ``(head_circuit, residual_gates, n_absorbed)`` of the original circuit cut
    after its ``cut_index``-th magic gate.

    Args:
        clifford_gates: the circuit's Clifford gates, in order.
        magic_gates: ``(breakpoint, gate)`` per magic gate, ``breakpoint`` being
            the number of Clifford gates before it (as
            :meth:`~mpstab.models.ansatze.Ansatz.partitionate_circuit` returns).
        cut_index: magic gates in the head.
        keep: indices of the head's magic gates to keep (the others are outside
            the observable's light cone).
        nqubits: circuit width.
        absorb: move the trailing Clifford gates into the residual.

    The residual is the absorbed gates in order, then the inverse of the
    prefix's Clifford gates, so that ``residual . head |0>`` is the kept dressed
    rotations applied to ``|0>``.
    """
    head = Circuit(nqubits)
    if cut_index == 0:
        return head, [], 0
    last = magic_gates[cut_index - 1][0]
    sequence = []  # (gate, is_magic), the prefix in circuit order
    pending = iter(enumerate(magic_gates[:cut_index]))
    current = next(pending, None)
    for index in range(last + 1):
        while current is not None and current[1][0] == index:
            if current[0] in keep:
                sequence.append((current[1][1], True))
            current = next(pending, None)
        if index < last:
            sequence.append((clifford_gates[index], False))

    absorbed = []
    if absorb:
        live, core = set(), []
        for gate, is_magic in reversed(sequence):
            if is_magic or live & set(gate.qubits):
                core.append(gate)
                live |= set(gate.qubits)
            else:
                absorbed.append(gate)
        core.reverse()
        absorbed.reverse()
    else:
        core = [gate for gate, _ in sequence]
    for gate in core:
        head.add(gate)
    head = head.copy(deep=True)  # its own gates, not the ansatz's

    residual = [entry for gate in absorbed for entry in clifford_gate_names(gate)]
    prefix = [
        entry for gate in clifford_gates[:last] for entry in clifford_gate_names(gate)
    ]
    return head, residual + _inverse(prefix), len(absorbed)
