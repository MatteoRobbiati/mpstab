"""
Head synthesis with rustiq's Pauli-network synthesis, which cancels gates across
rotations.

``rustiq`` is optional and not on PyPI, so it is imported lazily; install it
with ``pip install "mpstab[rustiq]"``.
"""

from __future__ import annotations

import stim

from mpstab.qpu.synthesis.utils import CLIFFORD_TABLEAUS, place_rotations


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

    head = place_rotations(skeleton, paulis, angles, nqubits)

    residual_tableau = stim.Tableau(nqubits)
    for name, qubits in residual:
        residual_tableau.append(CLIFFORD_TABLEAUS[name], qubits)
    return head, residual_tableau, residual


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
