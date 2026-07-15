"""Foldable resynthesis via the low-level rustiq Pauli-network API.

Unlike Qiskit's ``rustiq`` high-level-synthesis plugin (which resynthesizes
the exact and up-to-Clifford circuits independently, leaving a *non*-Clifford
residual), the low-level ``rustiq.pauli_network_synthesis`` has a clean
structure that this module exploits directly::

    skel_uc = pauli_network_synthesis(paulis, metric, order, fix_clifford=False)
    skel_ex = pauli_network_synthesis(paulis, metric, order, fix_clifford=True)
    assert skel_ex[:len(skel_uc)] == skel_uc          # uc is a prefix of ex
    tail = skel_ex[len(skel_uc):]                       # pure Clifford

The "head" circuit (``skel_uc`` with the Pauli rotations placed) is what you
run on hardware; the pure-Clifford ``tail`` is reabsorbed into the observable
with a Clifford simulator (``stim``, via :class:`~mpstab.engines.StimEngine`)::

    target = U(tail) @ U(head)                          (tail applied last)
    <target|O|target> = <head 0| tail^dag O tail | head 0>

so ``M = tail^dag O tail`` is another Pauli, computed via stim -- no dense
matrices, fully scalable.

This module isolates the optional ``rustiq`` dependency (imported lazily
inside the functions that need it, mirroring how
:mod:`mpstab.evolutors.optimization` isolates its optional ``quimb.tensor``
DMRG dependency). Install it with the ``rustiq`` extra:
``pip install "mpstab[rustiq]"``.
"""
from __future__ import annotations

from typing import List, Tuple

import stim
from qibo import Circuit, gates

from mpstab.engines import StimEngine

_STIM = {
    "H": "H", "S": "S", "Sd": "S_DAG", "SqrtX": "SQRT_X", "SqrtXd": "SQRT_X_DAG",
    "CNOT": "CX", "CZ": "CZ", "X": "X", "Z": "Z", "Y": "Y",
}
_AXIS_ROT = {1: "RX", 2: "RY", 3: "RZ"}
_GATE_TAB = {k: stim.Tableau.from_named_gate(v) for k, v in _STIM.items()}

_CLIFFORD_1Q_TO_QIBO = {
    "H": gates.H, "S": gates.S, "Sd": gates.SDG,
    "SqrtX": gates.SX, "SqrtXd": gates.SXDG,
    "X": gates.X, "Y": gates.Y, "Z": gates.Z,
}
_CLIFFORD_2Q_TO_QIBO = {"CNOT": gates.CNOT, "CZ": gates.CZ}
_ROTATION_TO_QIBO = {"RX": gates.RX, "RY": gates.RY, "RZ": gates.RZ}


def _import_rustiq():
    try:
        from rustiq import Metric, pauli_network_synthesis
    except ImportError as e:
        raise ImportError(
            "rustiq is required for foldable low-level-rustiq resynthesis. "
            'Install it with: pip install "mpstab[rustiq]"'
        ) from e
    return Metric, pauli_network_synthesis


def build_head_and_residual(
    paulis: List[str],
    angles: List[float],
    metric_name: str = "count",
    preserve_order: bool = True,
):
    """
    Synthesize the foldable "head" circuit and its pure-Clifford residual tail.

    Args:
        paulis: dressed Pauli axis strings (qubit-0-leftmost), circuit order.
        angles: matching rotation angles.
        metric_name: rustiq metric, "count" or "depth".
        preserve_order: keep the (non-commuting) rotation order.

    Returns:
        (head, tail_tableau, tail_gates):
          - head: gate list -- ("RX"/"RY"/"RZ", [q], theta) and Clifford gates
            ("H"/"S"/"Sd"/"SqrtX"/"SqrtXd"/"CNOT"/"CZ"/"X"/"Y"/"Z", [qubits]).
          - tail_tableau: stim.Tableau of the residual Clifford (apply last).
          - tail_gates: the residual as a gate list (pure Clifford). Note this
            gate list is non-deterministic (rustiq's ``fix_clifford=True`` RNG)
            even though ``tail_tableau`` -- the Clifford operator it implements
            -- is unique; compare tableaus, not gate lists, across runs.
    """
    Metric, pauli_network_synthesis = _import_rustiq()

    if not paulis:
        return [], stim.Tableau(0), []

    n = len(paulis[0])
    metric = Metric(metric_name)
    skel_uc = pauli_network_synthesis(paulis, metric, preserve_order, fix_clifford=False)
    skel_ex = pauli_network_synthesis(paulis, metric, preserve_order, fix_clifford=True)
    if skel_ex[: len(skel_uc)] != skel_uc:
        raise RuntimeError("rustiq prefix property violated (uc not a prefix of ex)")
    tail = skel_ex[len(skel_uc) :]

    # ---- rotation placement (optimized) ----
    # Each pauli's image is kept as a stim.PauliString and updated with
    # .after(gate_tableau, targets) only when the gate touches its support --
    # no re-conjugation from scratch after every skeleton gate. A bitmask
    # anticommutation graph gives an O(1) `blockers` counter (# earlier
    # unresolved anticommuting paulis) instead of an O(n)-per-check inner loop.
    # Emission order is ascending index, drain-to-fixpoint.
    m = len(paulis)
    xm = [0] * m
    zm = [0] * m
    for i, p in enumerate(paulis):
        for q, c in enumerate(p):
            if c == "X" or c == "Y":
                xm[i] |= 1 << q
            if c == "Z" or c == "Y":
                zm[i] |= 1 << q
    blockers = [0] * m
    later_dep = [[] for _ in range(m)]
    for i in range(m):
        xi, zi = xm[i], zm[i]
        for j in range(i):
            if ((xi & zm[j]) ^ (zi & xm[j])).bit_count() & 1:
                blockers[i] += 1
                later_dep[j].append(i)

    images = [stim.PauliString(p) for p in paulis]
    resolved = [False] * m
    head = []

    def _emit(i):
        img = images[i]
        support = [q for q in range(n) if img[q] != 0]
        q = support[0] if support else 0
        axis = img[q] if support else 3
        sgn = 1 if img.sign == 1 else -1
        head.append((_AXIS_ROT[axis], [q], float(angles[i]) * sgn))
        resolved[i] = True
        for k in later_dep[i]:
            blockers[k] -= 1

    def _drain():
        progress = True
        while progress:
            progress = False
            for i in range(m):
                if not resolved[i] and blockers[i] == 0 and images[i].weight <= 1:
                    _emit(i)
                    progress = True

    _drain()
    for name, qs in skel_uc:
        tab = _GATE_TAB[name]
        for i in range(m):
            if resolved[i]:
                continue
            im = images[i]
            if any(im[q] != 0 for q in qs):
                images[i] = im.after(tab, targets=qs)
        head.append((name, qs))
        _drain()

    if not all(resolved):
        raise RuntimeError(f"failed to place all rotations: {resolved}")

    tail_tab = stim.Tableau(n)
    for name, qs in tail:
        tail_tab.append(_GATE_TAB[name], qs)
    return head, tail_tab, tail


def head_counts_only(
    paulis: List[str], metric_name: str = "count", preserve_order: bool = True
):
    """
    Gate counts of the foldable head WITHOUT placing rotations (no placement
    loop, no ``fix_clifford=True`` call). The head is the rustiq Clifford
    skeleton with one single-qubit rotation per pauli interleaved, so::

        head 2-qubit gates = 2-qubit gates in the skeleton   (rotations are 1Q)
        head total gates   = len(skeleton) + len(paulis)

    A single rustiq call; much cheaper than :func:`build_head_and_residual` and
    exact for counting purposes. Returns ``(total, two_qubit)``.
    """
    Metric, pauli_network_synthesis = _import_rustiq()

    if not paulis:
        return 0, 0
    skel = pauli_network_synthesis(
        paulis, Metric(metric_name), preserve_order, fix_clifford=False
    )
    two_q = sum(1 for g in skel if len(g[1]) == 2)
    return len(skel) + len(paulis), two_q


def fold_observable(tail_tableau: "stim.Tableau", pauli_str: str, sign: float = 1.0):
    """
    Reabsorb the residual Clifford into an observable: returns
    ``M = tail^dag O tail`` as a (signed) Pauli, so that
    ``<target|O|target> = <head 0| M | head 0>``.

    Thin wrapper around :meth:`mpstab.engines.StimEngine.fold_pauli_through_tableau`
    -- the single canonical implementation of the tableau-folding math.

    Args:
        tail_tableau: residual Clifford tableau from :func:`build_head_and_residual`.
        pauli_str: observable Pauli string (qubit-0-leftmost), e.g. "XZIZ".
        sign: +/-1 prefactor on the observable.

    Returns:
        (pauli_str, sign): the folded observable (qubit-0-leftmost).
    """
    return StimEngine().fold_pauli_through_tableau(pauli_str, tail_tableau, sign)


def head_to_qibo_circuit(head: List[tuple], nqubits: int) -> Circuit:
    """
    Convert a resynthesized ``head`` (or pure-Clifford ``tail_gates``) gate
    list -- as returned by :func:`build_head_and_residual` -- into a runnable
    qibo :class:`~qibo.models.circuit.Circuit`.

    Each entry is either a placed rotation ``(axis, [q], theta)`` with
    ``axis`` in ``{"RX", "RY", "RZ"}``, or a Clifford gate ``(name, qubits)``
    with ``name`` in ``{"H", "S", "Sd", "SqrtX", "SqrtXd", "CNOT", "CZ", "X",
    "Y", "Z"}``.
    """
    circuit = Circuit(nqubits)
    for entry in head:
        if len(entry) == 3:
            axis, qubits, angle = entry
            circuit.add(_ROTATION_TO_QIBO[axis](qubits[0], theta=angle))
        else:
            name, qubits = entry
            if name in _CLIFFORD_2Q_TO_QIBO:
                circuit.add(_CLIFFORD_2Q_TO_QIBO[name](*qubits))
            else:
                circuit.add(_CLIFFORD_1Q_TO_QIBO[name](qubits[0]))
    return circuit
