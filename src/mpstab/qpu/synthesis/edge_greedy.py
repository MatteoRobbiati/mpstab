"""
Head synthesis one coupler gate at a time, judged on every rotation that can go
next: rustiq's idea, restricted to the device couplers.

Rustiq builds the skeleton gate by gate, choosing each Clifford for its effect
on all the pending strings, but on an all-to-all device. Extraction
(:mod:`~mpstab.qpu.synthesis.extraction`) reduces one rotation at a time, with
CNOTs inside its support. Here every step is a CNOT **on a coupler** of the
device, preceded by a local basis change on its two qubits, chosen on all the
rotations of the front at once, with a discounted look-ahead on the later ones.
A string's cost is the size of the Steiner tree joining its support on the
device: the CNOTs it still needs, with no routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import stim

from mpstab.qpu.synthesis.abstract import HeadSynthesis
from mpstab.qpu.synthesis.coupling import (
    CouplingGraph,
    PauliFrame,
    hashable_connectivity,
    head_and_residual_from_skeleton,
)
from mpstab.qpu.synthesis.steiner import steiner_reduce
from mpstab.qpu.synthesis.utils import CLIFFORD_TABLEAUS

#: Local basis changes tried on each qubit of a CNOT: single-qubit gates need no routing.
_BASES = (None, "H", "SqrtX")


def _candidate_table() -> dict:
    """For each pair of local bases, the map (a, b) -> (a', b') of the two
    Paulis (stim codes) on control and target under basis change then CNOT."""
    table = {}
    for basis_c in _BASES:
        for basis_t in _BASES:
            tableau = stim.Tableau(2)
            if basis_c:
                tableau.append(CLIFFORD_TABLEAUS[basis_c], [0])
            if basis_t:
                tableau.append(CLIFFORD_TABLEAUS[basis_t], [1])
            tableau.append(CLIFFORD_TABLEAUS["CNOT"], [0, 1])
            mapping = {}
            for a in range(4):
                for b in range(4):
                    pauli = stim.PauliString(2)
                    pauli[0], pauli[1] = a, b
                    out = pauli.after(tableau, targets=[0, 1])
                    mapping[(a, b)] = (out[0], out[1])
            table[(basis_c, basis_t)] = mapping
    return table


_TABLE = _candidate_table()


def edge_greedy_skeleton(
    paulis, graph: CouplingGraph, lookahead: int = 10, gamma: float = 0.7, lam: float = 0.5
) -> list:
    """
    The Clifford skeleton, one coupler gate at a time.

    Each step tries every coupler in both orientations, with each of the nine
    local basis changes on its two qubits, and keeps the gate that most lowers

    .. math::
        \\sum_{P \\in \\text{front}} T(P) + \\lambda \\sum_{k=1}^{K} \\gamma^k T(P_k),

    ``T`` being the Steiner-tree size of a string's support on the device, the
    front the rotations that can be placed now, and ``P_k`` the next
    ``lookahead`` = ``K`` pending ones. Ties go to fewer basis gates. When no
    gate lowers it (a plateau) one Steiner step reduces the cheapest front
    rotation (:func:`~mpstab.qpu.synthesis.steiner.steiner_reduce`), so the
    loop always terminates.
    """
    frame = PauliFrame(paulis)
    nqubits = len(paulis[0])
    edges = graph.directed_edges
    while not frame.done:
        weights, rank = {}, 0
        for i, done in enumerate(frame.resolved):
            if done:
                continue
            if frame.blockers[i] == 0:
                weights[i] = 1.0
            elif rank < lookahead:
                rank += 1
                weights[i] = lam * gamma**rank
        front = [i for i, w in weights.items() if w == 1.0]
        images = {i: [frame.images[i][q] for q in range(nqubits)] for i in weights}
        cost = {i: graph.tree_cost([q for q, c in enumerate(img) if c]) for i, img in images.items()}

        best, best_score = None, -1e-9
        for control, target in edges:
            touched = [i for i in weights if images[i][control] or images[i][target]]
            if not any(weights[i] == 1.0 for i in touched):
                continue
            for basis, mapping in _TABLE.items():
                delta = 0.0
                for i in touched:
                    pair = (images[i][control], images[i][target])
                    new = mapping[pair]
                    if new == pair:
                        continue
                    image = images[i][:]
                    image[control], image[target] = new
                    delta += weights[i] * (graph.tree_cost([q for q, c in enumerate(image) if c]) - cost[i])
                score = delta + 1e-3 * ((basis[0] is not None) + (basis[1] is not None))
                if score < best_score:
                    best, best_score = (control, target, basis), score
        if best is None:
            steiner_reduce(frame, min(front, key=lambda j: (cost[j], j)), graph)
            continue
        control, target, (basis_c, basis_t) = best
        if basis_c:
            frame.apply(basis_c, [control])
        if basis_t:
            frame.apply(basis_t, [target])
        frame.apply("CNOT", [control, target])
    return frame.skeleton


def build_edge_greedy_head_and_residual(
    paulis: list[str],
    angles: list[float],
    connectivity=None,
    lookahead: int = 10,
    gamma: float = 0.7,
    lam: float = 0.5,
):
    """
    Synthesise a head circuit and its pure-Clifford residual one coupler gate
    at a time, see :func:`edge_greedy_skeleton`.

    Args:
        paulis: dressed Pauli axis strings (qubit-0-leftmost), in circuit order.
        angles: matching rotation angles.
        connectivity: ``None`` (all-to-all), ``"line"``, or the edges
            ``[(q0, q1), ...]`` of the coupling graph, in the qubit numbering of
            the strings. Every CNOT of the head is on one of them.
        lookahead: how many pending rotations beyond the front are weighed.
        gamma: discount per rank of look-ahead.
        lam: weight of the look-ahead against the front.

    Returns:
        ``(head, residual_tableau, residual_gates)``. Deterministic.
    """
    if not paulis:
        return [], stim.Tableau(0), []
    nqubits = len(paulis[0])
    skeleton = edge_greedy_skeleton(
        paulis, CouplingGraph(connectivity, nqubits), lookahead=lookahead, gamma=gamma, lam=lam
    )
    return head_and_residual_from_skeleton(skeleton, paulis, angles, nqubits)


@dataclass(frozen=True)
class EdgeGreedySynthesis(HeadSynthesis):
    """
    Coupler-by-coupler greedy synthesis, see
    :func:`build_edge_greedy_head_and_residual`.

    Args:
        connectivity: ``None`` (all-to-all), ``"line"``, or the edges
            ``[(q0, q1), ...]`` of the coupling graph, in the qubit numbering of
            the strings. Every CNOT of the head is on one of them.
        lookahead: pending rotations weighed beyond the front.
        gamma: discount per rank of look-ahead.
        lam: weight of the look-ahead against the front.
    """

    connectivity: object = None
    lookahead: int = 10
    gamma: float = 0.7
    lam: float = 0.5

    name: ClassVar[str] = "edge_greedy"

    def __post_init__(self):
        object.__setattr__(self, "connectivity", hashable_connectivity(self.connectivity))

    def synthesize(self, paulis, angles):
        return build_edge_greedy_head_and_residual(
            paulis, angles, connectivity=self.connectivity,
            lookahead=self.lookahead, gamma=self.gamma, lam=self.lam,
        )
