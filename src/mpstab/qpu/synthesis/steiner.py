"""
Head synthesis along Steiner trees of the coupling graph: no CNOT ever leaves a
coupler, so the head needs no routing.

Like :mod:`~mpstab.qpu.synthesis.extraction`, it brings each rotation onto one
qubit with Cliffords that are never undone. The difference is where the CNOTs
may go: extraction picks them between sites of the rotation's own support and
charges a distant pair the SWAPs a router would add; here every CNOT is on a
coupler, and when the support is not connected on the device the string is
first *moved* through the qubits in between (the Steiner nodes of the tree).
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

#: Single-qubit Clifford that maps the Pauli (stim code: 1=X, 2=Y, 3=Z) onto Z.
TO_Z = {1: "H", 2: "SqrtX", 3: None}


def steiner_reduce(frame: PauliFrame, i: int, graph: CouplingGraph) -> None:
    """
    Bring rotation ``i`` onto one qubit, every CNOT on a coupler.

    The string's sites are turned to ``Z``; then, on an approximate Steiner
    tree of its support, the parity is collected from the leaves to the root,
    one CNOT per tree edge (``Z_child Z_parent -> Z_parent``), plus one more
    for each tree node the string does not touch, where the ``Z`` is first
    moved onto that node. The root is the support qubit of smallest
    eccentricity in the tree. Gates may also resolve other rotations, which the
    frame records.
    """
    while not frame.resolved[i]:
        for qubit in frame.support(i):
            gate = TO_Z[frame.images[i][qubit]]
            if gate:
                frame.apply(gate, [qubit])
                if frame.resolved[i]:
                    return
        support = frame.support(i)
        if len(support) <= 1:
            return
        tree = graph.steiner_tree(support)

        def depth_from(root):
            seen, frontier, depth = {root}, [root], 0
            while frontier:
                following = [o for n in frontier for o in tree[n] if o not in seen]
                seen.update(following)
                frontier = following
                depth += bool(following)
            return depth

        root = min(support, key=lambda s: (depth_from(s), s))
        parent, order, frontier = {root: None}, [], [root]
        while frontier:  # BFS order; reversed, children come before parents
            following = []
            for node in frontier:
                order.append(node)
                for other in tree[node]:
                    if other not in parent:
                        parent[other] = node
                        following.append(other)
            frontier = following
        for child in reversed(order[1:]):
            up = parent[child]
            image = frame.images[i]
            if frame.resolved[i] or image[child] == 0:
                continue
            if image[up] == 0:  # a node the string does not touch: move Z there first
                frame.apply("CNOT", [up, child])
            frame.apply("CNOT", [child, up])


def steiner_skeleton(paulis, graph: CouplingGraph) -> list:
    """The Clifford skeleton: the rotations one at a time, in order."""
    frame = PauliFrame(paulis)
    for i in range(len(paulis)):
        steiner_reduce(frame, i, graph)
    return frame.skeleton


def build_steiner_head_and_residual(paulis: list[str], angles: list[float], connectivity=None):
    """
    Synthesise a head circuit and its pure-Clifford residual along Steiner
    trees of the coupling graph.

    Args:
        paulis: dressed Pauli axis strings (qubit-0-leftmost), in circuit order.
        angles: matching rotation angles.
        connectivity: ``None`` (all-to-all), ``"line"``, or the edges
            ``[(q0, q1), ...]`` of the coupling graph, in the qubit numbering of
            the strings. Every CNOT of the head is on one of them.

    Returns:
        ``(head, residual_tableau, residual_gates)``, the same shape as
        :func:`~mpstab.qpu.synthesis.extraction.build_extraction_head_and_residual`.
        Deterministic.
    """
    if not paulis:
        return [], stim.Tableau(0), []
    nqubits = len(paulis[0])
    skeleton = steiner_skeleton(paulis, CouplingGraph(connectivity, nqubits))
    return head_and_residual_from_skeleton(skeleton, paulis, angles, nqubits)


@dataclass(frozen=True)
class SteinerSynthesis(HeadSynthesis):
    """
    Steiner-tree synthesis, see :func:`build_steiner_head_and_residual`.

    Args:
        connectivity: ``None`` (all-to-all), ``"line"``, or the edges
            ``[(q0, q1), ...]`` of the coupling graph, in the qubit numbering of
            the strings. Every CNOT of the head is on one of them.
    """

    connectivity: object = None

    name: ClassVar[str] = "steiner"

    def __post_init__(self):
        object.__setattr__(self, "connectivity", hashable_connectivity(self.connectivity))

    def synthesize(self, paulis, angles):
        return build_steiner_head_and_residual(paulis, angles, connectivity=self.connectivity)
