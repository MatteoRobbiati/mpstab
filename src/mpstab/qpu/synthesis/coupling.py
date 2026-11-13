"""
What the edge-restricted synthesisers share: the coupling graph and the Pauli frame.

:mod:`~mpstab.qpu.synthesis.steiner` and :mod:`~mpstab.qpu.synthesis.edge_greedy`
build a Clifford skeleton made **only of gates on the couplers** of the device,
so that their head needs no routing. They need two things:

- :class:`CouplingGraph`: the device connectivity, in the qubit numbering of
  the strings, with shortest paths and approximate Steiner trees;
- :class:`PauliFrame`: the image of every rotation under the skeleton built so
  far, resolved in the same order as
  :func:`~mpstab.qpu.synthesis.utils.place_rotations` will place them.

Nothing here needs a dependency beyond ``stim``.
"""

from __future__ import annotations

import stim

from mpstab.qpu.synthesis.extraction import connectivity_distances
from mpstab.qpu.synthesis.utils import CLIFFORD_TABLEAUS

#: The inverse of each Clifford gate an edge-restricted skeleton is made of.
SKELETON_INVERSE = {"H": "H", "SqrtX": "SqrtXd", "CNOT": "CNOT"}


class CouplingGraph:
    """
    A device connectivity in the qubit numbering of the strings.

    Args:
        connectivity: ``None`` (all-to-all), ``"line"``, or the edges
            ``[(q0, q1), ...]`` of the coupling graph, as for
            :class:`~mpstab.qpu.synthesis.extraction.ExtractionSynthesis`.
        nqubits: number of qubits.

    Raises:
        ValueError: if the graph is not connected.
    """

    def __init__(self, connectivity, nqubits: int):
        self.nqubits = nqubits
        if connectivity is None:
            edges = [(a, b) for a in range(nqubits) for b in range(a + 1, nqubits)]
        elif isinstance(connectivity, str):
            if connectivity != "line":
                raise ValueError(
                    f"Unknown connectivity {connectivity!r}: use 'line' or a list of edges."
                )
            edges = [(q, q + 1) for q in range(nqubits - 1)]
        else:
            edges = sorted({tuple(sorted(edge)) for edge in connectivity})
        self.edges = edges
        self.neighbours = [[] for _ in range(nqubits)]
        for a, b in edges:
            self.neighbours[a].append(b)
            self.neighbours[b].append(a)
        for row in self.neighbours:
            row.sort()
        if connectivity is None:
            self.distances = [[int(a != b) for b in range(nqubits)] for a in range(nqubits)]
        else:
            self.distances = connectivity_distances(edges, nqubits)

    @property
    def directed_edges(self) -> list:
        """Every coupler in both orientations, as (control, target) candidates."""
        return self.edges + [(b, a) for a, b in self.edges]

    def has_edge(self, a: int, b: int) -> bool:
        return b in self.neighbours[a]

    def shortest_path(self, source: int, target: int) -> list:
        """A shortest path, as a list of nodes (BFS, smallest neighbour first)."""
        parent = {source: None}
        frontier = [source]
        while target not in parent:
            following = []
            for node in frontier:
                for other in self.neighbours[node]:
                    if other not in parent:
                        parent[other] = node
                        following.append(other)
            frontier = following
        path = [target]
        while path[-1] != source:
            path.append(parent[path[-1]])
        return path[::-1]

    def tree_cost(self, support) -> int:
        """
        Edges of an approximate Steiner tree joining ``support``: the minimum
        spanning tree of the shortest-path distances (Prim), at most twice the
        optimum. The number of CNOTs that collapse a string onto one qubit.
        """
        support = list(support)
        if len(support) <= 1:
            return 0
        best = {q: self.distances[support[0]][q] for q in support[1:]}
        total = 0
        while best:
            q = min(best, key=best.get)
            total += best.pop(q)
            for r in best:
                best[r] = min(best[r], self.distances[q][r])
        return total

    def steiner_tree(self, terminals) -> dict:
        """
        An approximate Steiner tree of ``terminals`` (Kou, Markowsky and Berman):
        the minimum spanning tree of their shortest-path distances, each of its
        edges expanded into a shortest path, spanned again and pruned of
        non-terminal leaves. Returned as an adjacency ``{node: [neighbours]}``.
        """
        terminals = sorted(set(terminals))
        if len(terminals) == 1:
            return {terminals[0]: []}
        # 1. MST of the metric closure on the terminals
        joined, closure = [terminals[0]], []
        pending = terminals[1:]
        while pending:
            a, b = min(((j, p) for j in joined for p in pending),
                       key=lambda e: (self.distances[e[0]][e[1]], e))
            closure.append((a, b))
            joined.append(b)
            pending.remove(b)
        # 2. expand every closure edge into a shortest path
        edges = set()
        for a, b in closure:
            path = self.shortest_path(a, b)
            edges.update(tuple(sorted(e)) for e in zip(path, path[1:]))
        # 3. spanning tree of that subgraph (BFS from the first terminal)
        adjacency: dict = {}
        for a, b in edges:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        tree = {terminals[0]: []}
        frontier = [terminals[0]]
        while frontier:
            following = []
            for node in frontier:
                for other in sorted(adjacency[node]):
                    if other not in tree:
                        tree[other] = [node]
                        tree[node].append(other)
                        following.append(other)
            frontier = following
        # 4. prune the leaves that are not terminals
        keep = set(terminals)
        leaves = [n for n, nb in tree.items() if len(nb) == 1 and n not in keep]
        while leaves:
            leaf = leaves.pop()
            (other,) = tree.pop(leaf)
            tree[other].remove(leaf)
            if len(tree[other]) == 1 and other not in keep:
                leaves.append(other)
        return tree


class PauliFrame:
    """
    The Pauli image of every rotation under the skeleton built so far.

    A rotation is *resolved* once its image has weight at most one and no
    earlier unresolved rotation anticommutes with it: exactly when
    :func:`~mpstab.qpu.synthesis.utils.place_rotations` would emit it. The
    unresolved rotations with no unresolved anticommuting predecessor are the
    *front*, the ones the skeleton can work on now.
    """

    def __init__(self, paulis):
        n = len(paulis)
        self.nqubits = len(paulis[0])
        self.images = [stim.PauliString(p) for p in paulis]
        xs = [sum(1 << q for q, c in enumerate(p) if c in "XY") for p in paulis]
        zs = [sum(1 << q for q, c in enumerate(p) if c in "ZY") for p in paulis]
        self.blockers = [0] * n
        self.blocks = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i):
                if ((xs[i] & zs[j]) ^ (zs[i] & xs[j])).bit_count() & 1:
                    self.blockers[i] += 1
                    self.blocks[j].append(i)
        self.resolved = [False] * n
        self.skeleton: list = []
        self._drain()

    def _drain(self):
        progress = True
        while progress:
            progress = False
            for i, image in enumerate(self.images):
                if not self.resolved[i] and self.blockers[i] == 0 and image.weight <= 1:
                    self.resolved[i] = True
                    for later in self.blocks[i]:
                        self.blockers[later] -= 1
                    progress = True

    @property
    def done(self) -> bool:
        return all(self.resolved)

    def front(self) -> list:
        return [i for i, r in enumerate(self.resolved) if not r and self.blockers[i] == 0]

    def apply(self, name: str, qubits) -> None:
        """Append a Clifford gate to the skeleton and update the images it touches."""
        tableau = CLIFFORD_TABLEAUS[name]
        for i, image in enumerate(self.images):
            if not self.resolved[i] and any(image[q] != 0 for q in qubits):
                self.images[i] = image.after(tableau, targets=list(qubits))
        self.skeleton.append((name, list(qubits)))
        self._drain()

    def support(self, i: int) -> list:
        image = self.images[i]
        return [q for q in range(self.nqubits) if image[q] != 0]


def head_and_residual_from_skeleton(skeleton, paulis, angles, nqubits):
    """
    ``(head, residual_tableau, residual_gates)`` from a skeleton of ``H``,
    ``SqrtX`` and ``CNOT`` gates: the rotations placed into it, and its inverse
    as the residual.
    """
    from mpstab.qpu.synthesis.utils import place_rotations

    head = place_rotations(skeleton, paulis, angles, nqubits)
    residual = [(SKELETON_INVERSE[name], qubits) for name, qubits in reversed(skeleton)]
    residual_tableau = stim.Tableau(nqubits)
    for name, qubits in residual:
        residual_tableau.append(CLIFFORD_TABLEAUS[name], qubits)
    return head, residual_tableau, residual


def hashable_connectivity(connectivity):
    """``connectivity`` as a hashable value, so a synthesiser can key the resynthesis cache."""
    if connectivity is None or isinstance(connectivity, str):
        return connectivity
    return tuple(tuple(edge) for edge in connectivity)
