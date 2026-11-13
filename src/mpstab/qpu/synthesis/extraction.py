"""
Head synthesis by Clifford extraction (the idea of QuCLEAR, arXiv:2408.13316).

Each rotation is reduced to a single qubit by a Clifford that is never undone
but pushed through the rotations still to come. It needs no extra dependency and
can be told the device connectivity, so that the CNOTs it chooses stay local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import stim

from mpstab.qpu.synthesis.abstract import HeadSynthesis
from mpstab.qpu.synthesis.utils import CLIFFORD_TABLEAUS, place_rotations

#: The inverse of each Clifford gate an extraction skeleton is made of.
_EXTRACTION_INVERSE = {"H": "H", "SqrtX": "SqrtXd", "CNOT": "CNOT"}


#: What :func:`build_extraction_head_and_residual` tries, keeping the cheapest:
#: ``locality`` weighs how far apart the sites of the coming strings end up (it
#: only matters with a connectivity), ``reorder`` lets a rotation overtake the
#: ones it commutes with, ``lookahead`` is how many coming strings a CNOT is
#: judged on. No single setting wins on every circuit.
EXTRACTION_STRATEGIES = (
    {"locality": 2.0, "reorder": True, "lookahead": 8},
    {"locality": 2.0, "reorder": False, "lookahead": 8},
    {"locality": 2.0, "reorder": False, "lookahead": 4},
    {"locality": 0.0, "reorder": True, "lookahead": 8},
    {"locality": 0.0, "reorder": False, "lookahead": 8},
)


#: A CNOT between qubits at distance ``d`` costs about ``d - 1`` SWAPs more.
_SWAP_COST = 3.0


def connectivity_distances(connectivity, nqubits: int):
    """
    All-pairs distances of a device connectivity, as a ``nqubits x nqubits``
    list, or ``None`` for all-to-all.

    Args:
        connectivity: ``None`` (all-to-all), ``"line"`` (qubit ``q`` next to
            ``q + 1``), or the edges ``[(q0, q1), ...]`` of the coupling graph.
        nqubits: number of qubits.
    """
    if connectivity is None:
        return None
    if isinstance(connectivity, str):
        if connectivity != "line":
            raise ValueError(
                f"Unknown connectivity {connectivity!r}: use 'line' or a list of edges."
            )
        edges = [(q, q + 1) for q in range(nqubits - 1)]
    else:
        edges = [tuple(edge) for edge in connectivity]
    neighbours = [[] for _ in range(nqubits)]
    for q0, q1 in edges:
        neighbours[q0].append(q1)
        neighbours[q1].append(q0)
    distances = []
    for source in range(nqubits):
        row = [None] * nqubits
        row[source] = 0
        frontier = [source]
        while frontier:
            following = []
            for qubit in frontier:
                for other in neighbours[qubit]:
                    if row[other] is None:
                        row[other] = row[qubit] + 1
                        following.append(other)
            frontier = following
        if any(d is None for d in row):
            raise ValueError("The connectivity graph is not connected.")
        distances.append(row)
    return distances


def _extraction_skeleton(paulis, distances, locality, reorder, lookahead, discount=0.7):
    """
    The Clifford skeleton of one extraction strategy, and its estimated cost in
    two-qubit gates on the device (a CNOT at distance ``d`` counts as
    ``1 + 3 * (d - 1)``).

    Every string is carried as two bit masks (``x`` and ``z`` parts). For the
    rotation in turn: single-qubit Cliffords take its string to all-``Z``, then
    CNOTs between two sites of its support remove one site each until one is
    left. Each CNOT is the one that leaves the coming strings touching those
    sites lightest (and, with a connectivity, most local).
    """
    nqubits = len(paulis[0])
    xs = [sum(1 << q for q, c in enumerate(p) if c in "XY") for p in paulis]
    zs = [sum(1 << q for q, c in enumerate(p) if c in "ZY") for p in paulis]
    remaining = list(range(len(paulis)))
    skeleton: list = []
    estimated_cost = 0.0
    spread_cache: dict = {}

    def spread(mask):
        """Length of a tree joining the support of ``mask`` on the device."""
        if distances is None:
            return 0
        if mask not in spread_cache:
            sites = [q for q in range(nqubits) if (mask >> q) & 1]
            joined, total = sites[:1], 0
            pending = sites[1:]
            while pending:  # Prim: attach the closest site left
                best = min(pending, key=lambda q: min(distances[q][j] for j in joined))
                total += min(distances[best][j] for j in joined)
                joined.append(best)
                pending.remove(best)
            spread_cache[mask] = total
        return spread_cache[mask]

    def string_cost(mask):
        weight = mask.bit_count()
        if weight < 2 or not locality or distances is None:
            return weight
        return weight + locality * (spread(mask) - (weight - 1))

    def apply(name, qubits, strings):
        skeleton.append((name, list(qubits)))
        if name == "H":
            bit = 1 << qubits[0]
            for i in strings:
                if bool(xs[i] & bit) != bool(zs[i] & bit):
                    xs[i] ^= bit
                    zs[i] ^= bit
        elif name == "SqrtX":  # X -> X, Y -> Z, Z -> Y, up to signs
            bit = 1 << qubits[0]
            for i in strings:
                if zs[i] & bit:
                    xs[i] ^= bit
        else:
            control, target = 1 << qubits[0], 1 << qubits[1]
            for i in strings:
                if xs[i] & control:
                    xs[i] ^= target
                if zs[i] & target:
                    zs[i] ^= control

    def anticommute(i, j):
        return ((xs[i] & zs[j]).bit_count() + (zs[i] & xs[j]).bit_count()) & 1

    while remaining:
        pick = 0
        if reorder:  # a rotation may overtake the ones it commutes with
            front = [
                k
                for k, i in enumerate(remaining[:24])
                if not any(anticommute(i, j) for j in remaining[:k])
            ]
            # the cheapest of them first: lightest, and with a connectivity
            # most local, so that a commuting block is reduced from its easy end
            pick = min(
                front,
                key=lambda k: (string_cost(xs[remaining[k]] | zs[remaining[k]]), k),
            )
        current = remaining.pop(pick)
        strings = [current] + remaining
        support = [q for q in range(nqubits) if ((xs[current] | zs[current]) >> q) & 1]
        for qubit in support:
            has_x, has_z = (xs[current] >> qubit) & 1, (zs[current] >> qubit) & 1
            if has_x and has_z:
                apply("SqrtX", [qubit], strings)
            elif has_x:
                apply("H", [qubit], strings)

        while len(support) > 1:
            mask = sum(1 << q for q in support)
            touching = [j for j in remaining[:400] if (xs[j] | zs[j]) & mask]
            touching = touching[: 4 * lookahead]
            best = None
            for control in support:
                for target in support:
                    if control == target:
                        continue
                    c_bit, t_bit = 1 << control, 1 << target
                    cost, seen = 0.0, 0
                    for j in touching:
                        x, z = xs[j], zs[j]
                        if not (x | z) & (c_bit | t_bit):
                            continue
                        before = string_cost(x | z)
                        if x & c_bit:
                            x ^= t_bit
                        if z & t_bit:
                            z ^= c_bit
                        cost += discount**seen * (string_cost(x | z) - before)
                        seen += 1
                        if seen == lookahead:
                            break
                    if distances is not None:
                        cost += locality * (distances[control][target] - 1)
                    if best is None or cost < best[0]:
                        best = (cost, control, target)
            _, control, target = best
            apply("CNOT", [control, target], strings)
            support.remove(control)
            estimated_cost += 1.0
            if distances is not None:
                estimated_cost += _SWAP_COST * (distances[control][target] - 1)
    return skeleton, estimated_cost


def build_extraction_head_and_residual(
    paulis: list[str],
    angles: list[float],
    connectivity=None,
    strategies=EXTRACTION_STRATEGIES,
):
    """
    Synthesise a head circuit and its pure-Clifford residual by Clifford
    extraction.

    A rotation of a weight-``w`` string needs ``w - 1`` CNOTs to bring it onto
    one qubit, and the textbook circuit spends as many again to undo them.
    Here they are not undone: the Clifford is pushed through the rotations
    still to come, whose strings change accordingly, and what has accumulated
    at the end is the residual, folded into the observable. This is the idea
    of QuCLEAR (Liu et al., arXiv:2408.13316); the choice of each CNOT is our
    own greedy rule, see :func:`_extraction_skeleton`.

    Every strategy of ``strategies`` is run and the cheapest head kept, cost
    being the two-qubit gates the device would need: with a ``connectivity`` a
    CNOT between distant qubits is charged the SWAPs that route it.

    Args:
        paulis: dressed Pauli axis strings (qubit-0-leftmost), in circuit order.
        angles: matching rotation angles.
        connectivity: ``None`` (all-to-all), ``"line"``, or the edges
            ``[(q0, q1), ...]`` of the device coupling graph, in the qubit
            numbering of the strings.
        strategies: the settings to try, see :data:`EXTRACTION_STRATEGIES`.

    Returns:
        ``(head, residual_tableau, residual_gates)``, the same shape as
        :func:`~mpstab.qpu.synthesis.rustiq.build_head_and_residual`. Deterministic.
    """
    if not paulis:
        return [], stim.Tableau(0), []

    nqubits = len(paulis[0])
    distances = connectivity_distances(connectivity, nqubits)
    best = None
    tried = set()
    for strategy in strategies:
        if distances is None:  # locality has no meaning: skip the duplicates
            strategy = {**strategy, "locality": 0.0}
        key = tuple(sorted(strategy.items()))
        if key in tried:
            continue
        tried.add(key)
        skeleton, cost = _extraction_skeleton(paulis, distances, **strategy)
        if best is None or cost < best[0]:
            best = (cost, skeleton)
    skeleton = best[1]

    head = place_rotations(skeleton, paulis, angles, nqubits)
    residual = [
        (_EXTRACTION_INVERSE[name], qubits) for name, qubits in reversed(skeleton)
    ]
    residual_tableau = stim.Tableau(nqubits)
    for name, qubits in residual:
        residual_tableau.append(CLIFFORD_TABLEAUS[name], qubits)
    return head, residual_tableau, residual


@dataclass(frozen=True)
class ExtractionSynthesis(HeadSynthesis):
    """
    Clifford-extraction synthesis, see :func:`build_extraction_head_and_residual`.

    Args:
        connectivity: ``None`` (all-to-all), ``"line"``, or the edges
            ``[(q0, q1), ...]`` of the device coupling graph, in the qubit
            numbering of the strings. The CNOTs chosen stay local on it.
        strategies: the settings to try, keeping the cheapest head, see
            :data:`EXTRACTION_STRATEGIES`.
    """

    connectivity: object = None
    strategies: tuple = field(default=EXTRACTION_STRATEGIES, hash=False, repr=False)

    name: ClassVar[str] = "extraction"

    def __post_init__(self):
        connectivity = self.connectivity
        if connectivity is not None and not isinstance(connectivity, str):
            # hashable, so the instance can key the resynthesis cache
            connectivity = tuple(tuple(edge) for edge in connectivity)
            object.__setattr__(self, "connectivity", connectivity)

    def synthesize(self, paulis, angles):
        return build_extraction_head_and_residual(
            paulis, angles, connectivity=self.connectivity, strategies=self.strategies
        )
