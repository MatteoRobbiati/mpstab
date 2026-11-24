"""
Coupling topologies: which pairs of qubits an ansatz couples.

A :class:`Topology` says which pairs of qubits an ansatz couples. It is a
graph whose nodes are the qubits ``0, 1, ..., nqubits - 1`` and whose edges
are the coupled pairs, kept in order.

An ansatz takes it through its ``graph`` field and puts one two-qubit gate on
every edge, a layer of disjoint edges at a time, at every step or layer of the
circuit. The rest is up to the ansatz: the single-qubit gates, the angles and
which two-qubit gate goes on the edges (CZ, RZZ, RBS, ...). For example,
``RampedKickedIsing(nqubits=12, graph="grid")`` couples the qubits of a
3 x 4 lattice instead of a chain. For a kicked Ising ansatz, the topology is
the interaction graph of its Hamiltonian.

Where a topology comes from:

- a name: ``"chain"``, ``"ring"``, ``"star"``, ``"grid"`` (or
  ``"grid:RxC"``), ``"complete"``, see :meth:`Topology.named`;
- a list of edges, see :meth:`Topology.from_edges`;
- a graph object (``networkx``, ``rustworkx``, an adjacency mapping), see
  :meth:`Topology.from_graph`;
- the coupling map of a device, see :meth:`Topology.from_device`, to
  build the ansatz on the couplings the chip has.

:meth:`Topology.coerce` accepts any of these inputs and is what the ansätze
call.

Two properties matter for building circuits:

- **edge order**: it is kept, because it fixes the order of the gates;
- **layers**: :attr:`Topology.layers` groups the edges into layers of
  disjoint bonds, which can run in parallel.

An ansatz runs on a device without routing when every pair it couples is
coupled on the chip: :meth:`Topology.is_subgraph_of` checks it against the
coupling map of the device. Building the ansatz on that coupling map, or on
part of it, is the simplest way to get there.

Example::

    from mpstab import Topology

    grid = Topology.named("grid", 6)  # 2 x 3 lattice
    grid.edges   # ((0, 1), (3, 4), (0, 3), (1, 4), (2, 5), (1, 2), (4, 5))
    grid.depth   # 3 layers of disjoint bonds

    coupling_map = [(0, 1), (1, 0), (1, 2), (2, 1)]  # a 3-qubit device
    Topology.named("chain", 3).is_subgraph_of(coupling_map)  # True
    Topology.named("ring", 3).is_subgraph_of(coupling_map)   # False: no (2, 0)

A topology holds qubit-qubit couplings only: no couplers or resonators, no
allowed gate directions, no fidelities. How a device realises a coupling (a
direct gate, a SWAP network, a shared bus or resonator) is left to the
compilation for that device.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Integral
from typing import Iterable, Sequence

import networkx as nx
import numpy as np

Edge = tuple[int, int]

#: Names accepted by :meth:`Topology.named`; ``grid`` also takes a shape,
#: ``"grid:RxC"``.
TOPOLOGIES = ("chain", "ring", "star", "grid", "complete")


@dataclass(frozen=True)
class Topology:
    """
    The pairs of qubits an ansatz couples, on ``nqubits`` qubits, as an ordered
    tuple of edges.

    Rather than calling ``Topology(nqubits, edges)`` directly, build one with
    a class method, which works out ``nqubits`` and the edges for you:
    :meth:`named`, :meth:`from_edges`, :meth:`from_graph`,
    :meth:`from_device`, or :meth:`coerce`, which accepts the input of
    any of them.

    Rules on the edges, checked at construction:

    - each edge is a pair of distinct qubits in ``[0, nqubits)``;
    - each pair appears at most once, in either orientation;
    - order and orientation are kept as given: an ansatz applies its gates in
      that order and, for directed gates such as CNOT, with that orientation.

    Two topologies are equal when they have the same ``nqubits`` and the same
    edges in the same order and orientation, since they build different
    circuits otherwise. To compare the graphs alone, ignoring order and
    orientation, check :meth:`is_subgraph_of` both ways.

    Args:
        nqubits: number of qubits. Qubits with no edge are allowed.
        edges: the edges, in order.
        name: a label such as ``"grid:4x5"``; ``None`` for a custom graph.
            It does not take part in equality.

    Raises:
        ValueError: on a negative ``nqubits``, a self-loop, a qubit out of
            range or a repeated pair.
    """

    nqubits: int
    edges: tuple[Edge, ...]
    name: str | None = field(default=None, compare=False)

    def __post_init__(self):
        if self.nqubits < 0:
            raise ValueError(f"nqubits must be non-negative, got {self.nqubits}.")
        edges, seen = [], set()
        for edge in self.edges:
            a, b = _edge(edge)
            if a == b or not (a < self.nqubits and b < self.nqubits):
                raise ValueError(
                    f"Edge {(a, b)} is not a pair of distinct qubits in "
                    f"[0, {self.nqubits})."
                )
            key = (min(a, b), max(a, b))
            if key in seen:
                raise ValueError(f"Edge {(a, b)} appears twice in the topology.")
            seen.add(key)
            edges.append((a, b))
        object.__setattr__(self, "edges", tuple(edges))

    @classmethod
    def named(cls, name: str, nqubits: int) -> Topology:
        """
        A named topology on ``nqubits`` qubits.

        ===============  ===================================================
        Name             Edges
        ===============  ===================================================
        ``chain``        ``(q, q+1)``: even bonds first, then odd bonds.
        ``ring``         the chain plus ``(n-1, 0)`` when ``n > 2``.
        ``star``         ``(0, leaf)`` for every leaf; the hub is qubit 0.
        ``grid``         a ``R x C`` lattice, qubits numbered row by row.
        ``grid:RxC``     the same, with the shape given explicitly.
        ``complete``     every pair, in round-robin order.
        ===============  ===================================================

        Default grid shape: ``R`` is the largest divisor of ``n`` not above
        ``sqrt(n)``, so the rectangle is as square as possible. A prime
        ``n > 3`` only fits a ``1 x n`` rectangle and has no default shape: ask
        for ``"grid:1xn"`` (a chain) or give the edges.

        The edges are listed so that :attr:`layers` uses as few layers as
        possible: the maximum degree, plus one for an odd ring and for the
        complete graph on an odd number of qubits.

        Args:
            name: one of :data:`TOPOLOGIES`, or ``"grid:RxC"``.
            nqubits: number of qubits.

        Raises:
            ValueError: on an unknown name, or a grid shape that does not hold
                ``nqubits`` qubits.
        """
        n = int(nqubits)
        base, _, shape = name.partition(":")
        if shape and base != "grid":
            raise ValueError(f"Only 'grid' takes a shape, got {name!r}.")
        if base == "chain":
            edges = [(q, q + 1) for start in (0, 1) for q in range(start, n - 1, 2)]
        elif base == "ring":
            edges = cls.named("chain", n).edges + (((n - 1, 0),) if n > 2 else ())
        elif base == "star":
            edges = [(0, leaf) for leaf in range(1, n)]
        elif base == "grid":
            rows, cols = _grid_shape(n, shape)
            edges = grid_edges(rows, cols)
            name = f"grid:{rows}x{cols}"
        elif base == "complete":
            edges = _round_robin(n)
        else:
            raise ValueError(
                f"Unknown topology {name!r}; choose one of {TOPOLOGIES} "
                "(or 'grid:RxC')."
            )
        return cls(n, tuple(edges), name)

    @classmethod
    def from_edges(
        cls, edges: Iterable[Sequence[int]], nqubits: int | None = None
    ) -> Topology:
        """
        A topology with exactly these edges, in this order and orientation.

        The list is taken strictly: a repeated pair or a self-loop is an error.
        For a list that holds both directions of each pair, as a device
        coupling map does, use :meth:`from_device`.

        Args:
            edges: pairs of qubit indices.
            nqubits: number of qubits. Defaults to one more than the largest
                index in ``edges``.
        """
        edges = [_edge(edge) for edge in edges]
        if nqubits is None:
            nqubits = 1 + max((q for edge in edges for q in edge), default=-1)
        return cls(int(nqubits), tuple(edges))

    @classmethod
    def from_graph(
        cls,
        graph,
        nqubits: int | None = None,
    ) -> Topology:
        """
        A topology from a graph object, ignoring directions and repetitions.

        What is cleaned up, unlike :meth:`from_edges`:

        - a pair listed in both directions is kept once, in its first
          orientation;
        - repeated pairs and self-loops are dropped.

        The nodes must be the qubit indices ``0, 1, ...``. Relabel the graph
        first (``networkx.relabel_nodes``) if they are not. A device whose
        qubits couple only through a shared resonator has no qubit-qubit
        edges: describe it with ``"star"``, a qubit standing for the hub.

        Args:
            graph: one of

                - a ``networkx.Graph`` (anything with ``.edges``);
                - a ``rustworkx.PyGraph`` (anything with ``.edge_list()``);
                - an adjacency mapping ``{q: [neighbours of q], ...}``;
                - a sequence of pairs, such as a device coupling map.
            nqubits: number of qubits. By default, one more than the largest
                qubit index among the nodes of ``graph``, so that qubits with
                no edge are kept; a sequence of pairs lists no such qubits,
                and there it is one more than the largest index in the edges.
        """
        if nqubits is None:
            nodes = _nodes(graph)
            if nodes is not None:
                nqubits = 1 + max((_qubit(q) for q in nodes), default=-1)
        edges, seen = [], set()
        for a, b in _pairs(graph):
            if a == b:
                continue
            a, b = _edge((a, b))
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                edges.append((a, b))
        return cls.from_edges(edges, nqubits)

    @classmethod
    def from_device(cls, device: GraphLike, nqubits: int | None = None) -> Topology:
        """
        The topology of a device, from whatever describes its couplings.

        - a :class:`Topology` is used as it is;
        - a name is built on ``nqubits`` qubits, see :meth:`named`;
        - anything else is read as a coupling map, with :meth:`from_graph`:
          directions and repeated pairs are ignored.

        Unlike :meth:`coerce`, which builds the topology of an ansatz, a list of
        pairs is read tolerantly, since a coupling map lists each pair in both
        directions, and the device keeps all its qubits, however many the
        circuit has.

        Args:
            device: the couplings of the device.
            nqubits: number of qubits of a named device; ignored otherwise.

        Raises:
            ValueError: if a name comes without ``nqubits``.
        """
        if isinstance(device, Topology):
            return device
        if isinstance(device, str):
            if nqubits is None:
                raise ValueError(f"The named topology {device!r} needs nqubits.")
            return cls.named(device, nqubits)
        return cls.from_graph(device)

    @classmethod
    def coerce(cls, graph, nqubits: int | None = None) -> Topology:
        """
        A :class:`Topology` from whatever describes one.

        This is what the ``graph`` field of an ansatz goes through.

        ============================  =========================================
        Input                         Read by
        ============================  =========================================
        a :class:`Topology`           used as is
        a ``str``                     :meth:`named` (needs ``nqubits``)
        a graph object or a mapping   :meth:`from_graph`
        a sequence of pairs           :meth:`from_edges` (strict)
        ============================  =========================================

        A sequence of pairs is a ``list``, a ``tuple``, a numpy array or any
        other iterable of pairs. It is read strictly, so a device coupling map,
        which lists each pair in both directions, must go through
        :meth:`from_device` first.

        Args:
            graph: the description of the topology, see the table above.
            nqubits: the number of qubits the topology must have. Required for
                a name. For the other inputs, if the graph has fewer qubits,
                the missing ones are added with no edges (and the ``name`` is
                dropped); if it has more, it is an error.

        Raises:
            ValueError: if a name comes without ``nqubits``, or the graph has
                more than ``nqubits`` qubits.
        """
        if isinstance(graph, Topology):
            topology = graph
        elif isinstance(graph, str):
            if nqubits is None:
                raise ValueError(f"The named topology {graph!r} needs nqubits.")
            return cls.named(graph, nqubits)
        elif _is_graph_object(graph):
            topology = cls.from_graph(graph)
        else:
            topology = cls.from_edges(graph)
        if nqubits is None or topology.nqubits == nqubits:
            return topology
        if topology.nqubits > nqubits:
            raise ValueError(
                f"The topology has {topology.nqubits} qubits, more than the "
                f"{nqubits} asked for."
            )
        # with extra qubits, a label such as "grid:2x3" no longer describes it
        return cls(int(nqubits), topology.edges)

    @cached_property
    def layers(self) -> tuple[tuple[Edge, ...], ...]:
        """
        The edges grouped into layers of disjoint bonds.

        No two bonds of a layer share a qubit, so they can run in parallel and
        the number of layers is the two-qubit depth of one pass over the
        edges. See :func:`coupling_layers` for how the layers are built. The
        layers are tuples, like the edges, so the topology stays immutable.
        """
        return tuple(tuple(layer) for layer in coupling_layers(self.edges))

    @property
    def depth(self) -> int:
        """Number of :attr:`layers`."""
        return len(self.layers)

    def neighbours(self, qubit: int) -> list[int]:
        """The qubits coupled to ``qubit``, in edge order."""
        return [b if a == qubit else a for a, b in self.edges if qubit in (a, b)]

    def degree(self, qubit: int) -> int:
        """Number of edges at ``qubit``."""
        return len(self.neighbours(qubit))

    def is_subgraph_of(self, other: GraphLike) -> bool:
        """
        Whether every edge of this topology is also an edge of ``other``.

        Order and orientation of the edges are ignored, and ``other`` may have
        more qubits. Typical use: ``ansatz.topology.is_subgraph_of(coupling_map)``
        tells whether every two-qubit gate of the ansatz sits on a coupling of
        the device, so that the circuit runs there without routing. The qubits
        are compared by index: qubit 3 of the ansatz is qubit 3 of the device.

        Args:
            other: the larger topology, such as the coupling map of a device,
                read by :meth:`from_device` (a name is built on
                ``self.nqubits`` qubits).
        """
        other = Topology.from_device(other, self.nqubits)
        if self.nqubits > other.nqubits:
            return False
        available = {frozenset(edge) for edge in other.edges}
        return all(frozenset(edge) in available for edge in self.edges)

    def is_connected(self) -> bool:
        """Whether every qubit can reach every other one through the edges."""
        return self.nqubits <= 1 or nx.is_connected(self.to_networkx())

    def to_networkx(self) -> nx.Graph:
        """The topology as an undirected ``networkx.Graph`` on ``0 .. nqubits - 1``."""
        graph = nx.Graph()
        graph.add_nodes_from(range(self.nqubits))
        graph.add_edges_from(self.edges)
        return graph

    def __len__(self) -> int:
        """Number of edges. A topology with no edges is falsy: test ``is None``."""
        return len(self.edges)


def grid_edges(rows: int, cols: int) -> list[Edge]:
    """
    Edges of a ``rows x cols`` lattice, qubits numbered row by row.

    The edges come in four groups, each row-major: even horizontal, even
    vertical, odd horizontal, odd vertical bonds. With this order
    :func:`coupling_layers` needs at most four layers.
    """
    horizontal = [
        [
            (r * cols + c, r * cols + c + 1)
            for r in range(rows)
            for c in range(start, cols - 1, 2)
        ]
        for start in (0, 1)
    ]
    vertical = [
        [
            (r * cols + c, (r + 1) * cols + c)
            for r in range(start, rows - 1, 2)
            for c in range(cols)
        ]
        for start in (0, 1)
    ]
    return horizontal[0] + vertical[0] + horizontal[1] + vertical[1]


def coupling_layers(edges: Sequence[Edge]) -> list[list[Edge]]:
    """
    Split ``edges`` into layers of disjoint bonds (first-fit colouring).

    The edges are taken in order, and each goes into the first layer where
    neither of its qubits is used yet; if there is none, a new layer is
    opened. The result therefore depends on the order of ``edges``.

    Example::

        coupling_layers([(0, 1), (1, 2), (2, 3)])
        # [[(0, 1), (2, 3)], [(1, 2)]]
    """
    layers: list[list[Edge]] = []
    busy: list[set] = []
    for a, b in edges:
        for layer, used in zip(layers, busy):
            if a not in used and b not in used:
                layer.append((a, b))
                used.update((a, b))
                break
        else:
            layers.append([(a, b)])
            busy.append({a, b})
    return layers


def _grid_shape(n: int, shape: str) -> tuple[int, int]:
    """The ``(rows, cols)`` of a grid on ``n`` qubits, from ``"RxC"`` or by default."""
    if shape:
        try:
            rows, cols = (int(x) for x in shape.lower().split("x"))
        except ValueError:
            raise ValueError(f"A grid shape reads 'RxC', got {shape!r}.") from None
        if rows < 1 or cols < 1 or rows * cols != n:
            raise ValueError(f"A {rows}x{cols} grid does not hold {n} qubits.")
        return rows, cols
    if n == 0:
        return 0, 0
    rows = max(d for d in range(1, int(np.floor(np.sqrt(n))) + 1) if n % d == 0)
    if rows == 1 and n > 3:
        raise ValueError(
            f"{n} qubits only fill a 1 x {n} grid; ask for 'grid:1x{n}' (a chain) "
            "or give the edges of the lattice you want."
        )
    return rows, n // rows


def _round_robin(n: int) -> list[Edge]:
    """Every pair of ``n`` qubits, as rounds of disjoint pairs (circle method)."""
    players = list(range(n)) + ([None] if n % 2 else [])
    m = len(players)
    edges = []
    for _ in range(m - 1):
        for i in range(m // 2):
            a, b = players[i], players[m - 1 - i]
            if a is not None and b is not None:
                edges.append((min(a, b), max(a, b)))
        players = [players[0], players[-1]] + players[1:-1]
    return edges


def _pairs(graph):
    """The pairs of a graph object, an adjacency mapping or a sequence of pairs."""
    if hasattr(graph, "edge_list"):  # rustworkx
        return list(graph.edge_list())
    if hasattr(graph, "edges"):  # networkx
        edges = graph.edges
        return list(edges() if callable(edges) else edges)
    if isinstance(graph, Mapping):
        return [(q, other) for q, others in graph.items() for other in others]
    return [tuple(pair) for pair in graph]


def _nodes(graph) -> list | None:
    """The nodes of a graph object or an adjacency mapping; ``None`` for a sequence of pairs."""
    if hasattr(graph, "node_indices"):  # rustworkx
        return list(graph.node_indices())
    if hasattr(graph, "nodes"):  # networkx
        return list(graph.nodes)
    if isinstance(graph, Mapping):
        nodes = dict.fromkeys(graph)  # the keys, then neighbours that are no key
        for others in graph.values():
            nodes.update(dict.fromkeys(others))
        return list(nodes)
    return None


def _is_graph_object(graph) -> bool:
    """Whether ``graph`` is a graph object or a mapping, rather than a sequence of pairs."""
    return (
        hasattr(graph, "edge_list")  # rustworkx
        or hasattr(graph, "edges")  # networkx
        or isinstance(graph, Mapping)
    )


def _qubit(node) -> int:
    """``node`` as an ``int``; raises if it is not a qubit index."""
    if not isinstance(node, Integral) or isinstance(node, bool) or node < 0:
        raise TypeError(
            f"Node {node!r} is not a qubit index: relabel the graph with the "
            "integers 0, 1, ... (e.g. networkx.relabel_nodes)."
        )
    return int(node)


def _edge(edge) -> Edge:
    """``edge`` as a pair of ``int``; raises if it is not two qubit indices."""
    pair = tuple(edge)
    if len(pair) != 2:
        raise ValueError(f"An edge must be a pair of qubits, got {pair!r}.")
    return _qubit(pair[0]), _qubit(pair[1])


#: Anything :meth:`Topology.coerce` accepts.
GraphLike = Topology | str | nx.Graph | Mapping | Sequence[Sequence[int]]
