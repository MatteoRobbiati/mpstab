"""
Checks for `mpstab.models.topology` and its use in the ansätze.

- `Topology`: the named topologies, validation, the constructors (edges,
  graphs, coupling maps, devices), `coerce`, the layers and the other
  properties, `is_subgraph_of`;
- the `graph` field: every ansatz places its two-qubit gates on the edges it
  is given, in their order, and its default circuit is the one it had before
  the field existed, gate for gate;
- `Ansatz.topology`, the pairs a circuit couples;
- `connectivity` of the transpiled ansätze;
- the two-qubit fixes of `HardwareEfficient` and `HammingWeightPreserving`.
"""

from functools import reduce

import networkx as nx
import numpy as np
import pytest
from qibo import set_backend

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models import ansatze as A
from mpstab.models.topology import TOPOLOGIES, Topology, coupling_layers

set_backend("numpy")

_PAULIS = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}

# a 3-qubit device, as a provider lists it: each pair in both directions
_COUPLING_MAP = [(0, 1), (1, 0), (1, 2), (2, 1)]


def _two_qubit_pairs(circuit):
    return {tuple(sorted(g.qubits)) for g in circuit.queue if len(g.qubits) == 2}


def _canonical(edges):
    return {tuple(sorted(e)) for e in edges}


def _gates(ansatz):
    """Every gate of the circuit: name, qubits and angles."""
    return [
        (gate.name, gate.qubits, tuple(float(p) for p in gate.parameters))
        for gate in ansatz.circuit.queue
    ]


def _layout(circuit):
    """Every gate of the circuit by name and qubits, without the angles."""
    return [(gate.name, gate.qubits) for gate in circuit.queue]


# Named topologies and validation


@pytest.mark.parametrize(
    "name, nqubits, n_edges, depth, label",
    [
        ("chain", 6, 5, 2, "chain"),
        ("ring", 6, 6, 2, "ring"),
        ("ring", 5, 5, 3, "ring"),
        ("star", 6, 5, 5, "star"),
        ("grid", 6, 7, 3, "grid:2x3"),
        ("grid", 9, 12, 4, "grid:3x3"),
        ("grid", 10, 13, 3, "grid:2x5"),
        ("grid", 12, 17, 4, "grid:3x4"),
        ("grid", 20, 31, 4, "grid:4x5"),
        ("grid:3x6", 18, 27, 4, "grid:3x6"),
        ("grid:1x7", 7, 6, 2, "grid:1x7"),
        ("complete", 4, 6, 3, "complete"),
        ("complete", 7, 21, 7, "complete"),
    ],
)
def test_named_topologies(name, nqubits, n_edges, depth, label):
    topology = Topology.named(name, nqubits)
    assert len(topology) == n_edges
    assert topology.depth == depth
    assert topology.name == label
    assert topology.is_connected()


def test_named_topologies_are_connected():
    for name in TOPOLOGIES:
        for nqubits in range(2, 13):
            if name == "grid" and nqubits in (5, 7, 11):
                continue  # primes above 3 have no default grid shape
            assert Topology.named(name, nqubits).is_connected(), (name, nqubits)


def test_grid_is_row_major_lattice():
    topology = Topology.named("grid", 6)  # 2 x 3
    assert _canonical(topology.edges) == {
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (0, 3),
        (1, 4),
        (2, 5),
    }


@pytest.mark.parametrize(
    "name, nqubits",
    [("grid", 7), ("grid:2x3", 7), ("grid:axb", 6), ("ring:2x3", 6), ("hexagon", 4)],
)
def test_bad_names_and_shapes(name, nqubits):
    with pytest.raises(ValueError):
        Topology.named(name, nqubits)


@pytest.mark.parametrize(
    "edges", [[(0, 0)], [(0, 4)], [(0, 1), (1, 0)], [(0, 1, 2)]], ids=str
)
def test_invalid_edges(edges):
    with pytest.raises(ValueError):
        Topology(4, tuple(edges))


# Graphs, coupling maps and devices


def test_from_graph_inputs_agree():
    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 3))
    mapping = {q: list(grid.neighbors(q)) for q in grid.nodes}
    a = Topology.from_graph(grid)
    b = Topology.from_graph(mapping)
    c = Topology.from_edges(list(grid.edges))
    assert _canonical(a.edges) == _canonical(b.edges) == _canonical(c.edges)
    assert _canonical(a.edges) == _canonical(Topology.named("grid", 6).edges)


def test_from_rustworkx():
    rx = pytest.importorskip("rustworkx")
    graph = rx.PyGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from_no_data([(0, 1), (1, 2), (2, 3)])
    assert _canonical(Topology.from_graph(graph).edges) == {(0, 1), (1, 2), (2, 3)}


def test_from_graph_keeps_isolated_qubits():
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from([(0, 1), (1, 2)])
    assert Topology.from_graph(graph).nqubits == 5
    assert Topology.from_graph({0: [1], 1: [0], 2: []}).nqubits == 3


def test_adjacency_mapping_counts_neighbours_as_nodes():
    assert Topology.from_graph({0: [1]}).edges == ((0, 1),)
    assert Topology.from_graph({0: [1, 3]}).nqubits == 4


def test_coupling_map_drops_directions_duplicates_and_loops():
    topology = Topology.from_device(_COUPLING_MAP + [(2, 2)])
    assert topology.edges == ((0, 1), (1, 2))
    assert topology.nqubits == 3


def test_non_qubit_nodes_are_rejected():
    with pytest.raises(TypeError):
        Topology.from_device([(0, "R"), ("R", 0), (0, 1)])
    with pytest.raises(TypeError):
        Topology.from_graph(nx.Graph([(0, "R"), (0, 1)]))


def test_from_device():
    device = Topology.from_device(_COUPLING_MAP)
    assert device.edges == ((0, 1), (1, 2))
    assert Topology.from_device(device) is device
    assert Topology.from_device("ring", 4) == Topology.named("ring", 4)
    with pytest.raises(ValueError):
        Topology.from_device("ring")


# coerce


def test_coerce():
    topology = Topology.named("ring", 4)
    assert Topology.coerce(topology) is topology
    assert Topology.coerce("ring", 4) == topology
    with pytest.raises(ValueError):
        Topology.coerce("ring")
    widened = Topology.coerce([(0, 1)], nqubits=3)
    assert widened.nqubits == 3 and widened.edges == ((0, 1),)
    with pytest.raises(ValueError):
        Topology.coerce([(0, 3)], nqubits=3)
    with pytest.raises(ValueError):
        Topology.coerce(Topology(6, ((0, 1),)), nqubits=4)


def test_any_sequence_of_pairs_is_read_strictly():
    for edges in (_COUPLING_MAP, tuple(_COUPLING_MAP), np.array(_COUPLING_MAP)):
        with pytest.raises(ValueError):
            Topology.coerce(edges)
    assert Topology.coerce(np.array([(0, 1), (1, 2)])).edges == ((0, 1), (1, 2))


def test_widening_drops_the_name():
    widened = Topology.coerce(Topology.named("grid", 6), 7)
    assert widened.nqubits == 7 and widened.name is None
    assert widened.edges == Topology.named("grid", 6).edges


# Layers and the other properties


def test_layers():
    for name in TOPOLOGIES:
        topology = Topology.named(name, 6)
        assert [list(layer) for layer in topology.layers] == coupling_layers(
            topology.edges
        )
        for layer in topology.layers:
            qubits = [q for bond in layer for q in bond]
            assert len(qubits) == len(set(qubits)), name
    chain = Topology.named("chain", 4)
    assert chain.layers == (((0, 1), (2, 3)), ((1, 2),))
    with pytest.raises(AttributeError):
        chain.layers[0].append((0, 3))


def test_neighbours_degree_and_networkx():
    topology = Topology.named("star", 5)
    graph = topology.to_networkx()
    assert sorted(graph.nodes) == [0, 1, 2, 3, 4] and graph.number_of_edges() == 4
    assert topology.neighbours(0) == [1, 2, 3, 4] and topology.degree(0) == 4
    assert topology.neighbours(3) == [0] and topology.degree(3) == 1


def test_is_subgraph_of():
    device = Topology.named("grid", 6)  # 2 x 3
    assert Topology.named("chain", 3).is_subgraph_of(device)
    assert device.is_subgraph_of(device)
    # order and orientation do not matter
    assert Topology.from_edges([(4, 1), (1, 0)]).is_subgraph_of(device)
    # (0, 2) is not a bond of the lattice
    assert not Topology.from_edges([(0, 2)]).is_subgraph_of(device)
    assert not Topology.named("complete", 6).is_subgraph_of(device)
    # more qubits than the device, even without edges
    assert not Topology(7, ()).is_subgraph_of(device)
    # a coupling map or a name is accepted on the right
    assert Topology.named("chain", 3).is_subgraph_of(_COUPLING_MAP)
    assert Topology.named("chain", 5).is_subgraph_of("ring")


# The graph field of the ansätze

_GRAPHS = [
    "chain",
    "ring",
    "star",
    "grid",
    "complete",
    [(0, 3), (1, 2), (2, 5), (4, 5)],
]


@pytest.mark.parametrize("graph", _GRAPHS, ids=str)
@pytest.mark.parametrize(
    "build",
    [
        lambda g: A.KickedIsing(nqubits=6, n_steps=2, theta_h=0.3, graph=g),
        lambda g: A.ParametricKickedIsing(nqubits=6, n_steps=2, delta_h=0.3, graph=g),
        lambda g: A.RampedKickedIsing(nqubits=6, n_steps=2, delta_h=0.3, graph=g),
        lambda g: A.HardwareEfficient(nqubits=6, nlayers=2, graph=g),
        lambda g: A.HardwareEfficientBrickwork(nqubits=6, nlayers=2, graph=g),
        lambda g: A.HammingWeightPreserving(nqubits=6, nlayers=1, graph=g),
        lambda g: A.FloquetAnsatz(nqubits=6, nlayers=1, graph=g),
        lambda g: A.TrotterIsing(nqubits=6, n_steps=1, graph=g, transpile=False),
    ],
    ids=["KI", "PKI", "RKI", "HE", "HEB", "HWP", "Floquet", "Trotter"],
)
def test_two_qubit_gates_follow_the_graph(build, graph):
    edges = _canonical(Topology.coerce(graph, 6).edges)
    ansatz = build(graph)
    assert _two_qubit_pairs(ansatz.circuit) == edges
    assert _canonical(ansatz.topology.edges) == edges
    assert ansatz.topology.nqubits == 6


def test_gates_follow_the_order_of_the_edges():
    edges = [(0, 2), (1, 3), (0, 1)]
    ansatz = A.RampedKickedIsing(nqubits=4, n_steps=1, graph=edges, delta_h=0.1)
    cnots = [g.qubits for g in ansatz.circuit.queue if g.name == "cx"]
    # layers [(0, 2), (1, 3)] then [(0, 1)]; each RZZ is two CNOTs
    assert cnots == [bond for bond in edges for _ in range(2)]
    assert ansatz.coupling_depth == 2


def test_explicit_edge_lists_are_strict_but_coupling_maps_are_not():
    with pytest.raises(ValueError):
        A.KickedIsing(nqubits=3, graph=_COUPLING_MAP)
    topology = Topology.from_device(_COUPLING_MAP)
    assert A.KickedIsing(nqubits=3, graph=topology).edges == [(0, 1), (1, 2)]


@pytest.mark.parametrize("bond_density", [1.0, 0.6])
def test_star_graph_is_star_kicked_ising(bond_density):
    knobs = dict(
        n_steps=3,
        delta_h=0.4,
        delta_h_last=0.1,
        delta_j=0.2,
        bond_density=bond_density,
        seed=3,
    )
    on_graph = A.RampedKickedIsing(nqubits=6, graph="star", **knobs)
    assert _gates(on_graph) == _gates(A.StarKickedIsing(nqubits=6, **knobs))


def test_graph_is_an_entanglement_parameter():
    for cls in (
        A.KickedIsing,
        A.ParametricKickedIsing,
        A.RampedKickedIsing,
        A.HardwareEfficient,
        A.HardwareEfficientBrickwork,
        A.HammingWeightPreserving,
        A.FloquetAnsatz,
        A.TrotterIsing,
    ):
        assert "graph" in cls.entanglement_parameters, cls.__name__


def test_hamming_weight_preserved_on_any_graph():
    ansatz = A.HammingWeightPreserving(nqubits=4, nlayers=2, graph="complete")
    state = np.asarray(ansatz.circuit.copy()().state())
    weights = np.array([bin(i).count("1") for i in range(16)])
    assert np.allclose(state[weights != 2], 0.0, atol=1e-10)


@pytest.mark.parametrize("graph", ["grid", "complete", "star"])
def test_hsmpo_matches_statevector_on_graphs(graph):
    ansatz = A.ParametricKickedIsing(
        nqubits=6, n_steps=2, delta_h=0.3, delta_j=0.2, graph=graph
    )
    state = np.asarray(ansatz.circuit.copy()().state())
    for pauli in ("ZZIIII", "IXIIIZ", "XYIIZX"):
        operator = reduce(np.kron, [_PAULIS[p] for p in pauli])
        exact = float(np.real(np.vdot(state, operator @ state)))
        assert HSMPO(ansatz=ansatz).expectation(pauli) == pytest.approx(
            exact, abs=1e-10
        )


# The defaults are the circuits from before the graph field, gate for gate.
# The expected layouts below are written from the code on main (ca4ad3b).


@pytest.mark.parametrize("n", range(2, 10))
def test_default_chain_is_the_old_even_odd_pattern(n):
    # the kicked Ising ansätze and FloquetAnsatz couple even bonds, then odd ones
    even = tuple((q, q + 1) for q in range(0, n - 1, 2))
    odd = tuple((q, q + 1) for q in range(1, n - 1, 2))
    assert Topology.named("chain", n).layers == tuple(
        layer for layer in (even, odd) if layer
    )


def test_default_star_is_the_old_hub_leaf_order():
    # StarKickedIsing coupled the hub to each leaf in order, as a single layer
    assert A.StarKickedIsing(nqubits=6).edges == [(0, leaf) for leaf in range(1, 6)]


def test_default_hardware_efficient():
    n, nlayers = 6, 2
    expected = []
    for _ in range(nlayers):
        expected += [("ry", (q,)) for q in range(n)]
        expected += [("cz", (q, (q + 1) % n)) for q in range(n)]
    assert _layout(A.HardwareEfficient(nqubits=n, nlayers=nlayers).circuit) == expected


@pytest.mark.parametrize("n", [5, 6])
def test_default_hardware_efficient_brickwork(n):
    nlayers = 2
    rotations = [("ry", (q,)) for q in range(n)] + [("rz", (q,)) for q in range(n)]
    expected = []
    for _ in range(nlayers):
        expected += rotations
        expected += [("cz", (q, q + 1)) for q in range(0, n - 1, 2)]
        expected += rotations
        expected += [("cz", (q, q + 1)) for q in range(1, n - 2, 2)]
        expected += [("cz", (0, n - 1))]
    expected += [("ry", (q,)) for q in range(n)]
    ansatz = A.HardwareEfficientBrickwork(nqubits=n, nlayers=nlayers)
    assert _layout(ansatz.circuit) == expected


def test_default_hamming_weight_preserving():
    n, nlayers = 6, 2

    def rbs(q0, q1):
        return [
            ("cx", (q1, q0)),
            ("ry", (q1,)),
            ("cx", (q0, q1)),
            ("ry", (q1,)),
            ("cx", (q0, q1)),
            ("cx", (q1, q0)),
        ]

    expected = [("x", (q,)) for q in range(n // 2)]
    for _ in range(nlayers):
        for jump, start in ((1, 0), (1, 1), (2, 0), (2, 1), (2, 3)):
            for q in range(start, n, jump + 1):
                expected += rbs(q, (q + jump) % n)
    ansatz = A.HammingWeightPreserving(nqubits=n, nlayers=nlayers)
    assert _layout(ansatz.circuit) == expected


def test_default_trotter_ising():
    n = 6
    expected = []
    for _ in range(2):
        expected += [("rzz", (j, j + 1)) for j in range(n - 1)]
        expected += [("rx", (j,)) for j in range(n)]
    ansatz = A.TrotterIsing(nqubits=n, n_steps=2, transpile=False)
    assert _layout(ansatz.circuit) == expected


# Ansatz.topology


def test_topology_is_read_off_the_circuit():
    # the historical patterns, with no graph given
    ring = Topology.named("ring", 6)
    assert _canonical(A.HardwareEfficient(nqubits=6).topology.edges) == _canonical(
        ring.edges
    )
    # diluted bonds are missing from the topology, not from the graph
    diluted = A.RampedKickedIsing(
        nqubits=8, n_steps=1, bond_density=0.5, seed=1, graph="grid"
    )
    assert diluted.topology.is_subgraph_of("grid")
    assert len(diluted.topology) < len(diluted.edges)


# connectivity of the transpiled ansätze


_CHAIN = Topology.named("chain", 4)


@pytest.mark.parametrize(
    "connectivity",
    [
        _CHAIN.to_networkx(),
        _CHAIN,
        "chain",
        list(_CHAIN.edges),
        [pair for a, b in _CHAIN.edges for pair in ((a, b), (b, a))],
    ],
    ids=["networkx", "topology", "name", "edges", "coupling-map"],
)
def test_connectivity_accepts_any_description_of_the_device(connectivity):
    # the router places the qubits at random: check what holds every time
    routed = A.QFT(nqubits=4, connectivity=connectivity)
    assert routed.topology.is_subgraph_of(_CHAIN)
    assert len(routed.circuit.queue) > len(A.QFT(nqubits=4).circuit.queue)


def test_unrouted_circuit_does_not_fit_a_chain():
    assert not A.QFT(nqubits=4).topology.is_subgraph_of(_CHAIN)


# Two-qubit fixes


def test_hamming_weight_preserving_on_two_qubits():
    ansatz = A.HammingWeightPreserving(nqubits=2, nlayers=2)
    assert _two_qubit_pairs(ansatz.circuit) == {(0, 1)}
    state = np.asarray(ansatz.circuit.copy()().state())
    assert np.allclose(state[[0, 3]], 0.0, atol=1e-10)  # weight 1 only


def test_hardware_efficient_on_two_qubits_entangles():
    ansatz = A.HardwareEfficient(nqubits=2, nlayers=3)
    assert sum(len(g.qubits) == 2 for g in ansatz.circuit.queue) == 3
    assert ansatz.topology.edges == ((0, 1),)
    assert A.HardwareEfficient(nqubits=1).topology.edges == ()
