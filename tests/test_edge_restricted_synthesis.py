"""
Tests for the coupler-restricted head syntheses (``steiner``, ``edge_greedy``).

Every head must, with its residual, be the product of the rotations, checked
both on the state and at the Pauli level (:func:`verify_head` below, which
needs no state vector); every two-qubit gate must sit on a coupler; and the
expectation value read from these heads with the folded tail must be the exact
one.
"""

import copy
import math

import numpy as np
import pytest
import stim

from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import HardwareEfficient, ParametricKickedIsing
from mpstab.qpu.synthesis import (
    CLIFFORD_TABLEAUS,
    EdgeGreedySynthesis,
    ExtractionSynthesis,
    SteinerSynthesis,
    build_cnot_ladder_head_and_residual,
    build_edge_greedy_head_and_residual,
    build_steiner_head_and_residual,
    head_to_qibo_circuit,
)
from mpstab.qpu.synthesis.coupling import CouplingGraph

BUILDERS = {
    "steiner": build_steiner_head_and_residual,
    "edge_greedy": build_edge_greedy_head_and_residual,
}
CONNECTIVITIES = [None, "line", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (2, 5)]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _commute(a, b):
    return sum(x != "I" and y != "I" and x != y for x, y in zip(a, b)) % 2 == 0


def effective_rotations(head, nqubits):
    """Every rotation of ``head`` pushed through the Cliffords before it,
    ``exp(-i a Q / 2) C = C exp(-i a C^dag Q C / 2)``, so ``head = clifford . prod``."""
    clifford = stim.Tableau(nqubits)
    rotations = []
    for entry in head:
        if len(entry) == 3:
            axis, qubits, angle = entry
            local = stim.PauliString(nqubits)
            local[qubits[0]] = axis[1]
            generator = clifford.inverse()(local)
            text = str(generator)[1:].replace("_", "I")
            rotations.append((text, generator.sign.real * float(angle)))
        else:
            clifford.append(CLIFFORD_TABLEAUS[entry[0]], list(entry[1]))
    return rotations, clifford


def verify_head(head, residual_tableau, paulis, angles, atol=1e-9):
    """Pauli-level check, any size: the residual undoes the head's Clifford and
    the effective rotations are the given ones, in order up to swaps of
    commuting neighbours."""
    nqubits = len(paulis[0])
    rotations, clifford = effective_rotations(head, nqubits)
    assert clifford.then(residual_tableau) == stim.Tableau(nqubits)
    assert len(rotations) == len(paulis)
    remaining = list(range(len(paulis)))
    for pauli, angle in rotations:
        for slot, j in enumerate(remaining):
            if paulis[j] == pauli:
                break
            assert _commute(paulis[j], pauli), f"{pauli} moved past {paulis[j]}"
        else:
            raise AssertionError(f"{pauli} is not among the rotations")
        assert abs(math.remainder(angle - angles[j], 4 * math.pi)) < atol
        remaining.pop(slot)


def _random_rotations(nqubits, n_rotations, seed):
    rng = np.random.default_rng(seed)
    paulis = []
    while len(paulis) < n_rotations:
        pauli = "".join(rng.choice(list("IXYZ"), size=nqubits, p=[0.4, 0.2, 0.2, 0.2]))
        if pauli != "I" * nqubits:
            paulis.append(pauli)
    return paulis, [float(a) for a in rng.uniform(-np.pi, np.pi, size=n_rotations)]


def _state(gates, nqubits):
    return np.asarray(head_to_qibo_circuit(gates, nqubits)().state())


def _overlap_with_rotations(paulis, angles, head, residual_gates):
    nqubits = len(paulis[0])
    exact, _, _ = build_cnot_ladder_head_and_residual(paulis, angles)
    return abs(np.vdot(_state(exact, nqubits), _state(head + residual_gates, nqubits)))


# ---------------------------------------------------------------------------
# The heads are the rotations, on the couplers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(BUILDERS))
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("connectivity", CONNECTIVITIES)
def test_head_and_residual_are_the_rotations_on_the_couplers(method, seed, connectivity):
    paulis, angles = _random_rotations(6, 14, seed)
    head, tableau, residual = BUILDERS[method](paulis, angles, connectivity=connectivity)
    assert _overlap_with_rotations(paulis, angles, head, residual) == pytest.approx(1.0, abs=1e-9)
    verify_head(head, tableau, paulis, angles)
    edges = set(CouplingGraph(connectivity, 6).edges)
    assert all(tuple(sorted(g[1])) in edges for g in head if len(g) == 2 and len(g[1]) == 2)
    rebuilt = stim.Tableau(6)
    for name, qubits in residual:
        rebuilt.append(CLIFFORD_TABLEAUS[name], qubits)
    assert rebuilt == tableau


@pytest.mark.parametrize("method", list(BUILDERS))
def test_deterministic_and_empty(method):
    paulis, angles = _random_rotations(6, 10, seed=4)
    first = BUILDERS[method](paulis, angles, connectivity="line")
    second = BUILDERS[method](paulis, angles, connectivity="line")
    assert first[0] == second[0] and first[2] == second[2]
    assert BUILDERS[method]([], []) == ([], stim.Tableau(0), [])


def test_steiner_tree_joins_the_terminals_on_couplers():
    graph = CouplingGraph([(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6)], 7)
    tree = graph.steiner_tree([0, 4, 6])
    assert {0, 4, 6} <= set(tree)
    edges = {tuple(sorted((a, b))) for a, nbs in tree.items() for b in nbs}
    assert len(edges) == len(tree) - 1  # a tree
    assert edges <= set(graph.edges)
    assert graph.tree_cost([0, 4, 6]) >= len(edges) / 2  # the MST bound is within 2x


def test_unknown_connectivity_raises():
    with pytest.raises(ValueError, match="Unknown connectivity"):
        SteinerSynthesis(connectivity="ring").synthesize(["ZZ"], [0.1])
    with pytest.raises(ValueError, match="not connected"):
        build_edge_greedy_head_and_residual(["ZIZ"], [0.1], connectivity=[(0, 1)])


def test_on_the_line_the_couplers_beat_extraction_routed():
    """A parametric kicked Ising head on a line: the greedy head needs no
    routing, and its routed cost is below extraction's."""
    hs = HSynthSMPO.rotations_only(ParametricKickedIsing(nqubits=8, n_steps=4, delta_h=0.3))
    cut_index = len(hs.magic_gates)
    greedy = hs.resynthesize_head(cut_index, synthesis=EdgeGreedySynthesis(connectivity="line"))
    extraction = hs.resynthesize_head(cut_index, synthesis=ExtractionSynthesis(connectivity="line"))
    cost = lambda circuit: sum(  # noqa: E731
        1 + 3 * (abs(g.qubits[0] - g.qubits[1]) - 1) for g in circuit.queue if len(g.qubits) == 2
    )
    assert all(abs(g.qubits[0] - g.qubits[1]) == 1 for g in greedy.circuit.queue if len(g.qubits) == 2)
    assert cost(greedy.circuit) < cost(extraction.circuit)


def test_registered_by_name():
    hs = HSynthSMPO.rotations_only(ParametricKickedIsing(nqubits=5, n_steps=2, delta_h=0.2))
    cut_index = len(hs.magic_gates)
    assert hs.resynthesize_head(cut_index, synthesis="steiner").method == "steiner"
    assert hs.resynthesize_head(cut_index, synthesis="edge_greedy").method == "edge_greedy"


# ---------------------------------------------------------------------------
# The Pauli-level check itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["extraction", "cnot_ladder", "steiner", "edge_greedy", "rustiq"])
def test_verify_head_accepts_every_synthesis(method):
    if method == "rustiq":
        pytest.importorskip("rustiq")
    from mpstab.qpu.synthesis import resolve_synthesis

    paulis, angles = _random_rotations(6, 16, seed=11)
    head, tableau, _ = resolve_synthesis(method).synthesize(paulis, angles)
    verify_head(head, tableau, paulis, angles)


def test_verify_head_catches_a_wrong_head():
    paulis, angles = _random_rotations(5, 10, seed=2)
    head, tableau, _ = build_steiner_head_and_residual(paulis, angles, connectivity="line")
    i = next(k for k, g in enumerate(head) if len(g) == 3)
    wrong_angle = copy.deepcopy(head)
    wrong_angle[i] = (*wrong_angle[i][:2], wrong_angle[i][2] + 0.1)
    j = next(k for k, g in enumerate(head) if g[0] == "CNOT")
    no_cnot = head[:j] + head[j + 1:]
    for wrong in (wrong_angle, no_cnot):
        with pytest.raises(AssertionError):
            verify_head(wrong, tableau, paulis, angles)


# ---------------------------------------------------------------------------
# Expectation values from these heads
# ---------------------------------------------------------------------------


def _dense_expval(circuit, operator, sign):
    state = np.asarray(circuit().state())
    return sign * float(np.real(state.conj() @ (operator.to_dense() @ state)))


@pytest.mark.parametrize(
    "synthesis", [SteinerSynthesis(connectivity="line"), EdgeGreedySynthesis(connectivity="line")]
)
@pytest.mark.parametrize("cut_fraction", [0.3, 0.6, 1.0])
def test_folded_observable_on_the_head_is_exact(synthesis, cut_fraction):
    np.random.seed(0)
    hs = HSynthSMPO(ansatz=HardwareEfficient(nqubits=4, nlayers=3), head_synthesis=synthesis)
    cut_index = int(round(cut_fraction * len(hs.magic_gates)))
    head = hs.resynthesize_head(cut_index)
    for observable in ("XZIZ", "ZZII", "IYYI"):
        folded, sign = hs.build_post_head_mpo(observable, cut_index, None, residual=head.residual_tableau)
        assert _dense_expval(head.circuit, folded, sign) == pytest.approx(
            hs.expectation(observable), abs=1e-9
        )
