"""
Tests for the Clifford-extraction head synthesis
(:func:`mpstab.quantum_hardware.synthesis.build_extraction_head_and_residual`).

The head and its residual must together be the product of the rotations, on
random Pauli strings and for every strategy and connectivity; and the
expectation values measured from an extraction head must agree with the
classical contraction of the same split, as they do for the other syntheses.
"""

import numpy as np
import pytest
import stim

from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import HardwareEfficient, ParametricKickedIsing
from mpstab.quantum_hardware.synthesis import (
    CLIFFORD_TABLEAUS,
    EXTRACTION_STRATEGIES,
    build_cnot_ladder_head_and_residual,
    build_extraction_head_and_residual,
    connectivity_distances,
    count_two_qubit_gates,
    head_to_qibo_circuit,
)


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
    """|<exact product of the rotations | residual . head>| on |0...0>."""
    nqubits = len(paulis[0])
    exact, _, _ = build_cnot_ladder_head_and_residual(paulis, angles)
    return abs(np.vdot(_state(exact, nqubits), _state(head + residual_gates, nqubits)))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "connectivity", [None, "line", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (2, 5)]]
)
def test_head_and_residual_are_the_rotations(seed, connectivity):
    paulis, angles = _random_rotations(6, 14, seed)
    head, tableau, residual = build_extraction_head_and_residual(
        paulis, angles, connectivity=connectivity
    )
    assert _overlap_with_rotations(paulis, angles, head, residual) == pytest.approx(
        1.0, abs=1e-9
    )
    # the tableau is the residual gate list
    rebuilt = stim.Tableau(6)
    for name, qubits in residual:
        rebuilt.append(CLIFFORD_TABLEAUS[name], qubits)
    assert rebuilt == tableau


@pytest.mark.parametrize("strategy", EXTRACTION_STRATEGIES)
def test_every_strategy_is_exact(strategy):
    paulis, angles = _random_rotations(5, 12, seed=7)
    head, _, residual = build_extraction_head_and_residual(
        paulis, angles, connectivity="line", strategies=(strategy,)
    )
    assert _overlap_with_rotations(paulis, angles, head, residual) == pytest.approx(
        1.0, abs=1e-9
    )


def test_one_rotation_per_pauli_and_fewer_cnots_than_the_ladder():
    paulis, angles = _random_rotations(6, 20, seed=3)
    head, _, _ = build_extraction_head_and_residual(paulis, angles)
    ladder, _, _ = build_cnot_ladder_head_and_residual(paulis, angles)
    assert sum(1 for gate in head if len(gate) == 3) == len(paulis)
    # w - 1 CNOTs per weight-w rotation instead of 2 * (w - 1)
    assert count_two_qubit_gates(head) <= count_two_qubit_gates(ladder) // 2


def test_synthesis_is_deterministic():
    paulis, angles = _random_rotations(6, 14, seed=1)
    first = build_extraction_head_and_residual(paulis, angles, connectivity="line")
    second = build_extraction_head_and_residual(paulis, angles, connectivity="line")
    assert first[0] == second[0] and first[2] == second[2]


def test_empty_head():
    assert build_extraction_head_and_residual([], []) == ([], stim.Tableau(0), [])


def test_connectivity_distances():
    assert connectivity_distances(None, 3) is None
    assert connectivity_distances("line", 4)[0] == [0, 1, 2, 3]
    assert connectivity_distances([(0, 1), (1, 2), (2, 0)], 3)[0] == [0, 1, 1]
    with pytest.raises(ValueError, match="not connected"):
        connectivity_distances([(0, 1)], 3)
    with pytest.raises(ValueError, match="Unknown connectivity"):
        connectivity_distances("ring", 3)


def _routed_cost(circuit):
    """Two-qubit gates on a line: a CNOT at distance d costs d - 1 SWAPs more."""
    return sum(
        1 + 3 * (abs(g.qubits[0] - g.qubits[1]) - 1)
        for g in circuit.queue
        if len(g.qubits) == 2
    )


def test_line_connectivity_never_costs_more_on_the_line():
    # the strategies tried without a connectivity are also tried with one
    hs = HSynthSMPO.rotations_only(
        ParametricKickedIsing(nqubits=8, n_steps=4, delta_h=0.3)
    )
    cut_index = len(hs.magic_gates)
    local = hs.resynthesize_head(cut_index, method="extraction", connectivity="line")
    free = hs.resynthesize_head(cut_index, method="extraction")
    assert local.method == "extraction"
    assert _routed_cost(local.circuit) <= _routed_cost(free.circuit)


def test_resynthesize_head_method_selection():
    np.random.seed(0)
    hs = HSynthSMPO(
        ansatz=HardwareEfficient(nqubits=4, nlayers=2), head_synthesis="extraction"
    )
    cut_index = len(hs.magic_gates)
    assert hs.resynthesize_head(cut_index).method == "extraction"
    assert hs.resynthesize_head(cut_index, method="cnot_ladder").method == "cnot_ladder"
    assert hs.resynthesize_head(cut_index) is hs.resynthesize_head(cut_index)  # cached
    with pytest.raises(ValueError, match="Unknown head synthesis"):
        hs.resynthesize_head(cut_index, method="magic")


# ---------------------------------------------------------------------------
# Expectation values from an extraction head
# ---------------------------------------------------------------------------


def _dense_expval(circuit, operator, sign):
    state = np.asarray(circuit().state())
    return sign * float(np.real(state.conj() @ (operator.to_dense() @ state)))


@pytest.mark.parametrize("connectivity", [None, "line"])
@pytest.mark.parametrize("cut_fraction", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("observable", ["XZIZ", "ZZII", "IYYI"])
def test_folded_observable_on_the_head_is_exact(connectivity, cut_fraction, observable):
    """No shots: <head| folded tail-and-residual observable |head> is the exact value."""
    np.random.seed(0)
    hs = HSynthSMPO(
        ansatz=HardwareEfficient(nqubits=4, nlayers=3),
        head_synthesis="extraction",
        head_connectivity=connectivity,
    )
    cut_index = int(round(cut_fraction * len(hs.magic_gates)))
    head = hs.resynthesize_head(cut_index)
    exact = hs.expectation(observable)
    folded, sign = hs.build_post_head_mpo(
        observable, cut_index, None, residual=head.residual_tableau
    )
    assert _dense_expval(head.circuit, folded, sign) == pytest.approx(exact, abs=1e-9)


@pytest.mark.parametrize("delta_h", [0.0, 0.2])
def test_folded_observable_is_exact_on_a_floquet_circuit(delta_h):
    hs = HSynthSMPO(
        ansatz=ParametricKickedIsing(
            nqubits=5, n_steps=3, delta_h=delta_h, delta_z=0.3
        ),
        head_synthesis="extraction",
        head_connectivity="line",
    )
    cut_index = len(hs.magic_gates) // 2
    head = hs.resynthesize_head(cut_index)
    for observable in ("IIZII", "ZZIII", "XIIZI"):
        folded, sign = hs.build_post_head_mpo(
            observable, cut_index, None, residual=head.residual_tableau
        )
        assert _dense_expval(head.circuit, folded, sign) == pytest.approx(
            hs.expectation(observable), abs=1e-9
        )


@pytest.mark.parametrize("cut_fraction", [0.5, 1.0])
def test_measured_expectation_matches_simulation(cut_fraction):
    """The whole hardware path, with shots, from an extraction head."""
    np.random.seed(0)
    hs = HSynthSMPO(
        ansatz=HardwareEfficient(nqubits=4, nlayers=2),
        head_synthesis="extraction",
        head_connectivity="line",
    )
    cut_index = int(round(cut_fraction * len(hs.magic_gates)))
    exact = hs.expectation_at_cut("XZIZ", cut_index, method="simulation").value
    pauli = hs.expectation_at_cut(
        "XZIZ", cut_index, method="pauli", n_shots=150000, seed=1
    )
    shadows = hs.expectation_at_cut(
        "XZIZ",
        cut_index,
        method="shadows",
        n_shots=30000,
        seed=2,
        tail_handling="append_to_head_circuit",
    )
    assert abs(pauli.value - exact) <= 4 * pauli.total_error + 1e-6, pauli
    assert abs(shadows.value - exact) <= 4 * shadows.total_error + 1e-6, shadows


# ---------------------------------------------------------------------------
# Light cone: the rotations the observable cannot see
# ---------------------------------------------------------------------------


def _floquet(light_cone, nqubits=8, n_steps=2, **options):
    return HSynthSMPO(
        ansatz=ParametricKickedIsing(
            nqubits=nqubits, n_steps=n_steps, delta_h=0.2, delta_z=0.3
        ),
        head_synthesis="extraction",
        head_connectivity="line",
        light_cone=light_cone,
        **options,
    )


def test_relevant_rotations_are_a_strict_subset_for_a_local_observable():
    hs = _floquet(True)
    n_rotations = len(hs.magic_gates)
    local = hs.relevant_rotations("IIIZIIII")
    assert 0 < len(local) < n_rotations
    assert list(local) == sorted(local)
    # several terms: the union of the terms' light cones
    both = hs.relevant_rotations({"IIIZIIII": 1.0, "ZIIIIIII": 0.5})
    assert set(local) <= set(both)
    assert set(hs.relevant_rotations("ZIIIIIII")) <= set(both)


@pytest.mark.parametrize("cut_fraction", [0.4, 0.7, 1.0])
@pytest.mark.parametrize("observable", ["IIIZIIII", "ZZIIIIII", "IIXIIZII"])
def test_light_cone_head_gives_the_exact_value(cut_fraction, observable):
    """Dropping the rotations outside the light cone changes nothing, with fewer gates."""
    pruned, full = _floquet(True), _floquet(False)
    cut_index = int(round(cut_fraction * len(full.magic_gates)))
    head = pruned.resynthesize_head(cut_index, observable=observable)
    whole = full.resynthesize_head(cut_index)
    rotations = lambda circuit: sum(
        1 for g in circuit.queue if g.name == "rz"
    )  # noqa: E731
    assert rotations(head.circuit) <= rotations(whole.circuit)

    folded, sign = pruned.build_post_head_mpo(
        observable, cut_index, None, residual=head.residual_tableau
    )
    assert _dense_expval(head.circuit, folded, sign) == pytest.approx(
        full.expectation(observable), abs=1e-9
    )


def test_light_cone_removes_rotations_from_a_shallow_local_head():
    pruned, full = _floquet(True), _floquet(False)
    cut_index = len(full.magic_gates)
    head = pruned.resynthesize_head(cut_index, observable="IIIZIIII")
    whole = full.resynthesize_head(cut_index)
    assert head.n_two_qubit_gates < whole.n_two_qubit_gates


def test_light_cone_measured_expectation_matches_simulation():
    """The hardware routes with the light cone on, for a sum of local terms."""
    hs = _floquet(True, nqubits=5)
    observable = {"IIZII": 1.0, "ZZIII": -0.5}
    cut_index = len(hs.magic_gates) // 2
    exact = hs.expectation(observable)
    assert hs.expectation_at_cut(
        observable, cut_index, method="simulation"
    ).value == pytest.approx(exact, abs=1e-9)
    pauli = hs.expectation_at_cut(
        observable, cut_index, method="pauli", n_shots=150000, seed=1
    )
    assert abs(pauli.value - exact) <= 4 * pauli.total_error + 1e-6, pauli
