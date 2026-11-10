"""
`HSynthSMPO.expectation_at_cut`: the one measurement entry point, over its
shot-based routes ("pauli", "shadows" and "tnice"), against the reference-only
classical `expectation_at_cut(method="simulation")` / `expectation`.

`test_pauli_sampling.py` already checks the Pauli-sampling machinery in
isolation (coefficient extraction, the sampler, QWC grouping); these tests sit
one layer up, at the `HSynthSMPO` integration: that both routes reproduce the
exact value within a few `total_error`, for a Pauli string and for a
`SymbolicHamiltonian`; that the resynthesis Clifford-tail fold is load-bearing
(disabling it breaks the estimate); that a bare object with only
`execute_circuits` works as a backend; and that there is no way to get a value
out of `expectation_at_cut` without spending shots.
"""

import dataclasses
import pathlib

import numpy as np
import pytest
import stim
from qibo import get_backend, set_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Z

from mpstab.evolutors.hsynthsmpo import ExpectationResult, HSynthSMPO, ResynthesizedHead
from mpstab.models.ansatze import HardwareEfficient
from mpstab.quantum_hardware import (
    QiboSimulator,
    count_two_qubit_gates,
    estimate,
    execute_plan,
)
from mpstab.quantum_hardware.synthesis import (
    build_cnot_ladder_head_and_residual,
    head_to_qibo_circuit,
)

set_backend("numpy")

_STIM_1Q = {"H": "H", "S": "S", "Sd": "S_DAG"}
_STIM_2Q = {"CNOT": "CX", "CZ": "CZ"}


def _hs(nqubits, nlayers=2, seed=0, max_bond_dimension=None):
    np.random.seed(seed)
    return HSynthSMPO(
        ansatz=HardwareEfficient(nqubits=nqubits, nlayers=nlayers),
        max_bond_dimension=max_bond_dimension,
    )


def _small_hamiltonian(nqubits):
    form = 0
    for q in range(nqubits):
        form += Z(q) * Z((q + 1) % nqubits) + 0.5 * X(q)
    return SymbolicHamiltonian(form=form, nqubits=nqubits)


@pytest.fixture
def xy_observable():
    """
    A Pauli string that actually exercises the resynthesis Clifford-tail fold.

    HardwareEfficient's entangling layer is CZ, which is diagonal and
    commutes with every all-Z Pauli string, so an all-Z observable would make
    any test of the fold vacuous (folding through a nontrivial-but-Z-preserving
    Clifford leaves an all-Z string, and its expectation value, unchanged). A
    single-Y observable is a different trap for *this* ansatz specifically:
    HardwareEfficient's gates (RY, CZ) are all real, so the prepared state is
    always real-amplitude, and any observable with an odd number of Y's has
    exactly zero expectation on a real state regardless of cut or fold --
    also vacuous, just for a different reason. "XZIZ" (one X, no Y) avoids
    both traps at once.
    """
    return "XZIZ"


def _resynthesis_with_trailing_residual(hs, cut_index, n_trailing=3):
    """
    A rustiq-free :class:`ResynthesizedHead` with a non-trivial residual: the
    CNOT-ladder (exact, identity-residual) decomposition of the head, with its
    last ``n_trailing`` two-qubit Clifford gates carved off and reported as the
    residual, with the tableau built to match exactly.

    The head is no longer the full exact rotation, but head+residual together
    still are -- the same relationship a real rustiq resynthesis has with its
    Clifford residual (Eq. 8). This exists because ``rustiq`` is an optional,
    not-on-PyPI dependency (see ``mpstab.quantum_hardware.synthesis``), and the
    CNOT-ladder fallback alone always has an identity residual, so neither can
    exercise the fold on its own in an environment without rustiq installed.
    """
    dressed = hs._dressed_rotations()[:cut_index]
    head, _, _ = build_cnot_ladder_head_and_residual(
        [pauli for pauli, _ in dressed], [angle for _, angle in dressed]
    )
    split = len(head)
    taken = 0
    while split > 0 and taken < n_trailing and len(head[split - 1]) == 2:
        split -= 1
        taken += 1
    new_head, residual_gates = head[:split], head[split:]
    residual_tableau = stim.Tableau(hs.nqubits)
    for name, qubits in residual_gates:
        gate_name = _STIM_1Q.get(name) or _STIM_2Q.get(name)
        residual_tableau.append(stim.Tableau.from_named_gate(gate_name), qubits)
    return ResynthesizedHead(
        circuit=head_to_qibo_circuit(new_head, hs.nqubits),
        residual_tableau=residual_tableau,
        residual_gates=tuple(residual_gates),
        n_gates=len(new_head),
        n_two_qubit_gates=count_two_qubit_gates(new_head),
        cut_index=cut_index,
        method="cnot_ladder",
    )


class _TrailingResidualHSynthSMPO(HSynthSMPO):
    """
    HSynthSMPO whose resynthesis is rustiq-free but leaves a non-trivial
    residual (see :func:`_resynthesis_with_trailing_residual`). With
    ``drop_residual`` the same head is returned with an identity residual, so
    the device runs the head alone and nothing folds the residual back in.
    """

    drop_residual = False

    def resynthesize_head(self, cut_index, metric_name="count", preserve_order=True):
        resynthesis = _resynthesis_with_trailing_residual(self, cut_index)
        if self.drop_residual:
            resynthesis = dataclasses.replace(
                resynthesis, residual_tableau=stim.Tableau(self.nqubits)
            )
        return resynthesis


def _hs_trailing_residual(nqubits=4, nlayers=3, seed=0):
    np.random.seed(seed)
    return _TrailingResidualHSynthSMPO(
        ansatz=HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
    )


# ---------------------------------------------------------------------------
# ExpectationResult
# ---------------------------------------------------------------------------


def test_expectation_result_is_float_convertible():
    result = ExpectationResult(
        value=0.5,
        stderr=0.01,
        truncation_l1=0.0,
        truncation_l2=0.0,
        n_settings=1,
        n_shots=100,
    )
    assert float(result) == pytest.approx(0.5)
    assert result.total_error == pytest.approx(0.01, abs=1e-9)


def test_unknown_method_raises():
    hs = _hs(3)
    with pytest.raises(ValueError):
        hs.expectation_at_cut("ZZZ", cut_index=0, method="bogus", n_shots=1000)


# ---------------------------------------------------------------------------
# Staged pipeline: plan -> execute_plan -> estimate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pauli", "shadows"])
def test_plan_builds_without_running(method):
    hs = _hs(4)
    plan = hs.plan("ZZZZ", 2, method=method, n_shots=200, seed=0, out_folder=False)
    assert plan.method == method
    assert len(plan.circuits) == len(plan.shots) > 0
    assert sum(plan.shots) == 200
    assert all(circuit.measurements for circuit in plan.circuits)


def test_plan_rejects_simulation():
    hs = _hs(3)
    with pytest.raises(ValueError):
        hs.plan("ZZZ", 0, method="simulation", n_shots=100)


@pytest.mark.parametrize("method", ["pauli", "shadows", "tnice"])
def test_staged_pipeline_matches_expectation_at_cut(method, xy_observable):
    # The three stages by hand give exactly what expectation_at_cut gives in
    # one call, with the same seed and the same simulator seed.
    hs = _hs(4)
    cut_index = 2

    get_backend().set_seed(11)
    one_call = hs.expectation_at_cut(
        xy_observable, cut_index, method=method, n_shots=2000, seed=3
    )

    get_backend().set_seed(11)
    plan = hs.plan(
        xy_observable, cut_index, method=method, n_shots=2000, seed=3,
        out_folder=False,
    )
    frequencies = execute_plan(QiboSimulator(), plan)
    kwargs = {"seed": 3} if method == "tnice" else {}
    staged = estimate(plan, frequencies, **kwargs)

    assert staged.value == pytest.approx(one_call.value, abs=1e-12)
    assert staged.stderr == pytest.approx(one_call.stderr, abs=1e-12)


# ---------------------------------------------------------------------------
# 1. Both routes match method="simulation" / expectation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut_fraction", [0.0, 0.5, 1.0])
def test_both_routes_match_simulation_for_pauli_string(cut_fraction, xy_observable):
    hs = _hs(4, nlayers=2)
    n_dressed = len(hs.magic_gates)
    cut_index = int(round(cut_fraction * n_dressed))
    exact = hs.expectation_at_cut(xy_observable, cut_index, method="simulation").value

    pauli = hs.expectation_at_cut(
        xy_observable, cut_index, method="pauli", n_shots=150000, seed=1
    )
    # shots_per_setting stays at its default of 1: classical-shadow
    # unbiasedness needs a fresh random basis per shot, so a large value would
    # make the reported stderr optimistic rather than just faster to run.
    # tail_handling="append_to_head_circuit" (the default) because with rustiq
    # installed the resynthesis leaves a non-trivial Clifford residual at most
    # cuts, and appending it keeps the shadow variance of the unfolded tail.
    shadows = hs.expectation_at_cut(
        xy_observable,
        cut_index,
        method="shadows",
        n_shots=30000,
        seed=2,
        tail_handling="append_to_head_circuit",
    )

    assert abs(pauli.value - exact) <= 4 * pauli.total_error + 1e-6, pauli
    assert abs(shadows.value - exact) <= 4 * shadows.total_error + 1e-6, shadows


@pytest.mark.parametrize("cut_fraction", [0.0, 0.5, 1.0])
def test_both_routes_match_hsmpo_for_symbolic_hamiltonian(cut_fraction):
    hs = _hs(4, nlayers=2)
    hamiltonian = _small_hamiltonian(4)
    n_dressed = len(hs.magic_gates)
    cut_index = int(round(cut_fraction * n_dressed))
    exact = hs.expectation(hamiltonian)

    pauli = hs.expectation_at_cut(
        hamiltonian, cut_index, method="pauli", n_shots=200000, seed=3
    )
    shadows = hs.expectation_at_cut(
        hamiltonian,
        cut_index,
        method="shadows",
        n_shots=15000,
        seed=4,
        tail_handling="append_to_head_circuit",
    )

    assert abs(pauli.value - exact) <= 4 * pauli.total_error + 1e-3, pauli
    assert abs(shadows.value - exact) <= 4 * shadows.total_error + 1e-3, shadows


@pytest.mark.parametrize("cut_fraction", [0.0, 0.5, 1.0])
def test_tnice_matches_simulation_for_pauli_string(cut_fraction, xy_observable):
    """
    ``"tnice"`` shares its circuits and shots with ``"shadows"`` (same seed,
    same call shape) -- see ``HSynthSMPO._shadow_plan`` -- and only replaces
    the fixed canonical-dual post-processing with the MPS-optimized
    estimator, so the two routes are compared here on identical measurement
    data.
    """
    hs = _hs(4, nlayers=2)
    n_dressed = len(hs.magic_gates)
    cut_index = int(round(cut_fraction * n_dressed))
    exact = hs.expectation_at_cut(xy_observable, cut_index, method="simulation").value

    result = hs.expectation_at_cut(
        xy_observable,
        cut_index,
        method="tnice",
        n_shots=30000,
        seed=2,
        tail_handling="append_to_head_circuit",
    )
    if abs(result.value - exact) > 4 * result.total_error + 1e-6:
        # Seen failing only on the Linux CI. Report what is needed to tell an
        # estimator bug from a platform effect: the same measurement data through
        # the fixed canonical-dual ("shadows") post-processing, and the BLAS.
        shadows = hs.expectation_at_cut(
            xy_observable,
            cut_index,
            method="shadows",
            n_shots=30000,
            seed=2,
            tail_handling="append_to_head_circuit",
        )
        # Same shots, other tnice settings: 0 sweeps is the unbiased canonical-dual
        # seed, so it separates "the optimization breaks it" from the data.
        variants = {}
        for label, kwargs in (
            ("sweeps=0", {"n_sweeps": 0}),
            ("sweeps=1", {"n_sweeps": 1}),
            ("sweeps=2", {"n_sweeps": 2}),
            ("lam=0.9999", {"lam": 0.9999}),
        ):
            variants[label] = hs.expectation_at_cut(
                xy_observable,
                cut_index,
                method="tnice",
                n_shots=30000,
                seed=2,
                tail_handling="append_to_head_circuit",
                **kwargs,
            ).value
        try:
            blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        except Exception:  # pragma: no cover - diagnostics only
            blas = "unknown"
        pytest.fail(
            f"tnice {result}\nexact {exact:+.6f}\nshadows {shadows}\n"
            f"tnice variants {variants}\nblas {blas}"
        )


def test_tnice_appended_nontrivial_tail_matches_simulation(xy_observable):
    hs = _hs_trailing_residual()
    cut_index = len(hs.magic_gates) - 3

    exact = hs.expectation_at_cut(xy_observable, cut_index, method="simulation").value
    result = hs.expectation_at_cut(
        xy_observable, cut_index, method="tnice", n_shots=30000, seed=1
    )
    assert abs(result.value - exact) <= 4 * result.total_error, result


# ---------------------------------------------------------------------------
# 2. Clifford-tail fold correctness -- the important one
# ---------------------------------------------------------------------------


def test_clifford_tail_fold_matches_simulation_and_breaks_if_disabled(xy_observable):
    hs = _hs_trailing_residual()
    cut_index = len(hs.magic_gates) - 3

    exact = hs.expectation_at_cut(xy_observable, cut_index, method="simulation").value
    result = hs.expectation_at_cut(
        xy_observable, cut_index, method="pauli", n_shots=300000, seed=7
    )
    assert (
        abs(exact) > 0.1
    ), "test fixture must give a non-trivial exact value, see xy_observable"
    assert abs(result.value - exact) <= 4 * result.total_error, result

    # Disable the fold: same head, identity residual, so the sampled strings are
    # not folded back through the gates the head left out.
    hs.drop_residual = True
    broken = hs.expectation_at_cut(
        xy_observable, cut_index, method="pauli", n_shots=300000, seed=7
    )
    assert abs(broken.value - exact) > 4 * broken.total_error, (
        "disabling the Clifford-tail fold should produce a value that misses "
        "the exact reference by more than its own error bars -- if it "
        "doesn't, this test is no longer exercising the fold"
    )


def test_shadows_appended_nontrivial_tail_matches_simulation(xy_observable):
    hs = _hs_trailing_residual()
    cut_index = len(hs.magic_gates) - 3

    exact = hs.expectation_at_cut(xy_observable, cut_index, method="simulation").value
    result = hs.expectation_at_cut(
        xy_observable, cut_index, method="shadows", n_shots=30000, seed=1
    )
    assert abs(result.value - exact) <= 4 * result.total_error, result


def test_post_head_mpo_with_residual_matches_appended_residual(xy_observable):
    """
    ``build_post_head_mpo(residual=R)`` on the head circuit alone must equal the
    plain post-head MPO on head + residual gates, exactly (dense statevector
    check).
    """
    hs = _hs_trailing_residual()
    cut_index = len(hs.magic_gates) - 3
    resynthesis = hs.resynthesize_head(cut_index)
    assert resynthesis.residual_tableau != stim.Tableau(4)

    def dense_expval(circuit, operator, sign):
        state = np.asarray(circuit().state())
        return sign * float(np.real(state.conj() @ (operator.to_dense() @ state)))

    tail, sign = hs.build_post_head_mpo(xy_observable, cut_index, None)
    folded, folded_sign = hs.build_post_head_mpo(
        xy_observable, cut_index, None, residual=resynthesis.residual_tableau
    )
    appended = resynthesis.circuit + head_to_qibo_circuit(
        list(resynthesis.residual_gates), 4
    )
    exact = hs.expectation_at_cut(xy_observable, cut_index, method="simulation").value

    assert dense_expval(appended, tail, sign) == pytest.approx(exact, abs=1e-10)
    assert dense_expval(resynthesis.circuit, folded, folded_sign) == pytest.approx(
        exact, abs=1e-10
    )
    # Dropping the residual must not give the same number, or the test is vacuous.
    assert abs(dense_expval(resynthesis.circuit, tail, sign) - exact) > 1e-3


@pytest.mark.parametrize("method", ["shadows", "tnice"])
def test_fold_tail_handling_matches_simulation(xy_observable, method):
    hs = _hs_trailing_residual()
    cut_index = len(hs.magic_gates) - 3
    resynthesis = hs.resynthesize_head(cut_index)

    exact = hs.expectation_at_cut(xy_observable, cut_index, method="simulation").value
    result = hs.expectation_at_cut(
        xy_observable,
        cut_index,
        method=method,
        n_shots=30000,
        seed=1,
        tail_handling="fold",
    )
    assert abs(result.value - exact) <= 4 * result.total_error, result

    # Same seed, same bases: "fold" measures exactly the residual's gates fewer.
    def first_circuit(tail_handling):
        plan = hs._shadow_plan(
            terms={xy_observable: 1.0},
            constant=0.0,
            cut_index=cut_index,
            resynthesis=resynthesis,
            max_bond_dimension=None,
            n_shots=10,
            epsilon=None,
            seed=0,
            tail_handling=tail_handling,
        )
        return plan.circuits[0]

    n_residual = len(resynthesis.residual_gates)
    assert n_residual > 0
    assert (
        len(first_circuit("append_to_head_circuit").queue)
        - len(first_circuit("fold").queue)
        == n_residual
    )


def test_unknown_tail_handling_raises():
    hs = _hs(3)
    with pytest.raises(ValueError):
        hs.expectation_at_cut(
            "ZZZ", cut_index=0, method="shadows", n_shots=100, tail_handling="bogus"
        )


def test_tail_truncation_with_residual():
    hs = _hs(3)
    residual = hs.resynthesize_head(2).residual_tableau
    info = hs.tail_truncation("ZZZ", 0, residual=residual)
    assert info.fidelity_estimate == pytest.approx(1.0)
    assert info.relative_frobenius_error == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(NotImplementedError):
        hs.tail_truncation("ZZZ", 0, residual=residual, with_expval=True)


def test_shadow_truncation_l2_uses_requested_bond():
    # Regression: the estimate used self.max_bond_dimension, not the bond the
    # plan's MPOs were actually folded with.
    hs = _hs(6)  # untruncated by default
    resynthesis = hs.resynthesize_head(0)

    def truncation_l2(max_bond_dimension):
        return hs._shadow_plan(
            terms={"Z" * 6: 1.0},
            constant=0.0,
            cut_index=0,
            resynthesis=resynthesis,
            max_bond_dimension=max_bond_dimension,
            n_shots=10,
            epsilon=None,
            seed=0,
            tail_handling="append_to_head_circuit",
        ).truncation_l2

    assert truncation_l2(None) == 0.0
    assert truncation_l2(1) > 1e-6


# ---------------------------------------------------------------------------
# 3. A bare object exposing only execute_circuits works as a backend
# ---------------------------------------------------------------------------


def test_bare_object_with_execute_circuits_works_as_backend():
    class BareBackend:
        def execute_circuits(self, circuits, nshots):
            return [circuit(nshots=nshots).frequencies() for circuit in circuits]

    hs = _hs(4, nlayers=2)
    cut_index = len(hs.magic_gates) // 2
    result = hs.expectation_at_cut(
        "XZIZ", cut_index, method="pauli", n_shots=20000, seed=9, backend=BareBackend()
    )
    assert isinstance(result.value, float)
    assert result.n_shots > 0


def test_execute_plan_rejects_a_backend_that_drops_results():
    class DroppingBackend:
        def execute_circuits(self, circuits, nshots):
            return [circuit(nshots=nshots).frequencies() for circuit in circuits[1:]]

    hs = _hs(4)
    plan = hs.plan(
        "ZZZZ", 2, method="shadows", n_shots=20, seed=0, out_folder=False
    )
    with pytest.raises(ValueError, match="DroppingBackend"):
        execute_plan(DroppingBackend(), plan)


# ---------------------------------------------------------------------------
# 4. No route yields a value without spending shots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pauli", "shadows", "tnice"])
def test_expectation_at_cut_requires_exactly_one_of_shots_or_epsilon(method):
    hs = _hs(3)
    with pytest.raises(ValueError):
        hs.expectation_at_cut("ZZZ", cut_index=1, method=method)
    with pytest.raises(ValueError):
        hs.expectation_at_cut(
            "ZZZ", cut_index=1, method=method, n_shots=100, epsilon=0.1
        )


def test_epsilon_sizes_a_nonzero_shot_budget():
    hs = _hs(4, nlayers=2)
    cut_index = len(hs.magic_gates) // 2
    result = hs.expectation_at_cut(
        "XZIZ", cut_index, method="pauli", epsilon=0.05, seed=1
    )
    assert result.n_shots > 0


# ---------------------------------------------------------------------------
# 5. No orphan imports of the deleted modules
# ---------------------------------------------------------------------------


def test_no_orphan_imports_of_deleted_modules():

    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    forbidden = ("mpstab.measurements", "mpstab.models.noise", "models.noise")
    offenders = [
        (str(path), token)
        for path in src.rglob("*.py")
        for token in forbidden
        if token in path.read_text()
    ]
    assert not offenders, offenders
