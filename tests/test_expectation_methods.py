"""
`HSynthSMPO.expectation_at_cut`: the one measurement entry point, over its two
shot-based routes ("pauli" and "shadows"), against the reference-only exact
`expectation_from_split` / `expectation`.

`test_pauli_sampling.py` already checks the Pauli-sampling machinery in
isolation (coefficient extraction, the sampler, QWC grouping); these tests sit
one layer up, at the `HSynthSMPO` integration: that both routes reproduce the
exact value within a few `total_error`, for a Pauli string and for a
`SymbolicHamiltonian`; that the resynthesis Clifford-tail fold is load-bearing
(disabling it breaks the estimate); that a bare object with only
`execute_circuits` works as a backend; and that there is no way to get a value
out of `expectation_at_cut` without spending shots.
"""

import numpy as np
import pytest
import stim
from qibo import set_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Z

from mpstab.evolutors.hsynthsmpo import ExpectationResult, HSynthSMPO
from mpstab.models.ansatze import HardwareEfficient
from mpstab.quantum_hardware.synthesis import build_naive_head_and_residual

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


def _fake_resynthesis_with_trailing_tail(n_trailing=3):
    """
    A rustiq-free stand-in for ``build_head_and_residual`` that carves the
    last ``n_trailing`` pure-Clifford gates off the naive (exact,
    identity-tail) decomposition and reports them as a synthetic Clifford
    tail, with the tableau built to match exactly.

    The head is no longer the full exact rotation, but head+tail together
    still are -- the same relationship a real rustiq resynthesis has with its
    Clifford residual (Eq. 8). This exists because ``rustiq`` is an optional,
    not-on-PyPI dependency (see ``mpstab.quantum_hardware.synthesis``),
    and the naive fallback alone always has an identity tail, so neither can
    exercise the fold on its own in an environment without rustiq installed.
    """

    def fake(paulis, angles, metric_name="count", preserve_order=True):
        head, _, _ = build_naive_head_and_residual(paulis, angles)
        if not head:
            return head, stim.Tableau(0), []
        n = len(paulis[0])
        split = len(head)
        taken = 0
        while split > 0 and taken < n_trailing and len(head[split - 1]) == 2:
            split -= 1
            taken += 1
        new_head, tail_gates = head[:split], head[split:]
        tail_tableau = stim.Tableau(n)
        for name, qubits in tail_gates:
            gate_name = _STIM_1Q.get(name) or _STIM_2Q.get(name)
            tail_tableau.append(stim.Tableau.from_named_gate(gate_name), qubits)
        return new_head, tail_tableau, tail_gates

    return fake


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
# 1. Both routes match expectation_from_split / expectation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut_fraction", [0.0, 0.5, 1.0])
def test_both_routes_match_split_for_pauli_string(cut_fraction, xy_observable):
    hs = _hs(4, nlayers=2)
    n_dressed = len(hs.magic_gates)
    cut_index = int(round(cut_fraction * n_dressed))
    exact = hs.expectation_from_split(xy_observable, cut_index=cut_index)

    pauli = hs.expectation_at_cut(
        xy_observable, cut_index, method="pauli", n_shots=150000, seed=1
    )
    # shots_per_setting stays at its default of 1: classical-shadow
    # unbiasedness needs a fresh random basis per shot, so a large value would
    # make the reported stderr optimistic rather than just faster to run.
    # tail_handling="append" because with rustiq installed the resynthesis leaves
    # a non-trivial Clifford residual at most cuts, which shadows cannot fold.
    shadows = hs.expectation_at_cut(
        xy_observable,
        cut_index,
        method="shadows",
        n_shots=30000,
        seed=2,
        tail_handling="append",
    )

    assert abs(pauli.value - exact) <= 4 * pauli.total_error + 1e-6, pauli
    assert abs(shadows.value - exact) <= 4 * shadows.total_error + 1e-6, shadows


@pytest.mark.parametrize("cut_fraction", [0.0, 0.5, 1.0])
def test_both_routes_match_split_for_symbolic_hamiltonian(cut_fraction):
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
        tail_handling="append",
    )

    assert abs(pauli.value - exact) <= 4 * pauli.total_error + 1e-3, pauli
    assert abs(shadows.value - exact) <= 4 * shadows.total_error + 1e-3, shadows


@pytest.mark.parametrize("cut_fraction", [0.0, 0.5, 1.0])
def test_tnice_matches_split_for_pauli_string(cut_fraction, xy_observable):
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
    exact = hs.expectation_from_split(xy_observable, cut_index=cut_index)

    result = hs.expectation_at_cut(
        xy_observable,
        cut_index,
        method="tnice",
        n_shots=30000,
        seed=2,
        tail_handling="append",
    )
    assert abs(result.value - exact) <= 4 * result.total_error + 1e-6, result


def test_tnice_forbids_nontrivial_tail_unless_appended(monkeypatch, xy_observable):
    import mpstab.evolutors.hsynthsmpo as hsmod

    hs = _hs(4, nlayers=3, seed=0)
    cut_index = len(hs.magic_gates) - 3
    monkeypatch.setattr(
        hsmod, "build_head_and_residual", _fake_resynthesis_with_trailing_tail(3)
    )

    with pytest.raises(ValueError):
        hs.expectation_at_cut(
            xy_observable,
            cut_index,
            method="tnice",
            n_shots=1000,
            seed=1,
            tail_handling="forbid",
        )

    exact = hs.expectation_from_split(xy_observable, cut_index=cut_index)
    result = hs.expectation_at_cut(
        xy_observable,
        cut_index,
        method="tnice",
        n_shots=30000,
        seed=1,
        tail_handling="append",
    )
    assert abs(result.value - exact) <= 4 * result.total_error, result


# ---------------------------------------------------------------------------
# 2. Clifford-tail fold correctness -- the important one
# ---------------------------------------------------------------------------


def test_clifford_tail_fold_matches_split_and_breaks_if_disabled(
    monkeypatch, xy_observable
):
    import mpstab.evolutors.hsynthsmpo as hsmod

    hs = _hs(4, nlayers=3, seed=0)
    cut_index = len(hs.magic_gates) - 3
    monkeypatch.setattr(
        hsmod, "build_head_and_residual", _fake_resynthesis_with_trailing_tail(3)
    )

    exact = hs.expectation_from_split(xy_observable, cut_index=cut_index)
    result = hs.expectation_at_cut(
        xy_observable, cut_index, method="pauli", n_shots=300000, seed=7
    )
    assert (
        abs(exact) > 0.1
    ), "test fixture must give a non-trivial exact value, see xy_observable"
    assert abs(result.value - exact) <= 4 * result.total_error, result

    # Disable the fold: force fold_pauli_through_tableau to the identity map.
    monkeypatch.setattr(
        hs.stab_engine,
        "fold_pauli_through_tableau",
        lambda pauli_str, tableau, sign=1.0: (pauli_str, sign),
    )
    broken = hs.expectation_at_cut(
        xy_observable, cut_index, method="pauli", n_shots=300000, seed=7
    )
    assert abs(broken.value - exact) > 4 * broken.total_error, (
        "disabling the Clifford-tail fold should produce a value that misses "
        "the exact reference by more than its own error bars -- if it "
        "doesn't, this test is no longer exercising the fold"
    )


def test_shadows_forbids_nontrivial_tail_unless_appended(monkeypatch, xy_observable):
    import mpstab.evolutors.hsynthsmpo as hsmod

    hs = _hs(4, nlayers=3, seed=0)
    cut_index = len(hs.magic_gates) - 3
    monkeypatch.setattr(
        hsmod, "build_head_and_residual", _fake_resynthesis_with_trailing_tail(3)
    )

    with pytest.raises(ValueError):
        hs.expectation_at_cut(
            xy_observable,
            cut_index,
            method="shadows",
            n_shots=1000,
            seed=1,
            tail_handling="forbid",
        )

    exact = hs.expectation_from_split(xy_observable, cut_index=cut_index)
    result = hs.expectation_at_cut(
        xy_observable,
        cut_index,
        method="shadows",
        n_shots=30000,
        seed=1,
        tail_handling="append",
    )
    assert abs(result.value - exact) <= 4 * result.total_error, result


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
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    forbidden = ("mpstab.measurements", "mpstab.models.noise", "models.noise")
    offenders = [
        (str(path), token)
        for path in src.rglob("*.py")
        for token in forbidden
        if token in path.read_text()
    ]
    assert not offenders, offenders
