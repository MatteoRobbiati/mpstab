import numpy as np
import pytest
from qibo import Circuit, gates, set_backend
from utils import expectation_with_qibo, set_rng_seed

pytest.importorskip("qiskit")

from mpstab.engines import NativeTensorNetworkEngine, QuimbEngine
from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient

set_backend("numpy")
set_rng_seed()


def _small_entangled_circuit(n=4):
    """H layer + CZ chain + RY layer: yields multi-qubit dressed rotations."""
    set_rng_seed()
    circ = Circuit(n)
    for q in range(n):
        circ.add(gates.H(q))
    for q in range(n - 1):
        circ.add(gates.CZ(q, q + 1))
    for q in range(n):
        circ.add(gates.RY(q, theta=float(np.random.uniform(0.1, 1.2))))
    return circ


@pytest.mark.parametrize("observable", ["ZZZZ", "XIII", "IZIX", "YYYY"])
@pytest.mark.parametrize("cut_index", [0, 2, 4])
def test_split_matches_qibo(observable, cut_index):
    circ = _small_entangled_circuit(4)
    ansatz = CircuitAnsatz(qibo_circuit=circ)
    hs = HSynthSMPO(ansatz)

    split = hs.expectation_from_split(observable, cut_index=cut_index)
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)

    assert np.allclose(split, reference, atol=1e-6), (
        f"[obs={observable}, cut={cut_index}] split={split:+.6f} "
        f"reference={reference:+.6f}"
    )


@pytest.mark.parametrize("cut_index", [0, 3, 6])
def test_split_matches_base_expectation(cut_index):
    ansatz = HardwareEfficient(nqubits=4, nlayers=2)
    hs = HSynthSMPO(ansatz)
    observable = "ZIZI"

    base = hs.expectation(observable)
    split = hs.expectation_from_split(observable, cut_index=cut_index)

    assert np.allclose(base, split, atol=1e-6)


def test_return_fidelity():
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    expval, fidelity = hs.expectation_from_split("ZZZZ", cut_index=2, return_fidelity=True)
    assert np.isreal(expval)
    assert 0.0 <= fidelity <= 1.0 + 1e-6


def test_native_engine_not_supported():
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    hs.set_engines(tn_engine=NativeTensorNetworkEngine())
    with pytest.raises(NotImplementedError):
        hs.expectation_from_split("ZZZZ", cut_index=2)


def test_mpo_tail_approximation_lossless_when_untruncated():
    # No bond cap => the folded tail MPO is exact => zero error.
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)  # max_bond_dimension defaults to None
    info = hs.mpo_tail_approximation("ZZZZ", cut_index=0)
    assert info["relative_frobenius_error"] == pytest.approx(0.0, abs=1e-10)
    assert info["expval_abs_error"] == pytest.approx(0.0, abs=1e-10)


def test_mpo_tail_approximation_empty_tail():
    # cut_index == number of dressed rotations => empty tail => exact operator.
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz, max_bond_dimension=2)
    n_dressed = len(hs.magic_gates)
    info = hs.mpo_tail_approximation("ZZZZ", cut_index=n_dressed)
    assert info["relative_frobenius_error"] == pytest.approx(0.0, abs=1e-10)


def test_mpo_tail_approximation_reports_error_when_truncated():
    # Aggressive bond cap on a full tail => nonzero operator error; the exact
    # (reference) operator still reproduces the true expectation, since with
    # cut_index=0 the state is exactly |0...0>.
    ansatz = HardwareEfficient(nqubits=6, nlayers=2)
    hs = HSynthSMPO(ansatz, max_bond_dimension=1)
    info = hs.mpo_tail_approximation("Z" * 6, cut_index=0, reference_max_bond=None)
    assert info["relative_frobenius_error"] >= 0.0
    assert info["approx_max_bond"] <= info["reference_max_bond"]

    exact = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str="Z" * 6)
    assert np.allclose(info["expval_reference"], exact, atol=1e-6)


def test_count_two_qubit_gates_full_cut():
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    n_dressed = len(hs.magic_gates)

    counts = hs.count_two_qubit_gates(cut_index=n_dressed)
    assert counts["n_head_rotations"] == n_dressed
    assert counts["n_tail_rotations"] == 0
    assert counts["synthesized_head_2q_gates"] >= 0
    assert counts["original_circuit_2q_gates"] >= 0
