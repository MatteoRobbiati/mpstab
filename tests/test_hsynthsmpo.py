import numpy as np
import pytest
from qibo import Circuit, gates, set_backend
from utils import expectation_with_qibo, set_rng_seed

pytest.importorskip("qiskit")

from mpstab.engines import NativeTensorNetworkEngine, QuimbEngine
from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import (
    QAE,
    QFT,
    QPE,
    CircuitAnsatz,
    Grover,
    HardwareEfficient,
    QFTPhaseKernel,
    TrotterIsing,
)

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


def _apply_pauli_rotation(psi, pauli, angle, n):
    """Apply exp(-i*angle/2*P) to a dense statevector (row-major, qubit0 first)."""
    I2 = np.eye(2)
    P = {"I": I2, "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.array([[1, 0], [0, -1]])}
    mat = P[pauli[0]]
    for c in pauli[1:]:
        mat = np.kron(mat, P[c])
    rot = np.cos(angle / 2) * np.eye(2 ** n) - 1j * np.sin(angle / 2) * mat
    return rot @ psi


def test_synthesized_head_circuit_formats():
    from qibo import Circuit as QiboCircuit
    from qiskit import QuantumCircuit

    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    n_dressed = len(hs.magic_gates)

    assert isinstance(
        hs.synthesized_head_circuit(n_dressed, output_format="qibo"), QiboCircuit
    )
    assert isinstance(
        hs.synthesized_head_circuit(n_dressed, output_format="qiskit"), QuantumCircuit
    )

    qasm = hs.synthesized_head_circuit(n_dressed, output_format="qasm")
    assert isinstance(qasm, str) and "OPENQASM" in qasm

    with pytest.raises(ValueError):
        hs.synthesized_head_circuit(n_dressed, output_format="bogus")


def test_synthesized_head_circuit_is_faithful():
    # The resynthesized head, run on |0...0>, must reproduce (up to global phase)
    # the state obtained by applying the exact dressed head rotations.
    n = 4
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(n))
    hs = HSynthSMPO(ansatz)
    cut = len(hs.magic_gates)

    # exact reference state from the dressed rotations
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for pauli, angle in hs._dressed_rotations()[:cut]:
        psi = _apply_pauli_rotation(psi, pauli, angle, n)

    head_circuit = hs.synthesized_head_circuit(cut, output_format="qibo")
    psi_synth = head_circuit().state()

    overlap = np.abs(np.vdot(psi, psi_synth))
    assert np.isclose(overlap, 1.0, atol=1e-6)


def test_count_two_qubit_gates_full_cut():
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    n_dressed = len(hs.magic_gates)

    counts = hs.count_two_qubit_gates(cut_index=n_dressed)
    assert counts["n_head_rotations"] == n_dressed
    assert counts["n_tail_rotations"] == 0
    assert counts["synthesized_head_2q_gates"] >= 0
    assert counts["original_circuit_2q_gates"] >= 0


# ---------------------------------------------------------------------------
# Broad coverage over the circuit-library ansatze
# ---------------------------------------------------------------------------
def _observables(n):
    """A spread of Pauli observables of length n, including Y-heavy ones."""
    return [
        "Z" * n,
        "X" * n,
        "Y" * n,  # odd/even #Y -- exercises the expval convention
        ("XYZ" * n)[:n],
        ("IY" * n)[:n],
        "Y" + "I" * (n - 1),  # single Y on qubit 0 (was the bug trigger)
        "I" * (n - 1) + "Y",
    ]


_LIBRARY_ANSATZE = [
    ("QFT", lambda: QFT(nqubits=4)),
    ("QFTPhaseKernel", lambda: QFTPhaseKernel(nqubits=4)),
    ("Grover", lambda: Grover(nqubits=3)),
    ("QPE", lambda: QPE(n_counting=3)),
    ("QAE", lambda: QAE(n_counting=3)),
    ("TrotterIsing", lambda: TrotterIsing(nqubits=4, n_steps=2)),
]


@pytest.mark.parametrize("label,factory", _LIBRARY_ANSATZE, ids=[a[0] for a in _LIBRARY_ANSATZE])
def test_library_ansatze_split_matches_qibo(label, factory):
    ansatz = factory()
    hs = HSynthSMPO(ansatz)
    n = ansatz.nqubits
    n_dressed = len(hs.magic_gates)
    cuts = sorted({0, 1, n_dressed // 2, n_dressed - 1, n_dressed})

    for observable in _observables(n):
        reference = expectation_with_qibo(
            mpstab_ansatz=ansatz, observable_str=observable
        )
        # base HSMPO and every split cut must agree with the statevector.
        assert np.allclose(hs.expectation(observable), reference, atol=1e-6), (
            f"[{label} base / {observable}]"
        )
        for cut in cuts:
            split = hs.expectation_from_split(observable, cut_index=cut)
            assert np.allclose(split, reference, atol=1e-6), (
                f"[{label} / {observable} / cut={cut}] "
                f"split={split:+.6f} reference={reference:+.6f}"
            )


@pytest.mark.parametrize("nqubits", [6, 8, 10])
def test_split_matches_qibo_more_qubits(nqubits):
    # Larger systems (statevector still tractable). We sweep cuts with modest
    # tails (cut_index >= n_dressed // 2): with no bond cap those folds are
    # analytically exact and cheap. The opposite extreme (cut=0, the *entire*
    # scrambled observable folded into a single MPO) needs exponentially large
    # bond and is the intended approximation regime -- it is exercised exactly at
    # small n by test_library_ansatze_split_matches_qibo instead.
    ansatz = TrotterIsing(nqubits=nqubits, n_steps=2)
    hs = HSynthSMPO(ansatz)
    n_dressed = len(hs.magic_gates)
    cuts = sorted({n_dressed // 2, n_dressed - 3, n_dressed - 1, n_dressed})

    for observable in ["Z" * nqubits, "Y" * nqubits, ("XYZ" * nqubits)[:nqubits]]:
        reference = expectation_with_qibo(
            mpstab_ansatz=ansatz, observable_str=observable
        )
        for cut in cuts:
            split = hs.expectation_from_split(observable, cut_index=cut)
            assert np.allclose(split, reference, atol=1e-6), (
                f"[n={nqubits} / {observable} / cut={cut}] "
                f"split={split:+.6f} reference={reference:+.6f}"
            )
