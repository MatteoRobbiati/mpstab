"""
The pure-Python engines against the stim/quimb ones and the statevector.

They are kept as readable examples of the engine interface, so these tests
check that they give the same numbers wherever they are supported, and that
they fail clearly where they are not (DMRG, operator-side conjugation).
"""

import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.quantum_info import meyer_wallach_entanglement
from utils import expectation_with_qibo, small_entangled_circuit

from mpstab import HSMPO
from mpstab.engines import (
    NativeStabilizersEngine,
    NativeTensorNetworkEngine,
    QuimbEngine,
    StimEngine,
)
from mpstab.evolutors.hqsmpo import HQSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient, KickedIsing
from mpstab.models.metrics import meyer_wallach_mps
from mpstab.models.noise import DepolarizingError, NoiseModel


@pytest.mark.parametrize("observable", ["ZZZZZ", "ZZIXI", "XYIZX"])
def test_expectation_matches_qibo(stab_engine, tn_engine, observable):
    ansatz = HardwareEfficient(nqubits=5, nlayers=3)
    hsmpo = HSMPO(ansatz, stab_engine=stab_engine, tn_engine=tn_engine)
    expected = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)
    assert hsmpo.expectation(observable) == pytest.approx(expected, abs=1e-8)


def test_non_trivial_initial_state(tn_engine):
    """The native engine builds its MPS from the initial-state circuit itself."""
    initial_state = Circuit(4)
    for q in range(4):
        initial_state.add(gates.RY(q, theta=0.4 + 0.3 * q))
    ansatz = KickedIsing(nqubits=4)
    hsmpo = HSMPO(ansatz, initial_state=initial_state, tn_engine=tn_engine)
    reference = HSMPO(ansatz, initial_state=initial_state, tn_engine=QuimbEngine())
    assert hsmpo.expectation("ZXZI") == pytest.approx(
        reference.expectation("ZXZI"), abs=1e-10
    )


def test_meyer_wallach_matches_dense(tn_engine):
    circuit = small_entangled_circuit(4)
    dense = meyer_wallach_entanglement(circuit.copy()().state())
    hsmpo = HSMPO(ansatz=circuit.copy(), tn_engine=tn_engine)
    assert meyer_wallach_mps(hsmpo) == pytest.approx(dense, abs=1e-10)


def test_noisy_expectation_matches_quimb():
    """The engine's default rotation hooks give quimb's estimate."""
    ansatz = KickedIsing(nqubits=5)
    noise = NoiseModel().add(DepolarizingError(0.02), gates.CNOT)
    observable = {"ZZIII": 0.7, "IXIXI": -0.4}
    quimb = HSMPO(ansatz).noisy_expectation(observable, noise, n_samples=200, seed=9)
    native = HSMPO(ansatz, tn_engine=NativeTensorNetworkEngine()).noisy_expectation(
        observable, noise, n_samples=200, seed=9
    )
    assert native.n_patterns == quimb.n_patterns
    assert native.mean == pytest.approx(quimb.mean, abs=1e-10)


def test_backpropagate_at_matches_backpropagate_on_each_prefix(stab_engine):
    """The default (stim) and the native override honour the same contract."""
    circuit = HSMPO(small_entangled_circuit(4)).clifford_circuit
    observables = ["ZIII", "XYIZ", "IIXX", "YZZY", "ZZZZ"]
    breakpoints = [3, 0, len(circuit.queue), 1, 3]  # unsorted on purpose
    result = stab_engine.backpropagate_at(observables, breakpoints, circuit)
    for observable, k, got in zip(observables, breakpoints, result):
        prefix = Circuit(circuit.nqubits)
        for gate in circuit.queue[:k]:
            prefix.add(gate)
        expected = StimEngine().backpropagate(observable, prefix)
        assert (got[0], float(got[1])) == (expected[0], float(expected[1]))


def test_backpropagate_at_rejects_a_breakpoint_outside_the_circuit():
    circuit = HSMPO(small_entangled_circuit(4)).clifford_circuit
    with pytest.raises(ValueError, match="outside"):
        StimEngine().backpropagate_at(["ZIII"], [len(circuit.queue) + 1], circuit)


def test_noise_with_native_stabilizers_matches_stim():
    ansatz = KickedIsing(nqubits=5)
    noise = NoiseModel().add(DepolarizingError(0.02), gates.CNOT)
    observable = {"ZZIII": 0.7, "IXIXI": -0.4}
    stim = HSMPO(ansatz).noisy_expectation(observable, noise, n_samples=200, seed=9)
    native = HSMPO(ansatz, stab_engine=NativeStabilizersEngine()).noisy_expectation(
        observable, noise, n_samples=200, seed=9
    )
    assert native.n_patterns == stim.n_patterns
    assert native.mean == pytest.approx(stim.mean, abs=1e-10)


def test_hqsmpo_with_native_stabilizers():
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz, stab_engine=NativeStabilizersEngine())
    result = hs.expectation_at_cut("ZZZZ", 2, method="simulation")
    expected = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str="ZZZZ")
    assert np.real(result.value) == pytest.approx(expected, abs=1e-8)


def test_native_tensor_network_has_no_operator_conjugation():
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz, tn_engine=NativeTensorNetworkEngine())
    with pytest.raises(NotImplementedError, match="QuimbEngine"):
        hs.expectation_at_cut("ZZZZ", 2, method="simulation")


def test_native_tensor_network_has_no_dmrg():
    hsmpo = HSMPO(
        HardwareEfficient(nqubits=4, nlayers=2), tn_engine=NativeTensorNetworkEngine()
    )
    with pytest.raises(
        NotImplementedError, match="NativeTensorNetworkEngine does not implement"
    ):
        hsmpo.minimize_expectation(
            observables={"ZZZZ": 1.0}, bond_dims=[8], max_sweeps=2
        )


@pytest.mark.parametrize("seed", range(4))
def test_native_two_qubit_rotations_match_stim(seed):
    """RXX/RYY/RZZ at Clifford angles, which replaced magic gates become."""
    rng = np.random.default_rng(seed)
    rotations = [gates.RXX, gates.RYY, gates.RZZ]
    angles = [0.0, np.pi / 2, -np.pi / 2, np.pi, 3 * np.pi / 2]
    circuit = Circuit(3)
    for _ in range(12):
        a, b = (int(q) for q in rng.choice(3, size=2, replace=False))
        if rng.random() < 0.5:
            rotation = rotations[rng.integers(3)]
            circuit.add(rotation(a, b, theta=float(rng.choice(angles))))
        else:
            circuit.add([gates.CNOT(a, b), gates.H(a), gates.S(b)][rng.integers(3)])
    observables = ["".join(rng.choice(list("IXYZ"), 3)) for _ in range(6)]
    expected = StimEngine().backpropagate_many(observables, circuit)
    result = NativeStabilizersEngine().backpropagate_many(observables, circuit)
    assert [(p, int(s)) for p, s in result] == [(p, int(s)) for p, s in expected]


def test_native_rejects_non_clifford_two_qubit_rotation():
    circuit = Circuit(2)
    circuit.add(gates.RZZ(0, 1, theta=0.3))
    with pytest.raises(ValueError, match="not Clifford"):
        NativeStabilizersEngine().backpropagate("ZZ", circuit)


def test_native_rejects_unknown_gates():
    circuit = Circuit(2)
    circuit.add(gates.CRX(0, 1, theta=np.pi))
    with pytest.raises(ValueError, match="not supported"):
        NativeStabilizersEngine().backpropagate("ZZ", circuit)
