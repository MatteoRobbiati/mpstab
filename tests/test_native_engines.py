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
)
from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient, KickedIsing
from mpstab.models.metrics import meyer_wallach_mps
from mpstab.noise import DepolarizingError, NoiseModel


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


def test_hsynthsmpo_with_native_stabilizers():
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz, stab_engine=NativeStabilizersEngine())
    result = hs.expectation_at_cut("ZZZZ", 2, method="simulation")
    expected = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str="ZZZZ")
    assert np.real(result.value) == pytest.approx(expected, abs=1e-8)


def test_native_tensor_network_has_no_operator_conjugation():
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz, tn_engine=NativeTensorNetworkEngine())
    with pytest.raises(NotImplementedError, match="QuimbEngine"):
        hs.expectation_at_cut("ZZZZ", 2, method="simulation")


def test_native_tensor_network_has_no_dmrg():
    hsmpo = HSMPO(
        HardwareEfficient(nqubits=4, nlayers=2), tn_engine=NativeTensorNetworkEngine()
    )
    with pytest.raises(NotImplementedError, match="requires QuimbEngine"):
        hsmpo.minimize_expectation(
            observables={"ZZZZ": 1.0}, bond_dims=[8], max_sweeps=2
        )
