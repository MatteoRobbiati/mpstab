import numpy as np
import pytest
from qibo import Circuit, gates, set_backend
from utils import obs_string_to_qibo_hamiltonian

from mpstab.evolutors.models import HybridSurrogate
from mpstab.targets.ansatze import TranspiledAnsatz

set_backend("numpy")


@pytest.mark.parametrize("observable", ["ZIIXI", "XIXXI", "ZYXZI"])
def test_expectation_matches_qibo(observable):
    circ = Circuit(5)
    [circ.add(gates.H(q)) for q in range(5)]
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.RX(2, theta=0.3))
    circ.add(gates.RY(1, theta=0.5))

    ansatz = TranspiledAnsatz(original_circuit=circ)

    hs = HybridSurrogate(ansatz)
    exp_hybrid = hs.expectation(observable)

    # Exact expval from qibo
    state = circ().state()
    ham = obs_string_to_qibo_hamiltonian(observable)
    exp_qibo = ham.expectation(state)

    assert np.allclose(exp_hybrid, exp_qibo, atol=1e-6)
