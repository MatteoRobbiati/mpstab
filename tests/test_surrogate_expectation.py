import time

import numpy as np
import pytest
from qibo import Circuit, gates, set_backend
from utils import (
    DEFAULT_MAX_BD,
    DEFAULT_REPLACEMENT_PROBABILITY,
    obs_string_to_qibo_hamiltonian,
)

from mpstab.evolutors.models import HybridSurrogate
from mpstab.models.ansatze import HardwareEfficient, TranspiledAnsatz

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


def test_expectation_from_partition_with_qubit_scaling():
    times = []

    for nqubits in [4, 8, 12]:
        ans = HardwareEfficient(nqubits=nqubits, nlayers=3)
        hs = HybridSurrogate(ansatz=ans, max_bond_dimension=DEFAULT_MAX_BD)
        initial_time = time.time()
        hs.expectation_from_partition(
            observable="Z" * nqubits,
            replacement_probability=DEFAULT_REPLACEMENT_PROBABILITY,
        )
        times.append(time.time() - initial_time)

    assert times[0] < times[1]
    assert times[1] < times[2]


@pytest.mark.parametrize("method", ["closest", "random"])
def test_replacement_methods(method):

    nqubits = 6
    obs = "Z" * nqubits

    ans = HardwareEfficient(nqubits=nqubits, nlayers=3)
    hs = HybridSurrogate(ansatz=ans, max_bond_dimension=DEFAULT_MAX_BD)
    no_repl_expval = hs.expectation(observable=obs)
    repl_expval = hs.expectation_from_partition(
        observable=obs,
        replacement_probability=DEFAULT_REPLACEMENT_PROBABILITY,
        replacement_method=method,
    )

    assert no_repl_expval != repl_expval
