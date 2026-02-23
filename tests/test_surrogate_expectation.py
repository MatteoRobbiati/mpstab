import time

import numpy as np
import pytest
from qibo import Circuit, gates, set_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import I, X, Y, Z
from utils import (
    DEFAULT_MAX_BD,
    DEFAULT_REPLACEMENT_PROBABILITY,
    DEFAULT_RNG_SEED,
    expectation_with_qibo,
    set_rng_seed,
)

from mpstab.engines import StimEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient

set_backend("numpy")
set_rng_seed()


@pytest.mark.parametrize("observable", ["ZIIXI", "XIXXI", "ZYXZI"])
def test_expectation_matches_qibo(observable):
    circ = Circuit(5)
    [circ.add(gates.H(q)) for q in range(5)]
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.RX(2, theta=0.3))
    circ.add(gates.RY(1, theta=0.5))

    ansatz = CircuitAnsatz(qibo_circuit=circ)

    hs = HSMPO(ansatz)
    hs.set_engines(stab_engine=StimEngine())
    exp_hybrid = hs.expectation(observable)

    exp_qibo = expectation_with_qibo(
        mpstab_ansatz=ansatz,
        observable_str=observable,
    )

    assert np.allclose(exp_hybrid, exp_qibo, atol=1e-6)


def test_expectation_from_partition_with_qubit_scaling():
    times = []

    for nqubits in [4, 8, 12]:
        ans = HardwareEfficient(nqubits=nqubits, nlayers=3)
        hs = HSMPO(ansatz=ans)
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
    hs = HSMPO(ansatz=ans, max_bond_dimension=DEFAULT_MAX_BD)
    no_repl_expval = hs.expectation(observable=obs)
    repl_expval = hs.expectation_from_partition(
        observable=obs,
        replacement_probability=DEFAULT_REPLACEMENT_PROBABILITY,
        replacement_method=method,
    )[0]

    assert no_repl_expval != repl_expval


@pytest.mark.parametrize("nqubits", [5, 6, 7, 8])
def test_symbolic_hamiltonian_expectation(nqubits):

    set_rng_seed(DEFAULT_RNG_SEED)

    # Initialising a general ansatz
    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=2)

    symbols = [X, Y, Z, I]
    ham_form = 0
    n_terms = 5

    for _ in range(n_terms):
        coeff = np.random.uniform(0.5, 2.0)
        # Pick 1 or 2 random qubits for this term
        target_qubits = np.random.choice(range(nqubits), size=2, replace=False)
        paulis = np.random.choice(symbols, size=2)

        # Construct term: e.g. 1.2 * X(0) * Z(3)
        term = (
            coeff * paulis[0](int(target_qubits[0])) * paulis[1](int(target_qubits[1]))
        )
        ham_form += term

    # Add a constant shift to test completeness
    constant_shift = 0.5
    ham_form += constant_shift

    h = SymbolicHamiltonian(ham_form)

    hs = HSMPO(ansatz=ansatz)
    exp_mpstab = hs.expectation(h)

    exp_qibo = h.expectation_from_state(ansatz.circuit().state())

    assert np.allclose(exp_mpstab, exp_qibo, atol=1e-6)
