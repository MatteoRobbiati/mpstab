import time

import numpy as np
import pytest
from qibo import Circuit, gates, set_backend
from utils import (
    DEFAULT_MAX_BD,
    DEFAULT_REPLACEMENT_PROBABILITY,
    construct_symbolic_hamiltonian,
    construct_test_circuit,
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


def _all_pauli_strings(n):
    """All non-identity Pauli strings of length n."""
    import itertools

    return [
        "".join(p)
        for p in itertools.product("IXYZ", repeat=n)
        if set(p) != {"I"}
    ]


def test_expectation_matches_qibo_full_pauli_sweep():
    """Regression: base HSMPO must match qibo for *every* Pauli observable.

    Earlier tests only used Y observables that evaluated to ~0 on their circuit,
    so a sign error in the (transposed) expval contraction for observables with
    an odd number of Y's went unnoticed. This sweeps all 3-qubit Paulis on a
    circuit with generic nonzero expectation values, which pins that convention.
    """
    circ = Circuit(3)
    for q in range(3):
        circ.add(gates.H(q))
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.RY(1, theta=0.5))
    circ.add(gates.RX(2, theta=0.3))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.RZ(0, theta=0.7))

    ansatz = CircuitAnsatz(qibo_circuit=circ)
    hs = HSMPO(ansatz)

    for observable in _all_pauli_strings(3):
        exp_hybrid = hs.expectation(observable)
        exp_qibo = expectation_with_qibo(
            mpstab_ansatz=ansatz, observable_str=observable
        )
        assert np.allclose(exp_hybrid, exp_qibo, atol=1e-6), (
            f"[{observable}] hybrid={exp_hybrid:+.6f} qibo={exp_qibo:+.6f}"
        )


def test_expectation_from_partition_with_qubit_scaling():
    times = []

    for nqubits in [4, 12, 24]:
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


@pytest.mark.parametrize("rng_seed", range(5))
@pytest.mark.parametrize("nqubits", [4, 7, 9])
def test_symbolic_hamiltonian_expectation(rng_seed, nqubits):

    set_rng_seed(rng_seed)

    # Initialising a general ansatz
    circuit = construct_test_circuit(nqubits=nqubits, rng_seed=rng_seed)
    h = construct_symbolic_hamiltonian(nqubits=nqubits, rng_seed=rng_seed)

    hs = HSMPO(ansatz=circuit)
    exp_mpstab = hs.expectation(h)

    exp_qibo = h.expectation_from_state(circuit().state())

    assert np.allclose(exp_mpstab, exp_qibo, atol=1e-6)
