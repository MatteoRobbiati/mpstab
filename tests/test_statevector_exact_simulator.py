"""StatevectorExactSimulator simulates each state preparation once and only rotates it
into each measurement basis: the outcome distribution must not change."""

import numpy as np
import pytest
from qibo import Circuit, gates

from mpstab.qibo_tools.utilized_backends.statevector_exact_simulator import (
    StatevectorExactSimulator,
    _measured_probabilities,
    _prepared_state,
    _split_basis_measurement,
)
from mpstab.qpu.measurement.abstract import basis_measurement

N = 4


def _preparation():
    rng = np.random.default_rng(5)
    circuit = Circuit(N)
    for q in range(N):
        circuit.add(gates.RY(q, theta=rng.uniform(0, 3)))
    for q in range(N - 1):
        circuit.add(gates.CNOT(q, q + 1))
    circuit.add(gates.U3(1, 0.3, 0.2, 0.9))
    return circuit


def _exact(circuit, measured):
    """Outcome probabilities of ``measured`` from the unmeasured circuit."""
    bare = Circuit(N)
    for gate in circuit.queue:
        if not isinstance(gate, gates.M):
            bare.add(gate)
    probabilities = np.abs(np.asarray(bare().state())) ** 2
    probabilities = probabilities.reshape([2] * N)
    others = tuple(q for q in range(N) if q not in measured)
    marginal = probabilities.sum(axis=others)
    order = [sorted(measured).index(q) for q in measured]
    return np.transpose(marginal, order).ravel()


@pytest.mark.parametrize("basis", ["XYZX", "ZZZZ", "YZXY"])
def test_basis_probabilities_match_the_full_circuit(basis):
    circuit = _preparation() + basis_measurement(basis, N)
    preparation, rotations, measured = _split_basis_measurement(circuit)
    state = _prepared_state(N, preparation)
    np.testing.assert_allclose(
        _measured_probabilities(state, rotations, measured),
        _exact(circuit, measured),
        atol=1e-12,
    )


def test_subset_measured_in_any_order():
    circuit = _preparation()
    circuit.add(gates.H(3))
    circuit.add(gates.M(3, 0))
    preparation, rotations, measured = _split_basis_measurement(circuit)
    assert measured == (3, 0)
    state = _prepared_state(N, preparation)
    np.testing.assert_allclose(
        _measured_probabilities(state, rotations, measured),
        _exact(circuit, measured),
        atol=1e-12,
    )


def test_unsupported_circuits_run_as_they_are():
    noisy = _preparation()
    noisy.add(gates.DepolarizingChannel((0,), 0.1))
    noisy.add(gates.M(*range(N)))
    assert _split_basis_measurement(noisy) is None
    density = Circuit(N, density_matrix=True)
    density.add(gates.H(0))
    density.add(gates.M(0))
    assert _split_basis_measurement(density) is None

    results = StatevectorExactSimulator().execute_circuits(
        [noisy, _preparation() + basis_measurement("XXXX", N), density], 50
    )
    assert [sum(r.values()) for r in results] == [50, 50, 50]
    assert all(len(key) == N for key in results[0])
    assert set(results[2]) <= {"0", "1"}


def test_every_circuit_gets_its_own_samples():
    circuit = _preparation() + basis_measurement("XYZX", N)
    results = StatevectorExactSimulator().execute_circuits([circuit] * 3, 1000)
    assert all(sum(r.values()) == 1000 for r in results)
    assert results[0] != results[1]  # independent draws
