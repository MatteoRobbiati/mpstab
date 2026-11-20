"""Tests for mpstab.models.error_mitigation: CDR and vnCDR on HSMPO."""

import random

import numpy as np
import pytest
from qibo import Circuit, gates

from mpstab import HSMPO
from mpstab.models.ansatze import HardwareEfficient
from mpstab.models.error_mitigation import (
    MitigationNoiseMap,
    cdr,
    fold_circuit,
    noisy_value,
    training_circuits,
    vncdr,
)
from mpstab.models.noise import DepolarizingError, NoiseModel, PauliError

OBSERVABLE = "ZZII"


@pytest.fixture
def ansatz():
    ansatz = HardwareEfficient(nqubits=4, nlayers=2)
    rng = np.random.default_rng(1)
    ansatz.circuit.set_parameters(rng.uniform(-np.pi, np.pi, ansatz.nparams))
    return ansatz


@pytest.fixture
def noise():
    return (
        NoiseModel()
        .add(DepolarizingError(0.01))
        .add(PauliError({"ZZ": 0.01}), gates.CZ)
    )


def _exact(circuit):
    """Noiseless value of OBSERVABLE on ``circuit``."""
    return HSMPO(circuit, progress=False).expectation(OBSERVABLE)


def _angles(circuit):
    return [g.parameters for g in circuit.queue]


def test_training_circuits_keep_gate_sequence_and_seed(ansatz):
    first = training_circuits(ansatz, 5, replacement_probability=0.5, seed=3)
    again = training_circuits(ansatz, 5, replacement_probability=0.5, seed=3)
    assert [_angles(c) for c in first] == [_angles(c) for c in again]
    names = [g.name for g in ansatz.circuit.queue]
    for circuit in first:
        assert [g.name for g in circuit.queue] == names
    assert len({str(_angles(c)) for c in first}) > 1


def test_training_circuits_do_not_touch_global_random(ansatz):
    random.seed(7)
    expected = random.random()
    random.seed(7)
    training_circuits(ansatz, 3, replacement_probability=0.5, seed=0)
    assert random.random() == expected


def test_training_circuits_without_replacement_are_the_target(ansatz):
    (circuit,) = training_circuits(ansatz, 1, replacement_probability=0.0)
    assert _angles(circuit) == _angles(ansatz.circuit)


def test_fully_clifford_closest_warns(ansatz):
    with pytest.warns(UserWarning, match="same circuit"):
        training_circuits(ansatz, 2, replacement_probability=1.0)


def test_fold_circuit_keeps_the_unitary(ansatz):
    folded = fold_circuit(ansatz.circuit, 3)
    assert len(folded.queue) == 3 * len(ansatz.circuit.queue)
    np.testing.assert_allclose(folded().state(), ansatz.circuit().state(), atol=1e-10)
    with pytest.raises(ValueError, match="odd"):
        fold_circuit(ansatz.circuit, 2)


def test_cdr_inverts_a_linear_executor(ansatz):
    noise_map = cdr(
        ansatz,
        OBSERVABLE,
        lambda circuit: 0.8 * _exact(circuit) + 0.05,
        n_training_circuits=10,
        replacement_probability=0.5,
        seed=0,
    )
    np.testing.assert_allclose(noise_map.params, [1 / 0.8, -0.05 / 0.8], atol=1e-10)
    np.testing.assert_allclose(noise_map.residuals, 0, atol=1e-10)
    target = _exact(ansatz.circuit)
    assert noise_map(0.8 * target + 0.05) == pytest.approx(target)


def test_vncdr_extrapolates_a_known_decay(ansatz):
    levels = (1, 2, 3)
    damping = lambda level: 0.9**level - 0.1 * 0.8**level  # noqa: E731
    noise_map = vncdr(
        ansatz,
        OBSERVABLE,
        lambda circuit, level: damping(level) * _exact(circuit),
        noise_levels=levels,
        n_training_circuits=10,
        replacement_probability=0.5,
        seed=0,
    )
    assert noise_map.noisy.shape == (10, 3)
    np.testing.assert_allclose(noise_map.residuals, 0, atol=1e-10)
    target = _exact(ansatz.circuit)
    assert noise_map([damping(l) * target for l in levels]) == pytest.approx(target)
    with pytest.raises(ValueError, match="noise level"):
        noise_map([0.1, 0.2])


@pytest.mark.parametrize("amplification", ["scale", "fold"])
def test_mitigation_with_a_noise_model(ansatz, noise, amplification):
    target = HSMPO(ansatz, progress=False)
    exact = target.expectation(OBSERVABLE)
    noisy = noisy_value(target, OBSERVABLE, noise, seed=0)
    kwargs = dict(n_training_circuits=20, replacement_probability=0.7, seed=0)

    noise_map = cdr(ansatz, OBSERVABLE, noise, **kwargs)
    assert abs(noise_map(noisy) - exact) < abs(noisy - exact) / 2

    levels = (1, 3, 5)
    noise_map = vncdr(
        ansatz,
        OBSERVABLE,
        noise,
        noise_levels=levels,
        amplification=amplification,
        **kwargs,
    )
    values = [
        noisy_value(target, OBSERVABLE, noise, level, amplification, seed=0)
        for level in levels
    ]
    assert abs(noise_map(values) - exact) < abs(noisy - exact) / 4


def test_custom_model(ansatz):
    noise_map = cdr(
        ansatz,
        OBSERVABLE,
        lambda circuit: 0.5 * _exact(circuit),
        n_training_circuits=10,
        replacement_probability=0.5,
        model=lambda x, a: a * x,
        seed=0,
    )
    assert isinstance(noise_map, MitigationNoiseMap)
    np.testing.assert_allclose(noise_map.params, [2.0], atol=1e-6)
    with pytest.raises(ValueError, match="name its parameters"):
        cdr(
            ansatz,
            OBSERVABLE,
            lambda circuit: 0.5 * _exact(circuit),
            n_training_circuits=3,
            replacement_probability=0.5,
            model=lambda x, *p: p[0] * x,
        )


def test_executor_receives_the_initial_state(ansatz):
    initial_state = Circuit(ansatz.nqubits)
    initial_state.add(gates.X(0))
    seen = []

    def executor(circuit):
        seen.append(circuit)
        return 0.0

    cdr(
        ansatz,
        OBSERVABLE,
        executor,
        n_training_circuits=2,
        replacement_probability=0.5,
        initial_state=initial_state,
        seed=0,
    )
    assert all(isinstance(c.queue[0], gates.X) for c in seen)
    assert all(len(c.queue) == len(ansatz.circuit.queue) + 1 for c in seen)


def test_workers_do_not_change_the_result(ansatz, noise):
    kwargs = dict(
        n_training_circuits=4, replacement_probability=0.7, n_samples=50, seed=0
    )
    serial = vncdr(ansatz, OBSERVABLE, noise, noise_levels=(1, 2), **kwargs)
    parallel = vncdr(
        ansatz, OBSERVABLE, noise, noise_levels=(1, 2), n_workers=2, **kwargs
    )
    np.testing.assert_array_equal(serial.noisy, parallel.noisy)
    np.testing.assert_array_equal(serial.noisy_stderr, parallel.noisy_stderr)
    np.testing.assert_array_equal(serial.exact, parallel.exact)
    assert serial.noisy_stderr.shape == (4, 2)
