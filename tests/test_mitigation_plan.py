"""Tests for mpstab.models.error_mitigation.plan: vnCDR on HQSMPO plans."""

import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Z

from mpstab import HQSMPO, HSMPO, pauli_string_to_hamiltonian
from mpstab.models.ansatze import HardwareEfficient
from mpstab.models.error_mitigation import (
    MitigatedResult,
    VnCDRMitigation,
    mitigated_expectation_at_cut,
)
from mpstab.models.error_mitigation.plan import _spread
from mpstab.models.noise import DepolarizingError, NoiseModel, ReadoutError
from mpstab.qpu import NoisySimulator, QiboSimulator, execute_plan

OBSERVABLE = "IZZI"


@pytest.fixture
def model():
    np.random.seed(3)
    return HQSMPO(
        HardwareEfficient(nqubits=4, nlayers=3),
        head_synthesis="extraction",
        progress=False,
    )


def _cut(model):
    return int(model.n_magic_gates * 0.6)


def _plan_reference(model, plan):
    """The noiseless value of the plan's own estimator on the head."""
    head = model.resynthesize_head(_cut(model)).circuit
    _, coefficients = plan.target.recombination
    return HSMPO(head, progress=False).expectation(coefficients) + plan.target.constant


def test_plan_layout(model):
    mitigation = VnCDRMitigation(
        noise_levels=(1, 3),
        n_training_circuits=5,
        replacement_method="random",
        training_settings=2,
        training_shots=200,
        calibration_shots=100,
    )
    plan = mitigation.build_plan(model, OBSERVABLE, _cut(model), n_shots=1000, seed=0)
    n_settings = len(plan.target.circuits)
    kinds = [m.kind for m in plan.measurements]
    assert kinds.count("target") == 2 * n_settings
    assert kinds.count("training") == 5 * 2 * 2
    assert kinds.count("calibration") == 6
    assert len(plan.circuits) == len(plan.shots) == len(plan.measurements)
    assert plan.circuits[:n_settings] == plan.target.circuits
    assert len(plan.training_exact) == 5


def test_noiseless_run_matches_the_plan(model):
    mitigation = VnCDRMitigation(
        n_training_circuits=6,
        replacement_method="random",
        training_settings=None,
        training_shots=4000,
        calibration_shots=4000,
    )
    plan = mitigation.build_plan(model, OBSERVABLE, _cut(model), n_shots=20000, seed=1)
    result = plan.estimate(execute_plan(QiboSimulator(), plan))
    reference = _plan_reference(model, plan)
    assert isinstance(result, MitigatedResult)
    assert abs(result.unmitigated - reference) < 5 * result.stderr + 0.02
    assert abs(result.value - reference) < 0.05
    assert result.n_shots == sum(plan.shots)


def test_mitigation_reduces_the_error(model):
    noise = (
        NoiseModel()
        .add(DepolarizingError(5e-3))
        .add(DepolarizingError(3e-2), gates.CZ)
        .add(ReadoutError(0.02, 0.05))
    )
    mitigation = VnCDRMitigation(
        n_training_circuits=10,
        replacement_method="random",
        replacement_probability=0.9,
        training_settings=None,
        training_shots=6000,
        calibration_shots=20000,
    )
    plan = mitigation.build_plan(model, OBSERVABLE, _cut(model), n_shots=30000, seed=2)
    result = plan.estimate(execute_plan(NoisySimulator(noise, seed=0), plan))
    reference = _plan_reference(model, plan)
    unmitigated = abs(result.unmitigated - reference)
    assert abs(result.measurement_mitigated - reference) < unmitigated
    assert abs(result.cdr - reference) < unmitigated / 3
    assert abs(result.value - reference) < unmitigated / 3


HAMILTONIAN = SymbolicHamiltonian(
    sum(Z(q) * Z(q + 1) for q in range(3)) + 0.5 * sum(X(q) for q in range(4)),
    nqubits=4,
)


def test_training_settings_are_informative_and_distinct(model):
    mitigation = VnCDRMitigation(
        n_training_circuits=8, replacement_method="random", training_settings=2
    )
    plan = mitigation.build_plan(model, HAMILTONIAN, _cut(model), n_shots=2000, seed=0)
    assert len(plan.target.recombination[0]) > 2
    for i in range(8):
        drawn = [
            m.group
            for m in plan.measurements
            if m.kind == "training" and m.level == 1 and m.training_index == i
        ]
        assert len(drawn) == len(set(drawn)) <= 2
        assert all(abs(plan.training_group_exact[i, g]) > 1e-12 for g in drawn)


def test_pooled_fit_uses_one_point_per_circuit_and_setting(model):
    mitigation = VnCDRMitigation(
        n_training_circuits=5, replacement_method="random", training_settings=2
    )
    plan = mitigation.build_plan(model, HAMILTONIAN, _cut(model), n_shots=4000, seed=1)
    result = plan.estimate(execute_plan(QiboSimulator(), plan))
    n_points = sum(
        1 for m in plan.measurements if m.kind == "training" and m.level == 1
    )
    assert result.training_noisy.shape == (n_points, len(plan.noise_levels))
    assert result.value == pytest.approx(result.settings.sum() + plan.target.constant)


def test_calibration_undoes_asymmetric_readout(model):
    readout_only = NoiseModel().add(ReadoutError(0.02, 0.15))
    mitigation = VnCDRMitigation(
        noise_levels=(1,),
        n_training_circuits=3,
        replacement_method="random",
        training_settings=None,
        calibration_shots=100000,
    )
    plan = mitigation.build_plan(model, OBSERVABLE, _cut(model), n_shots=200000, seed=0)
    assert sum(m.kind == "calibration" for m in plan.measurements) == 6
    result = plan.estimate(execute_plan(NoisySimulator(readout_only, seed=0), plan))
    reference = _plan_reference(model, plan)
    assert abs(result.unmitigated - reference) > 0.1
    assert abs(result.measurement_mitigated - reference) < 0.02


def test_one_call_matches_the_stages(model):
    mitigation = VnCDRMitigation(
        n_training_circuits=4, replacement_method="random", training_settings=2
    )
    result = mitigated_expectation_at_cut(
        model, OBSERVABLE, _cut(model), mitigation, n_shots=2000, seed=0
    )
    assert np.isfinite(result.value)


def test_spread_picks_values_across_the_range():
    values = np.array([0.0, 0.01, 0.02, 0.5, 0.51, 0.98, 1.0])
    chosen = _spread(values, 3)
    assert sorted(values[chosen]) == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("levels", [(3, 5), (1, 2), (1, 3, 3.5)])
def test_noise_levels_must_be_odd_and_start_at_one(levels):
    with pytest.raises(ValueError, match="odd"):
        VnCDRMitigation(noise_levels=levels)


def test_noisy_simulator_readout_flips():
    circuit = Circuit(3)
    circuit.add(gates.M(0, 1, 2))
    certain_flip = NoiseModel().add(ReadoutError(1.0, 0.0), qubits=[1])
    (frequencies,) = NoisySimulator(certain_flip, seed=0).execute_circuits(
        [circuit], 50
    )
    assert frequencies == {"010": 50}


def test_noisy_simulator_without_noise_matches_the_state():
    circuit = Circuit(2)
    circuit.add(gates.RY(0, theta=1.1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(0, 1, basis=[gates.X, gates.Z]))
    (frequencies,) = NoisySimulator(NoiseModel(), seed=0).execute_circuits(
        [circuit], 200000
    )
    parity = sum(
        count * (-1) ** (int(b[0]) + int(b[1])) for b, count in frequencies.items()
    )
    state = Circuit(2)
    state.add(gates.RY(0, theta=1.1))
    state.add(gates.CNOT(0, 1))
    exact = pauli_string_to_hamiltonian("XZ").expectation_from_state(state().state())
    assert parity / 200000 == pytest.approx(exact, abs=0.01)


@pytest.mark.parametrize("gate", [gates.SX, gates.SXDG])
def test_hsmpo_runs_sqrt_x_gates(gate):
    circuit = Circuit(2)
    circuit.add(gates.RY(0, theta=0.3))
    circuit.add(gate(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gate(1))
    circuit.add(gates.RX(1, theta=0.7))
    state = circuit.copy(deep=True)().state()
    surrogate = HSMPO(circuit, progress=False)
    for observable in ("ZI", "XY", "YZ"):
        exact = pauli_string_to_hamiltonian(observable).expectation_from_state(state)
        assert surrogate.expectation(observable) == pytest.approx(exact, abs=1e-10)
