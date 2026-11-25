"""Tests for the variance and cost reductions of the noisy estimator."""

import numpy as np
import pytest
from qibo import Circuit, gates

from mpstab import HSMPO
from mpstab.engines import QuimbEngine
from mpstab.engines.tensor_networks import _mps_kernels as kernels
from mpstab.models.ansatze import HardwareEfficient, KickedIsing
from mpstab.models.noise import (
    AmplitudeDampingError,
    DepolarizingError,
    NoiseModel,
    PauliError,
    ThermalRelaxationError,
)
from mpstab.models.noise.compile import (
    _transfer_matrix,
    compile_noise,
    pauli_reset_mixture,
)
from mpstab.models.noise.events import KrausChannel
from mpstab.models.noise.mc_noise_estimator import noisy_expectation
from mpstab.pauli import PAULI_MATRICES, pauli_matrix

OBSERVABLE = {"ZZIII": 0.7, "IXIXI": -0.4, "IIYZI": 0.3}


def _exact(noise, circuit, observable):
    rho = noise.apply(circuit)().state()
    return sum(
        c * np.real(np.trace(rho @ pauli_matrix(p))) for p, c in observable.items()
    )


def _hardware_efficient():
    np.random.seed(3)
    return HardwareEfficient(nqubits=5, nlayers=2)


def _kicked_ising():
    return KickedIsing(nqubits=5, n_steps=2, theta_h=0.8, theta_z=0.3, theta_j=0.4)


# -- Positive mixtures ------------------------------------------------------------------


def test_thermal_relaxation_is_a_pauli_reset_mixture_when_t2_below_t1():
    error = ThermalRelaxationError(t1=1.0, t2=0.6, time=0.4, excited_population=0.2)
    mixture = pauli_reset_mixture(error.kraus)
    assert mixture is not None
    assert sum(mixture.values()) == pytest.approx(1.0)
    gamma = 1 - np.exp(-0.4)
    # The lightest mixture resets only by the net translation, gamma (1 - 2 p_e).
    assert mixture["R0"] == pytest.approx(gamma * (1 - 2 * 0.2))
    assert "R1" not in mixture
    # Lighter than the textbook form, resets with probability gamma.
    lambda_perp = np.exp(-0.4 / 0.6)
    textbook_identity = 1 - gamma - ((1 - gamma) - lambda_perp) / 2
    assert mixture["I"] > textbook_identity
    resets = {
        "R0": (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
        "R1": (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 0]])),
    }
    rebuilt = sum(
        p
        * _transfer_matrix(
            resets[label] if label in resets else (PAULI_MATRICES[label],)
        )
        for label, p in mixture.items()
    )
    assert rebuilt == pytest.approx(_transfer_matrix(error.kraus), abs=1e-12)


def _depolarizing_kraus(p):
    return (np.sqrt(1 - 3 * p / 4) * np.eye(2),) + tuple(
        np.sqrt(p / 4) * PAULI_MATRICES[label] for label in "XYZ"
    )


@pytest.mark.parametrize(
    "kraus",
    [
        ThermalRelaxationError(t1=1.0, t2=1.5, time=0.4).kraus,  # T2 > T1
        AmplitudeDampingError(0.1).kraus,  # T2 = 2 T1
        _depolarizing_kraus(0.1),  # a Pauli channel: nothing to gain
    ],
)
def test_channels_that_are_not_reset_mixtures(kraus):
    assert pauli_reset_mixture(kraus) is None


@pytest.mark.parametrize("make_ansatz", [_hardware_efficient, _kicked_ising])
def test_reset_mixture_matches_density_matrix_with_less_weight(make_ansatz):
    ansatz = make_ansatz()
    noise = NoiseModel().add(
        ThermalRelaxationError(t1=1.0, t2=0.7, time=0.01, excited_population=0.1)
    )
    exact = _exact(noise, ansatz.circuit, OBSERVABLE)
    hsmpo = HSMPO(ansatz)
    mixed = hsmpo.noisy_expectation(OBSERVABLE, noise, n_samples=2000, seed=1)
    quasi = hsmpo.noisy_expectation(
        OBSERVABLE, noise, n_samples=2000, seed=1, positive_mixtures=False
    )
    for result in (mixed, quasi):
        assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10
    # Probabilities instead of a quasi-probability: a lighter sampled stratum.
    assert sum(mixed.stratum_weights) == pytest.approx(1.0)
    assert sum(quasi.stratum_weights) > 1.0
    assert mixed.stratum_weights[2] < quasi.stratum_weights[2]


def test_single_reset_block_is_exact_anywhere():
    """One relaxation at every position, compiled as a mixture: k <= 1 is exact."""
    ansatz = _kicked_ising()
    circuit = ansatz.circuit
    hsmpo = HSMPO(circuit, tn_engine=QuimbEngine(svd_cutoff=0.0))
    error = ThermalRelaxationError(t1=1.0, t2=0.5, time=0.3, excited_population=0.3)
    for index in (-1, 3, 11, len(circuit.queue) - 2):
        noise_events = [KrausChannel(index, (2,), error.kraus)]
        compiled = compile_noise(hsmpo, noise_events)
        assert len(compiled.block_probs) == 1
        result = noisy_expectation(compiled, OBSERVABLE)
        exact = _events_exact(circuit, noise_events, OBSERVABLE)
        assert result.mean == pytest.approx(exact, abs=1e-10)


def _events_exact(circuit, events, observable):
    from mpstab.models.noise.model import _qibo_gate

    noisy = Circuit(circuit.nqubits, density_matrix=True)
    by_gate = {}
    for event in events:
        by_gate.setdefault(event.after_gate, []).append(event)
    for event in by_gate.get(-1, ()):
        noisy.add(_qibo_gate(event))
    for index, gate in enumerate(circuit.queue):
        noisy.add(gate)
        for event in by_gate.get(index, ()):
            noisy.add(_qibo_gate(event))
    rho = noisy().state()
    return sum(
        c * np.real(np.trace(rho @ pauli_matrix(p))) for p, c in observable.items()
    )


# -- Control variates --------------------------------------------------------------------


def test_multiplicative_model_is_exact_on_clifford_circuits():
    """Without rotations every fault only flips signs: the model is the value."""
    circuit = Circuit(4)
    for q in range(4):
        circuit.add(gates.H(q))
    for q in range(3):
        circuit.add(gates.CNOT(q, q + 1))
    circuit.add(gates.S(2))
    noise = NoiseModel().add(DepolarizingError(0.3))
    observable = {"XXYX": 0.6, "XIII": 0.4}
    exact = _exact(noise, circuit, observable)
    hsmpo = HSMPO(circuit)
    result = hsmpo.noisy_expectation(observable, noise, n_samples=500, seed=0)
    plain = hsmpo.noisy_expectation(
        observable, noise, n_samples=500, seed=0, control_variates=False
    )
    assert result.stratum_weights[2] > 0.1
    assert result.mean == pytest.approx(exact, abs=1e-10)
    assert result.stderr < 1e-10
    assert plain.stderr > 1e-3


@pytest.mark.parametrize("seed", range(4))
def test_control_variates_are_unbiased_and_reduce_variance(seed):
    ansatz = _hardware_efficient()
    noise = (
        NoiseModel()
        .add(DepolarizingError(0.03))
        .add(PauliError({"ZZ": 0.01}), gates.CZ)
    )
    exact = _exact(noise, ansatz.circuit, OBSERVABLE)
    result = HSMPO(ansatz).noisy_expectation(
        OBSERVABLE, noise, n_samples=400, seed=seed
    )
    assert result.stratum_weights[2] > 0.05
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10
    assert result.variance_reduction > 1.0


def test_control_variates_with_coherences_are_unbiased():
    ansatz = _kicked_ising()
    noise = NoiseModel().add(AmplitudeDampingError(0.03)).add(DepolarizingError(0.01))
    exact = _exact(noise, ansatz.circuit, OBSERVABLE)
    result = HSMPO(ansatz).noisy_expectation(OBSERVABLE, noise, n_samples=1500, seed=5)
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


# -- Sampling inside k = 1 -------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_sampled_order_one_tail_is_unbiased(seed):
    ansatz = _kicked_ising()
    noise = NoiseModel().add(DepolarizingError(0.01)).add(AmplitudeDampingError(0.01))
    exact = _exact(noise, ansatz.circuit, OBSERVABLE)
    hsmpo = HSMPO(ansatz)
    result = hsmpo.noisy_expectation(
        OBSERVABLE, noise, n_samples=1000, seed=seed, order_one_samples=30
    )
    full = hsmpo.noisy_expectation(OBSERVABLE, noise, n_samples=1000, seed=seed)
    assert result.order_one_samples == 30
    assert result.stderr > full.stderr
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


def test_order_one_bound_skips_faults_that_move_nothing():
    """A fault flipping no rotation and only the observable needs no MPS."""
    circuit = Circuit(2)
    circuit.add(gates.RY(0, theta=0.4))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.H(1))
    hsmpo = HSMPO(circuit)
    noise = NoiseModel().add(PauliError({"X": 0.01, "Z": 0.02}), gates.H)
    result = hsmpo.noisy_expectation("ZX", noise, n_samples=0, order_one_samples=10)
    assert result.order_one_samples == 0
    assert result.n_patterns == 1
    assert result.mean == pytest.approx(_exact(noise, circuit, {"ZX": 1.0}), abs=1e-12)


# -- Evaluation --------------------------------------------------------------------------


def test_parallel_evaluation_matches_serial():
    ansatz = _kicked_ising()
    noise = NoiseModel().add(AmplitudeDampingError(0.02)).add(DepolarizingError(0.01))
    hsmpo = HSMPO(ansatz)
    serial = hsmpo.noisy_expectation(OBSERVABLE, noise, n_samples=200, seed=2)
    parallel = hsmpo.noisy_expectation(
        OBSERVABLE, noise, n_samples=200, seed=2, n_workers=2
    )
    assert parallel.mean == pytest.approx(serial.mean, abs=1e-13)
    assert parallel.stderr == pytest.approx(serial.stderr, abs=1e-13)
    assert parallel.n_transitions == serial.n_transitions > 0


def test_transition_skips_shared_isometries():
    engine = QuimbEngine()
    rng = np.random.default_rng(0)
    circuit = Circuit(8)
    for layer in range(3):
        for q in range(8):
            circuit.add(gates.RY(q, theta=rng.uniform(0, 3)))
        for q in range(layer % 2, 7, 2):
            circuit.add(gates.CNOT(q, q + 1))
    ket = engine.working_state(engine.build_circuit_mps(8, circuit))
    bra = engine.copy_state(ket)
    engine.apply_rotation(bra, engine.rotation_operator("IIIXZIII", 0.7), None)
    for pauli in ("IIIIIIII", "IIIZIIII", "ZIIIIIIX", "IIYXIIII"):
        centers = (bra.center, ket.center)
        fast = kernels.pauli_transition(bra.arrays, ket.arrays, pauli, centers=centers)
        slow = kernels.pauli_transition(bra.arrays, ket.arrays, pauli)
        assert fast == pytest.approx(slow, abs=1e-13)
