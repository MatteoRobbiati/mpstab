"""Tests for the analytic gradient of the HSMPO surrogate."""

from functools import reduce

import numpy as np
import pytest
from qibo import Circuit, gates, symbols
from qibo.derivative import parameter_shift
from qibo.hamiltonians import SymbolicHamiltonian
from utils import construct_symbolic_hamiltonian

from mpstab import HSMPO
from mpstab.engines import QuimbEngine, StimEngine
from mpstab.evolutors.gradient import analytic_gradient
from mpstab.models.ansatze import HardwareEfficient
from mpstab.pauli import PAULI_MATRICES

OBSERVABLE = {"ZZIII": 0.7, "IXIXI": -0.4, "IIYZI": 0.3, "XIIIZ": 0.2}


def _matrix(pauli: str) -> np.ndarray:
    return reduce(np.kron, [PAULI_MATRICES[label] for label in pauli])


def _hamiltonian(observable: dict) -> SymbolicHamiltonian:
    """A ``{pauli_string: coefficient}`` observable as a qibo Hamiltonian."""
    form = 0
    for pauli, coefficient in observable.items():
        term = coefficient
        for qubit, label in enumerate(pauli):
            if label != "I":
                term *= getattr(symbols, label)(qubit)
        form += term
    return SymbolicHamiltonian(form, nqubits=len(next(iter(observable))))


def _exact_gradient(circuit: Circuit, observable) -> np.ndarray:
    """qibo's parameter-shift rule on its statevector, which is exact."""
    if not isinstance(observable, SymbolicHamiltonian):
        observable = _hamiltonian(observable)
    return np.array(
        [
            parameter_shift(circuit, observable, i)
            for i in range(len(circuit.get_parameters()))
        ]
    )


def _generic_circuit(nqubits: int = 5, nlayers: int = 3, seed: int = 0) -> Circuit:
    """Every supported rotation, generic angles, and entangling Cliffords."""
    rng = np.random.default_rng(seed)
    circuit = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, theta=rng.uniform(-np.pi, np.pi)))
            circuit.add(gates.RZ(q, theta=rng.uniform(-np.pi, np.pi)))
        for q in range(nqubits - 1):
            circuit.add(gates.CNOT(q, q + 1))
        circuit.add(gates.H(0))
        circuit.add(gates.S(2))
        circuit.add(gates.RX(1, theta=rng.uniform(-np.pi, np.pi)))
        circuit.add(gates.RZZ(1, 3, theta=rng.uniform(-np.pi, np.pi)))
        circuit.add(gates.RXX(0, nqubits - 1, theta=rng.uniform(-np.pi, np.pi)))
        circuit.add(gates.RYY(2, 3, theta=rng.uniform(-np.pi, np.pi)))
        circuit.add(gates.T(nqubits - 1))
    return circuit


METHODS = ["adjoint", "flips"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("light_cone", [True, False])
def test_gradient_generic_chain(light_cone, method):
    circuit = _generic_circuit()
    hsmpo = HSMPO(circuit)
    gradient = hsmpo.gradient(OBSERVABLE, light_cone=light_cone, method=method)
    np.testing.assert_allclose(
        gradient, _exact_gradient(circuit, OBSERVABLE), atol=1e-10
    )


@pytest.mark.parametrize("method", METHODS)
def test_gradient_symbolic_hamiltonian(method):
    """Several terms with a constant, as a qibo SymbolicHamiltonian."""
    circuit = _generic_circuit(nlayers=2, seed=1)
    hamiltonian = construct_symbolic_hamiltonian(nqubits=5, n_terms=6)
    np.testing.assert_allclose(
        HSMPO(circuit).gradient(hamiltonian, method=method),
        _exact_gradient(circuit, hamiltonian),
        atol=1e-10,
    )


def test_gradient_parameter_subset_and_order():
    circuit = _generic_circuit(nlayers=2, seed=2)
    hsmpo = HSMPO(circuit)
    full = hsmpo.gradient(OBSERVABLE)
    subset = [7, 2, 11]
    np.testing.assert_allclose(
        hsmpo.gradient(OBSERVABLE, parameters=subset), full[subset], atol=1e-12
    )


def test_gradient_commuting_tail_needs_no_new_state():
    """A rotation commuting with every later one costs no shifted MPS."""
    circuit = Circuit(3)
    circuit.add(gates.RY(0, theta=0.4))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RX(1, theta=0.9))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.RZ(0, theta=1.1))  # commutes with the RY(2) after it
    circuit.add(gates.RY(2, theta=-0.7))
    observable = {"XZI": 0.6, "YIX": -0.3, "IYZ": 0.5}
    hsmpo = HSMPO(circuit)

    result = analytic_gradient(hsmpo, observable, parameters=[2, 3], method="flips")
    # Only the base chain is built: neither rotation flips a later one.
    assert result.n_rotations == 4
    assert len(result.fidelities) == 1
    np.testing.assert_allclose(
        result.values, _exact_gradient(circuit, observable)[[2, 3]], atol=1e-10
    )


def test_gradient_outside_light_cone_is_zero():
    circuit = Circuit(3)
    circuit.add(gates.RY(0, theta=0.4))
    circuit.add(gates.RX(2, theta=0.9))  # never reaches qubit 0
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RZ(1, theta=0.3))
    hsmpo = HSMPO(circuit)
    gradient = hsmpo.gradient("ZII")
    assert gradient[1] == 0.0
    np.testing.assert_allclose(
        gradient, _exact_gradient(circuit, {"ZII": 1.0}), atol=1e-12
    )


@pytest.mark.parametrize("method", METHODS)
def test_gradient_dressing_sign(method):
    """H Y H = -Y: the dressed angle is minus the gate's, and so is d/dphi."""
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.RY(0, theta=0.8))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RX(1, theta=0.5))
    observable = {"ZZ": 1.0, "XY": 0.4}
    hsmpo = HSMPO(circuit)
    assert hsmpo._dress_generators(hsmpo.magic_gates, hsmpo.clifford_circuit)[0] == (
        "YI",
        -1,
    )
    np.testing.assert_allclose(
        hsmpo.gradient(observable, method=method),
        _exact_gradient(circuit, observable),
        atol=1e-12,
    )


@pytest.mark.parametrize("method", METHODS)
def test_gradient_gate_queued_twice_sums_its_rotations(method):
    """
    One parameter driving several rotations: a gate object queued twice. Both
    of its entries in ``get_parameters`` get the derivative with respect to
    its shared angle, the sum over its two rotations.
    """
    shared = gates.RY(0, theta=0.6)
    circuit = Circuit(2)
    circuit.add(shared)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RX(1, theta=0.3))
    circuit.add(shared)
    observable = {"ZX": 0.8, "XZ": -0.5}

    epsilon = 1e-6
    values = []
    for sign in (1, -1):
        shared.parameters = (0.6 + sign * epsilon,)
        values.append(_hamiltonian(observable).expectation(circuit().state()))
    shared.parameters = (0.6,)
    expected = (values[0] - values[1]) / (2 * epsilon)

    gradient = HSMPO(circuit).gradient(observable, method=method)
    np.testing.assert_allclose(gradient[[0, 2]], expected, atol=1e-8)


def test_gradient_decomposed_controlled_rotation():
    """
    A controlled rotation reaches HSMPO decomposed, one parameter per
    rotation: ``CRY(theta) = RY(theta/2) CX RY(-theta/2) CX``. The chain rule
    through the decomposition gives the controlled rotation's derivative.
    """
    theta = 1.3

    def circuit(theta_a, theta_b):
        circuit = Circuit(2)
        circuit.add(gates.RX(0, theta=0.7))
        circuit.add(gates.RY(1, theta=theta_a))
        circuit.add(gates.CNOT(0, 1))
        circuit.add(gates.RY(1, theta=theta_b))
        circuit.add(gates.CNOT(0, 1))
        return circuit

    observable = {"ZX": 0.8, "IZ": -0.5, "YY": 0.3}
    gradient = HSMPO(circuit(theta / 2, -theta / 2)).gradient(observable)
    derivative = 0.5 * gradient[1] - 0.5 * gradient[2]

    # qibo has no shift rule for a controlled rotation, whose generator has
    # three eigenvalues: compare with a central difference instead.
    epsilon = 1e-6
    values = []
    for sign in (1, -1):
        controlled = Circuit(2)
        controlled.add(gates.RX(0, theta=0.7))
        controlled.add(gates.CRY(0, 1, theta=theta + sign * epsilon))
        values.append(_hamiltonian(observable).expectation(controlled().state()))
    expected = (values[0] - values[1]) / (2 * epsilon)
    assert derivative == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("angle", [0.0, np.pi / 2, -np.pi])
def test_gradient_clifford_angle_raises(angle):
    circuit = Circuit(2)
    circuit.add(gates.RY(0, theta=0.4))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RZ(1, theta=angle))
    circuit.add(gates.RX(0, theta=0.9))
    hsmpo = HSMPO(circuit)
    with pytest.raises(ValueError, match="Clifford angle"):
        hsmpo.gradient("ZZ")
    # The others still differentiate.
    np.testing.assert_allclose(
        hsmpo.gradient("ZZ", parameters=[0, 2]),
        _exact_gradient(circuit, {"ZZ": 1.0})[[0, 2]],
        atol=1e-12,
    )


def test_gradient_parameter_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        HSMPO(_generic_circuit(nlayers=1)).gradient("ZZZZZ", parameters=[100])


@pytest.mark.parametrize("method", METHODS)
def test_gradient_finite_bond_dimension(method):
    """
    With a bond-dimension cap, the error tracks the truncation fidelities:
    each transition amplitude is off by at most ``2 sqrt(2 (1 - F))`` for
    normalized states of fidelity ``F``, times the summed coefficients.
    """
    np.random.seed(5)
    ansatz = HardwareEfficient(nqubits=10, nlayers=4)
    observable = {"Z" * 10: 0.5, "IXXIIIIIII": 1.0, "IIIIIZIIZI": -0.7}
    exact = _exact_gradient(ansatz.circuit, observable)

    hsmpo = HSMPO(ansatz, max_bond_dimension=8)
    result = analytic_gradient(hsmpo, observable, method=method)
    infidelity = 1 - result.fidelities.min()
    assert infidelity > 1e-3  # the cap does truncate here

    bound = sum(abs(c) for c in observable.values()) * 2 * np.sqrt(2 * infidelity)
    error = np.abs(result.values - exact).max()
    assert error > 1e-8
    assert error < bound

    # Without the cap or a singular-value cutoff the same chain is exact.
    hsmpo = HSMPO(ansatz)
    hsmpo.set_engines(tn_engine=QuimbEngine(svd_cutoff=0.0))
    np.testing.assert_allclose(
        hsmpo.gradient(observable, method=method), exact, atol=1e-9
    )


def test_gradient_adjoint_cost_is_independent_of_parameters():
    """One forward and one backward pass: at most 2 R rotations."""
    circuit = _generic_circuit(nlayers=3)
    hsmpo = HSMPO(circuit)
    result = analytic_gradient(hsmpo, OBSERVABLE, light_cone=False)
    assert result.n_rotations <= 2 * hsmpo.n_magic_gates - 1
    flips = analytic_gradient(hsmpo, OBSERVABLE, light_cone=False, method="flips")
    assert flips.n_rotations > result.n_rotations
    np.testing.assert_allclose(result.values, flips.values, atol=1e-10)


def test_gradient_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        HSMPO(_generic_circuit(nlayers=1)).gradient(OBSERVABLE, method="psr")


@pytest.mark.parametrize("method", METHODS)
def test_gradient_jax_backend_matches_numpy(method):
    """quimb's full-contraction fallback for the non-numpy backends."""
    jax = pytest.importorskip("jax")
    circuit = _generic_circuit(nlayers=1, seed=3)
    hsmpo = HSMPO(circuit)
    reference = hsmpo.gradient(OBSERVABLE, method=method)
    with jax.enable_x64(True):
        hsmpo.set_engines(
            stab_engine=StimEngine(), tn_engine=QuimbEngine(backend="jax")
        )
        gradient = hsmpo.gradient(OBSERVABLE, method=method)
    np.testing.assert_allclose(gradient, reference, atol=1e-7)


def test_pauli_transitions_match_dense_states():
    engine = QuimbEngine(svd_cutoff=0.0)
    circuits = [_generic_circuit(nqubits=4, nlayers=2, seed=s) for s in (4, 5)]
    states = []
    for circuit in circuits:
        hsmpo = HSMPO(circuit)
        hsmpo.set_engines(tn_engine=engine)
        states.append(hsmpo.original_circuit_mps)
    paulis = ["ZIXY", "IIII", "YYZX"]
    dense = [state.to_dense().ravel() for state in states]
    dense = [vector / np.linalg.norm(vector) for vector in dense]
    expected = [np.vdot(dense[0], _matrix(p) @ dense[1]) for p in paulis]
    np.testing.assert_allclose(
        engine.pauli_transitions(states[0], states[1], paulis), expected, atol=1e-12
    )


@pytest.mark.parametrize("method", METHODS)
def test_gradient_return_value(method):
    """The value comes from the gradient's own pass and equals expectation()."""
    hsmpo = HSMPO(_generic_circuit(nlayers=2, seed=6))
    value, gradient = hsmpo.gradient(OBSERVABLE, method=method, return_value=True)
    assert value == pytest.approx(hsmpo.expectation(OBSERVABLE), abs=1e-10)
    np.testing.assert_allclose(gradient, hsmpo.gradient(OBSERVABLE), atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("groups", [2, 3, 100])
def test_gradient_groups(method, groups):
    """Blocks of terms on their own light cones sum to the union's gradient."""
    hsmpo = HSMPO(_generic_circuit(nlayers=2, seed=6))
    value, gradient = hsmpo.gradient(
        OBSERVABLE, method=method, groups=groups, return_value=True
    )
    reference, expected = hsmpo.gradient(OBSERVABLE, method=method, return_value=True)
    assert value == pytest.approx(reference, abs=1e-10)
    np.testing.assert_allclose(gradient, expected, atol=1e-10)


def test_set_parameters_rebuilds_the_mps_lazily():
    """After set_parameters the gradient needs no MPS; expectation rebuilds it."""
    np.random.seed(7)
    hsmpo = HSMPO(HardwareEfficient(nqubits=5, nlayers=2))
    parameters = np.random.uniform(-np.pi, np.pi, hsmpo.nparams)
    hsmpo.set_parameters(parameters)
    assert "original_circuit_mps" not in hsmpo.__dict__
    value, gradient = hsmpo.gradient(OBSERVABLE, return_value=True)
    assert "original_circuit_mps" not in hsmpo.__dict__

    fresh = HSMPO(hsmpo.ansatz)
    assert hsmpo.expectation(OBSERVABLE) == pytest.approx(value, abs=1e-10)
    assert value == pytest.approx(fresh.expectation(OBSERVABLE), abs=1e-10)
    np.testing.assert_allclose(gradient, fresh.gradient(OBSERVABLE), atol=1e-10)
