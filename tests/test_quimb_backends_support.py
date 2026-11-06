import traceback

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from qibo import Circuit, gates

from mpstab.engines import QuimbEngine, StimEngine
from mpstab.engines.tensor_networks._jax_svd import safe_svd
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_backend_support(backend):

    nqubits = 10
    nlayers = 3
    observable = "ZX" * int(nqubits / 2)

    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
    hs = HSMPO(ansatz)

    hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine(backend=backend))

    expectation = hs.expectation(observable)

    assert expectation is not None


def _non_clifford_circuit(nqubits, nlayers, rng):
    """Rotations at generic angles, so every one of them is a magic gate."""
    circuit = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, theta=rng.uniform(0.3, 1.2)))
            circuit.add(gates.RZ(q, theta=rng.uniform(0.3, 1.2)))
        for q in range(nqubits - 1):
            circuit.add(gates.CNOT(q, q + 1))
    return circuit


def _dense_expectation(circuit, parameters, observable):
    """Exact <observable> from the statevector, in float64."""
    circuit = circuit.copy(deep=True)
    circuit.set_parameters(parameters)
    state = circuit().state(numpy=True)
    paulis = {"Z": np.diag([1.0, -1.0]), "X": np.array([[0.0, 1.0], [1.0, 0.0]])}
    operator = np.array([[1.0]])
    for pauli in observable:
        operator = np.kron(operator, paulis[pauli])
    return float(np.real(state.conj() @ operator @ state))


def test_gradients_jax():
    """
    ``jax.grad`` through ``HSMPO.set_parameters`` + ``expectation`` matches
    central finite differences of the exact statevector expectation.

    The angles are generic (non-Clifford) on purpose: a Clifford gate is folded
    into the stabilizer frame and cannot carry a traced parameter. Only jax is
    covered, since torch's ``backward`` cannot differentiate through the complex
    SVD the MPS evolution uses.
    """
    nqubits, nlayers, observable = 4, 2, "ZXZX"
    circuit = _non_clifford_circuit(nqubits, nlayers, np.random.default_rng(0))
    hs = HSMPO(CircuitAnsatz(qibo_circuit=circuit))
    hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine(backend="jax"))

    def f(parameters):
        hs.set_parameters(parameters)
        return hs.expectation(observable)

    parameters = np.array(circuit.get_parameters()).ravel()
    gradient = np.array(jax.grad(f)(parameters))

    if np.isnan(gradient).any():
        # Seen on one CI runner only: find the primitive that creates the first nan.
        try:
            with jax.debug_nans(True):
                jax.grad(f)(parameters)
            where = "debug_nans did not raise"
        except Exception:  # noqa: BLE001 - diagnostics only
            where = "\n".join(traceback.format_exc().splitlines()[-70:])
        pytest.fail(
            f"nan in {int(np.isnan(gradient).sum())}/{gradient.size} gradient "
            f"entries; jax {jax.__version__}\n{where}"
        )

    eps = 1e-5
    expected = np.array(
        [
            (
                _dense_expectation(circuit, parameters + eps * e, observable)
                - _dense_expectation(circuit, parameters - eps * e, observable)
            )
            / (2 * eps)
            for e in np.eye(len(parameters))
        ]
    )
    np.testing.assert_allclose(gradient, expected, atol=1e-4)


def _svd_loss(svd, matrix, weights):
    u, s, vh = svd(matrix, full_matrices=False)
    return jnp.real(jnp.sum(jnp.abs(u * s @ vh * weights) ** 2)) + jnp.sum(
        s * jnp.arange(s.size)
    )


@pytest.mark.parametrize("shape", [(4, 4), (6, 3), (3, 6)])
@pytest.mark.parametrize("is_complex", [False, True])
def test_safe_svd_gradient_matches_jax_on_generic_matrices(shape, is_complex):
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal(shape)
    if is_complex:
        matrix = matrix + 1j * rng.standard_normal(shape)
    weights = rng.standard_normal(shape)

    expected = jax.grad(lambda x: _svd_loss(jnp.linalg.svd, x, weights))(matrix)
    actual = jax.grad(lambda x: _svd_loss(safe_svd, x, weights))(matrix)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("zeros", [1, 2, 3])
def test_safe_svd_gradient_is_finite_on_repeated_zero_singular_values(zeros):
    """
    An MPS evolved from a product state is compressed through matrices like these
    at its first rotations. jax's own SVD derivative is ``nan`` there, which made
    the first-layer gradients ``nan`` on one CI runner.
    """
    matrix = np.diag([1.0] + [0.0] * zeros + [0.5] * (3 - zeros))
    matrix[1 : 1 + zeros, 1 : 1 + zeros] = 0.0
    weights = np.random.default_rng(1).standard_normal((4, 4))
    gradient = jax.grad(lambda x: _svd_loss(safe_svd, x, weights))(matrix)
    assert np.all(np.isfinite(np.asarray(gradient)))
