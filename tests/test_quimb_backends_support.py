import contextlib
import importlib.util
import traceback

import numpy as np
import pytest
from qibo import Circuit, gates
from utils import jax_x64

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mpstab.engines import QuimbEngine, StimEngine
from mpstab.engines.tensor_networks._jax_linalg import safe_qr, safe_svd
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient

#: torch is an optional backend: its cases are skipped when it is missing.
_TORCH = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


@pytest.mark.parametrize("backend", ["jax", pytest.param("torch", marks=_TORCH)])
def test_backend_support(backend):
    """Each backend reproduces the numpy engine's value, not just runs."""
    nqubits = 6
    nlayers = 3
    observable = "ZX" * int(nqubits / 2)

    np.random.seed(0)
    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
    hs = HSMPO(ansatz)
    reference = hs.expectation(observable)

    # jax defaults to single precision, whose error varies across platforms
    # (1e-6 to 2e-5), so it runs in double precision here. Even then its
    # safe_qr nudges degenerate factorizations by sqrt(eps) ~ 1.5e-8 to keep
    # gradients finite, which leaves value errors up to ~1e-8 depending on the
    # platform's linear algebra. torch stays in single precision.
    precision = jax_x64() if backend == "jax" else contextlib.nullcontext()
    tolerance = 1e-7 if backend == "jax" else 1e-4
    with precision:
        hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine(backend=backend))
        expectation = float(np.real(np.asarray(hs.expectation(observable))))

    assert expectation == pytest.approx(reference, abs=tolerance)
    assert abs(reference) > 1e-3  # a vanishing value would make this vacuous


def test_autodiff_backends_have_no_site_view():
    """jax keeps quimb's differentiable path: the NumPy site view is off."""
    engine = QuimbEngine(backend="jax")
    assert not engine.has_site_view and not engine.has_operator_sites


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
    # float64: where jax's QR derivative is degenerate, safe_qr nudges the input by
    # sqrt(eps), which is 1e-8 here but ~3e-4 in float32.
    with jax_x64():
        nqubits, nlayers, observable = 4, 2, "ZXZX"
        circuit = _non_clifford_circuit(nqubits, nlayers, np.random.default_rng(0))
        hs = HSMPO(
            CircuitAnsatz(qibo_circuit=circuit),
            stab_engine=StimEngine(),
            tn_engine=QuimbEngine(backend="jax"),
        )

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


def _qr_loss(qr, matrix, weights_q, weights_r):
    q, r = qr(matrix)
    return jnp.real(jnp.sum(q * weights_q) + jnp.sum(r * weights_r))


@pytest.mark.parametrize("shape", [(4, 4), (6, 3)])
@pytest.mark.parametrize("is_complex", [False, True])
def test_safe_qr_gradient_matches_jax_on_generic_matrices(shape, is_complex):
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal(shape)
    if is_complex:
        matrix = matrix + 1j * rng.standard_normal(shape)
    weights_q = rng.standard_normal((shape[0], shape[1]))
    weights_r = rng.standard_normal((shape[1], shape[1]))

    expected = jax.grad(lambda x: _qr_loss(jnp.linalg.qr, x, weights_q, weights_r))(
        matrix
    )
    actual = jax.grad(lambda x: _qr_loss(safe_qr, x, weights_q, weights_r))(matrix)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("zero_column", [0, 1, 2])
def test_safe_qr_gradient_is_finite_on_a_zero_column(zero_column):
    """
    A zero column puts a zero on ``R``'s diagonal, and jax's own QR derivative then
    returns ``nan`` (``invalid value (nan) encountered in triangular_solve``), which
    is what made the first-layer gradients ``nan`` on the macOS CI runner.
    """
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((4, 3))
    matrix[:, zero_column] = 0.0
    weights_q = rng.standard_normal((4, 3))
    weights_r = rng.standard_normal((3, 3))

    gradient = jax.grad(lambda x: _qr_loss(safe_qr, x, weights_q, weights_r))(matrix)
    assert np.all(np.isfinite(np.asarray(gradient)))


@pytest.mark.parametrize("zero_column", [0, 1, 2])
def test_safe_qr_gradient_is_accurate_in_float64(zero_column):
    """``Q @ R`` is the input, so the gradient of ``sum(w * Q @ R)`` must be ``w``."""
    rng = np.random.default_rng(2)
    matrix = rng.standard_normal((5, 3))
    matrix[:, zero_column] = 0.0
    weights = rng.standard_normal((5, 3))

    def product(x):
        q, r = safe_qr(x)
        return jnp.sum(weights * (q @ r))

    with jax_x64():
        gradient = jax.grad(product)(jnp.asarray(matrix))
    np.testing.assert_allclose(np.asarray(gradient), weights, atol=1e-6)


@pytest.mark.parametrize("shape", [(3, 7), (4, 5), (1, 3)])
def test_safe_qr_gradient_on_wide_matrices(shape):
    """jax has no QR derivative for wide matrices; safe_qr provides one."""
    rng = np.random.default_rng(5)
    with jax_x64():
        a = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
        q, r = safe_qr(a)
        np.testing.assert_allclose(q @ r, a, atol=1e-12)
        np.testing.assert_allclose(jnp.triu(r), r, atol=1e-14)

        weights = jnp.asarray(rng.normal(size=shape))

        def loss(x):
            q, r = safe_qr(x)
            return jnp.sum(weights * jnp.abs(q @ r) ** 2)

        grad = jax.grad(loss)(jnp.real(a))
        # q r = x, so the loss is sum(w x^2) and its gradient 2 w x.
        np.testing.assert_allclose(grad, 2 * weights * jnp.real(a), atol=1e-10)
