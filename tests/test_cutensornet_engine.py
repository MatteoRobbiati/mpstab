"""
The cuTensorNet engine against quimb.

Without a GPU, the engine runs on a NumPy stand-in for cupy and cuTensorNet
(``decompose`` by NumPy's QR and SVD, ``contract`` by ``np.einsum``) with the
call signatures of cuquantum's Python API, passed to it as its ``cuquantum``
argument: that checks everything the engine does on top of those calls, in
every environment. The ``gpu`` tests rerun
the comparisons on the real stack and are skipped when cupy, cuquantum or a
CUDA device is missing.
"""

import contextlib
import dataclasses
import functools
import pickle

import numpy as np
import pytest
from qibo import Circuit, gates
from utils import small_entangled_circuit

from mpstab import HQSMPO, HSMPO
from mpstab.engines import CuTensorNetEngine, QuimbEngine, SiteMPO, SiteMPS
from mpstab.engines.tensor_networks import cutensornet
from mpstab.engines.tensor_networks.cutensornet import CuQuantumAPI
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient, KickedIsing
from mpstab.models.noise import (
    AmplitudeDampingError,
    DepolarizingError,
    NoiseModel,
    mc_noise_estimator,
)

# -- a NumPy stand-in for cupy and cuTensorNet ----------------------------------


class _QRMethod:
    pass


class _SVDMethod:
    pass


def _decompose(subscripts, matrix, method=None, options=None):
    assert subscripts == "ij->ik,kj"
    if isinstance(method, _QRMethod):
        return np.linalg.qr(matrix)
    return np.linalg.svd(matrix, full_matrices=False)


def _contract(*operands, options=None):
    """``cuquantum.contract`` in interleaved form, by ``np.einsum``."""
    *pairs, output = operands
    labels = {}
    for modes in pairs[1::2] + [output]:
        for mode in modes:
            labels.setdefault(mode, len(labels))
    args = []
    for tensor, modes in zip(pairs[::2], pairs[1::2]):
        args += [tensor, [labels[m] for m in modes]]
    return np.einsum(*args, [labels[m] for m in output])


class _FakeCuda:
    """The bits of ``cupy.cuda`` the engine uses, with a settable device."""

    current = 0

    class runtime:
        @staticmethod
        def getDevice():
            return _FakeCuda.current

    @staticmethod
    def Device(device):
        return contextlib.nullcontext()


class _NumpyAsCupy:
    """cupy's array API, on numpy. Module-level, so it pickles to workers."""

    cuda = _FakeCuda
    asnumpy = staticmethod(np.asarray)

    def __getattr__(self, name):
        return getattr(np, name)


_NUMPY_CUQUANTUM = CuQuantumAPI(
    cupy=_NumpyAsCupy(),
    contract=_contract,
    decompose=_decompose,
    QRMethod=_QRMethod,
    SVDMethod=_SVDMethod,
)


def _gpu_available() -> bool:
    if cutensornet.cupy is None or cutensornet.decompose is None:
        return False
    try:
        return cutensornet.cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # cupy installed, but no usable CUDA driver or device
        return False


@pytest.fixture(
    params=[
        "numpy-stand-in",
        pytest.param(
            "gpu",
            marks=pytest.mark.skipif(
                not _gpu_available(), reason="needs cupy, cuquantum and a GPU"
            ),
        ),
    ]
)
def make_engine(request):
    """Builds CuTensorNetEngine instances on the stand-in, or on the GPU."""
    if request.param == "numpy-stand-in":
        return functools.partial(CuTensorNetEngine, cuquantum=_NUMPY_CUQUANTUM)
    return CuTensorNetEngine


def _models(ansatz, make_engine, cls=HSMPO, **kwargs):
    reference = cls(ansatz, tn_engine=QuimbEngine(svd_cutoff=0.0), **kwargs)
    model = cls(ansatz, tn_engine=make_engine(svd_cutoff=0.0), **kwargs)
    return reference, model


def _dense(sites):
    """The matrix of ``(left, right, out, in)`` operator site arrays."""
    matrix = np.ones((1, 1, 1))  # (out, in, bond)
    for site in sites:
        block = np.einsum("abl,lrkc->akbcr", matrix, site)
        a, k, b, c, r = block.shape
        matrix = block.reshape(a * k, b * c, r)
    return matrix[..., 0]


# -- tests ------------------------------------------------------------------------


def test_missing_dependencies_raise_install_hint():
    if cutensornet.cupy is not None and cutensornet.decompose is not None:
        pytest.skip("cupy and cuquantum are installed")
    with pytest.raises(ImportError, match=r"mpstab\[cutensornet\]"):
        CuTensorNetEngine()


def test_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="dtype"):
        CuTensorNetEngine(dtype="float64")


@pytest.mark.parametrize("max_bond", [None, 2])
def test_hsmpo_matches_quimb(make_engine, max_bond):
    reference, model = _models(
        HardwareEfficient(nqubits=6, nlayers=3),
        make_engine,
        max_bond_dimension=max_bond,
    )
    observable = {"ZZIXII": 0.7, "IIYIZX": -0.4, "IIIIII": 0.1}
    value, fidelity = model.expectation(observable, return_fidelity=True)
    expected, expected_fidelity = reference.expectation(
        observable, return_fidelity=True
    )
    assert value == pytest.approx(expected, abs=1e-10)
    assert fidelity == pytest.approx(expected_fidelity, abs=1e-10)
    assert isinstance(model.original_circuit_mps, SiteMPS)
    assert model.tn_engine.bond_dimension(
        model.original_circuit_mps
    ) == reference.tn_engine.bond_dimension(reference.original_circuit_mps)


def test_product_initial_state(make_engine):
    initial = Circuit(3)
    initial.add(gates.H(0))
    initial.add(gates.RY(2, 0.7))
    reference, model = _models(
        HardwareEfficient(nqubits=3, nlayers=2), make_engine, initial_state=initial
    )
    assert model.expectation("XIZ") == pytest.approx(reference.expectation("XIZ"))


def test_entangled_initial_state_rejected(make_engine):
    initial = Circuit(2)
    initial.add(gates.CNOT(0, 1))
    with pytest.raises(ValueError, match="one-qubit gates"):
        HSMPO(
            HardwareEfficient(nqubits=2, nlayers=1),
            initial_state=initial,
            tn_engine=make_engine(),
        )


def test_general_operator_expectation(make_engine):
    """A non-Pauli MPO goes through the cuTensorNet network contraction."""
    engine = make_engine(svd_cutoff=0.0)
    quimb = QuimbEngine(svd_cutoff=0.0)
    model = HSMPO(KickedIsing(nqubits=5), tn_engine=engine)
    reference = HSMPO(KickedIsing(nqubits=5), tn_engine=quimb)
    rotations = [("XYZII", 0.4), ("IZZXI", -0.9)]
    operator = engine.conjugate_operator_many(
        engine.pauli_mpo("ZIIIZ"), rotations, None
    )
    expected_operator = quimb.conjugate_operator_many(
        quimb.pauli_mpo("ZIIIZ"), rotations, None
    )
    assert isinstance(operator, SiteMPO) and operator.pauli is None
    np.testing.assert_allclose(
        _dense(engine.operator_site_arrays(operator)),
        _dense(quimb.operator_site_arrays(expected_operator)),
        atol=1e-12,
    )
    assert engine.expval(model.original_circuit_mps, operator) == pytest.approx(
        quimb.expval(reference.original_circuit_mps, expected_operator), abs=1e-10
    )


def test_hqsmpo_matches_quimb(make_engine):
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    reference, model = _models(ansatz, make_engine, cls=HQSMPO, max_bond_dimension=2)
    expected = reference.expectation_at_cut("ZZZZ", 2, method="simulation")
    result = model.expectation_at_cut("ZZZZ", 2, method="simulation")
    assert result.value == pytest.approx(expected.value, abs=1e-10)
    assert result.state_fidelity == pytest.approx(expected.state_fidelity, abs=1e-10)

    expected_tail = reference.tail_truncation("ZZZZ", 1, 1, exact=True)
    tail = model.tail_truncation("ZZZZ", 1, 1, exact=True)
    assert tail.fidelity_estimate == pytest.approx(
        expected_tail.fidelity_estimate, abs=1e-10
    )


def test_noisy_expectation_with_transitions(make_engine):
    """Amplitude damping needs transition amplitudes between two states."""
    reference, model = _models(KickedIsing(nqubits=4), make_engine)
    noise = NoiseModel().add(DepolarizingError(0.02), gates.CNOT)
    noise.add(AmplitudeDampingError(0.05), gates.CNOT)
    observable = {"ZZII": 0.7, "IXIX": -0.4}
    expected = reference.noisy_expectation(observable, noise, n_samples=50, seed=1)
    result = model.noisy_expectation(observable, noise, n_samples=50, seed=1)
    assert result.mean == pytest.approx(expected.mean, abs=1e-10)


def test_copies_are_independent(make_engine):
    engine = make_engine()
    model = HSMPO(HardwareEfficient(nqubits=4, nlayers=2), tn_engine=engine)
    state = model.original_circuit_mps
    before = engine.pauli_expvals(state, ["ZIII", "IXXI"])
    copy = engine.copy_state(state)
    engine.pauli_rot(copy, "XYZX", 0.8, None)
    np.testing.assert_allclose(engine.pauli_expvals(state, ["ZIII", "IXXI"]), before)


def test_single_precision(make_engine):
    """complex64 stays complex64 and agrees with quimb to float32 accuracy."""
    engine = make_engine(dtype="complex64", svd_cutoff=0.0)
    ansatz = HardwareEfficient(nqubits=5, nlayers=3)
    reference, model = (
        HSMPO(ansatz, tn_engine=tn) for tn in (QuimbEngine(svd_cutoff=0.0), engine)
    )
    state = model.original_circuit_mps
    assert {str(site.dtype) for site in state.sites} == {"complex64"}
    assert model.expectation("ZZIXI") == pytest.approx(
        reference.expectation("ZZIXI"), abs=1e-5
    )
    operator = engine.conjugate_operator_many(
        engine.pauli_mpo("ZIIIZ"), [("XYZII", 0.4)], None
    )
    assert {str(site.dtype) for site in operator.sites} == {"complex64"}


def test_mismatched_operator_raises(make_engine):
    engine = make_engine()
    state = HSMPO(HardwareEfficient(nqubits=4, nlayers=1), tn_engine=engine)
    operator = engine.conjugate_operator_many(
        engine.pauli_mpo("ZZZ"), [("XYI", 0.4)], None
    )
    with pytest.raises(ValueError, match="3 sites"):
        engine.expval(state.original_circuit_mps, operator)


def test_pickles(make_engine):
    engine = make_engine(svd_cutoff=1e-12)
    clone = pickle.loads(pickle.dumps(engine))
    assert clone.svd_cutoff == 1e-12 and clone.linalg is not None


def test_dmrg_is_unsupported(make_engine):
    model = HSMPO(HardwareEfficient(nqubits=3, nlayers=1), tn_engine=make_engine())
    with pytest.raises(NotImplementedError, match="CuTensorNetEngine does not"):
        model.minimize_expectation({"ZZI": 1.0}, bond_dims=[4], max_sweeps=1)


def test_handles_follow_the_device_and_are_destroyed():
    created, destroyed = [], []

    def create():
        created.append(_FakeCuda.current)
        return f"handle-{len(created)}"

    api = dataclasses.replace(
        _NUMPY_CUQUANTUM, create_handle=create, destroy_handle=destroyed.append
    )
    engine = CuTensorNetEngine(cuquantum=api)
    assert created == []  # handles are created on first use only
    HSMPO(HardwareEfficient(nqubits=3, nlayers=2), tn_engine=engine)
    assert created == [0]
    _FakeCuda.current = 1
    try:
        assert engine.linalg.options == {"handle": "handle-2"}
    finally:
        _FakeCuda.current = 0
    assert created == [0, 1]
    engine.linalg.close()
    assert sorted(destroyed) == ["handle-1", "handle-2"]


def test_workers_start_by_spawn():
    """A forked child cannot use the parent's CUDA context."""
    engine = CuTensorNetEngine(cuquantum=_NUMPY_CUQUANTUM)
    assert mc_noise_estimator._worker_context(engine).get_start_method() == "spawn"
    assert mc_noise_estimator._worker_context(QuimbEngine()) is None


def test_noisy_expectation_on_spawned_workers():
    """The engine pickles into spawned processes and gives the serial result."""
    ansatz = HardwareEfficient(nqubits=4, nlayers=2)
    reference, model = _models(
        ansatz, functools.partial(CuTensorNetEngine, cuquantum=_NUMPY_CUQUANTUM)
    )
    noise = NoiseModel().add(DepolarizingError(0.05))
    expected = reference.noisy_expectation("ZZZZ", noise, n_samples=20, seed=2)
    assert expected.n_patterns > 1  # so there is work to split across workers
    result = model.noisy_expectation("ZZZZ", noise, n_samples=20, seed=2, n_workers=2)
    assert result.mean == pytest.approx(expected.mean, abs=1e-10)
