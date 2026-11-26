"""
Nothing outside a tensor-network engine may look inside its states or
operators.

``OpaqueEngine`` runs quimb underneath but hands out its states and operators
in boxes with no attributes, so any code that reaches into them instead of
going through the :class:`~mpstab.engines.TensorNetworkEngine` interface fails
here. A new engine (tenpy, ...) then only has to implement that interface.
"""

import ast
import pathlib

import numpy as np
import pytest
from qibo import gates
from utils import small_entangled_circuit

import mpstab
from mpstab import HSMPO
from mpstab.engines import (
    NativeStabilizersEngine,
    QuimbEngine,
    StimEngine,
    TensorNetworkEngine,
)
from mpstab.evolutors.hqsmpo import HQSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient, KickedIsing
from mpstab.models.metrics import meyer_wallach_mps, stabilizer_renyi_entropy_mps
from mpstab.models.noise import DepolarizingError, NoiseModel
from mpstab.qibo_tools import StatevectorExactSimulator
from mpstab.qpu.execution import run_plan
from mpstab.qpu.measurement import ClassicalShadows, PauliSampling


class _Box:
    """An engine object with nothing to reach into."""

    __slots__ = ("inner",)

    def __init__(self, inner):
        self.inner = inner


class OpaqueEngine(TensorNetworkEngine):
    """
    QuimbEngine behind boxes: only the interface sees the quimb objects.

    It implements its own algorithms and no site view, as an engine written
    before the shared algorithms existed would: the base class must fall back
    to them.
    """

    def __init__(self):
        self.quimb = QuimbEngine(svd_cutoff=0.0)

    def build_circuit_mps(self, n, initial_state_circuit, max_bond_dimension=None):
        return _Box(
            self.quimb.build_circuit_mps(n, initial_state_circuit, max_bond_dimension)
        )

    def pauli_mpo(self, pauli_string):
        return _Box(self.quimb.pauli_mpo(pauli_string))

    def expval(self, state_circuit, operator):
        return self.quimb.expval(state_circuit.inner, operator.inner)

    def pauli_rot(self, state_circuit, generator, angle, max_bond_dimension):
        self.quimb.pauli_rot(state_circuit.inner, generator, angle, max_bond_dimension)

    def copy_state(self, state_circuit):
        return _Box(self.quimb.copy_state(state_circuit.inner))

    def squared_norm(self, state_circuit):
        return self.quimb.squared_norm(state_circuit.inner)

    def conjugate_operator(self, operator, generator, angle, max_bond_dimension):
        return _Box(
            self.quimb.conjugate_operator(
                operator.inner, generator, angle, max_bond_dimension
            )
        )

    def operator_site_arrays(self, operator):
        return self.quimb.operator_site_arrays(operator.inner)

    def pauli_sum_mpo(self, terms):
        return _Box(self.quimb.pauli_sum_mpo(terms))

    def minimize_energy(self, operator, initial_state, **options):
        result = self.quimb.minimize_energy(
            operator.inner, initial_state.inner, **options
        )
        return {**result, "state": _Box(result["state"])}

    def circuit_truncation_fidelity(self, nqubits, circuit, max_bond_dimension):
        return self.quimb.circuit_truncation_fidelity(
            nqubits, circuit, max_bond_dimension
        )


class SiteViewOpaqueEngine(OpaqueEngine):
    """
    The opaque engine with the representation hooks, so the shared algorithms
    of the base class run on it.
    """

    def state_sites(self, state_circuit):
        return self.quimb.state_sites(state_circuit.inner)

    def operator_from_sites(self, sites, like):
        return _Box(self.quimb.operator_from_sites(sites, like.inner))


@pytest.fixture(
    params=[OpaqueEngine, SiteViewOpaqueEngine], ids=["own-algorithms", "site-view"]
)
def opaque_engine(request):
    return request.param


def test_site_view_capabilities():
    assert QuimbEngine().has_site_view and QuimbEngine().has_operator_sites
    assert not OpaqueEngine().has_site_view
    assert not OpaqueEngine().has_operator_sites
    assert SiteViewOpaqueEngine().has_site_view
    assert SiteViewOpaqueEngine().has_operator_sites


def _pair(ansatz, engine_class, cls=HSMPO, **kwargs):
    """The same model on quimb and on the opaque engine."""
    reference = cls(ansatz, tn_engine=QuimbEngine(svd_cutoff=0.0), **kwargs)
    opaque = cls(ansatz, tn_engine=engine_class(), **kwargs)
    return reference, opaque


def test_hsmpo_expectation_and_fidelities(opaque_engine):
    reference, opaque = _pair(
        HardwareEfficient(nqubits=5, nlayers=3), opaque_engine, max_bond_dimension=2
    )
    expected = reference.expectation("ZZIXI", return_fidelity=True)
    assert opaque.expectation("ZZIXI", return_fidelity=True) == pytest.approx(expected)
    assert opaque.truncation_fidelity() == pytest.approx(
        reference.truncation_fidelity()
    )
    assert opaque.truncation_fidelity_pure_tn == pytest.approx(
        reference.truncation_fidelity_pure_tn
    )


def test_metrics(opaque_engine):
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    reference, opaque = _pair(ansatz, opaque_engine)
    assert meyer_wallach_mps(opaque) == pytest.approx(meyer_wallach_mps(reference))
    expected = stabilizer_renyi_entropy_mps(reference, alpha=2, n_samples=200, seed=0)
    assert stabilizer_renyi_entropy_mps(
        opaque, alpha=2, n_samples=200, seed=0
    ) == pytest.approx(expected)


def test_noisy_expectation(opaque_engine):
    reference, opaque = _pair(KickedIsing(nqubits=5), opaque_engine)
    noise = NoiseModel().add(DepolarizingError(0.02), gates.CNOT)
    observable = {"ZZIII": 0.7, "IXIXI": -0.4}
    expected = reference.noisy_expectation(observable, noise, n_samples=100, seed=1)
    result = opaque.noisy_expectation(observable, noise, n_samples=100, seed=1)
    assert result.mean == pytest.approx(expected.mean)
    np.testing.assert_allclose(result.fidelities, expected.fidelities)


def test_dmrg(opaque_engine):
    reference, opaque = _pair(HardwareEfficient(nqubits=4, nlayers=2), opaque_engine)
    options = dict(bond_dims=[4], max_sweeps=2, verbosity=0)
    hamiltonian = {"ZZII": 1.0, "IXXI": 0.5, "IIZZ": -0.3}
    expected = reference.minimize_expectation(hamiltonian, **options)["energy"]
    assert opaque.minimize_expectation(hamiltonian, **options)[
        "energy"
    ] == pytest.approx(expected, abs=1e-8)


def test_hqsmpo_simulation_and_tail_truncation(opaque_engine):
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    reference, opaque = _pair(ansatz, opaque_engine, cls=HQSMPO, max_bond_dimension=2)
    expected = reference.expectation_at_cut("ZZZZ", 2, method="simulation")
    result = opaque.expectation_at_cut("ZZZZ", 2, method="simulation")
    assert result.value == pytest.approx(expected.value)
    assert result.state_fidelity == pytest.approx(expected.state_fidelity)
    assert result.truncation_l2 == pytest.approx(expected.truncation_l2)

    expected_tail = reference.tail_truncation("ZZZZ", 1, 1, exact=True)
    tail = opaque.tail_truncation("ZZZZ", 1, 1, exact=True)
    assert tail.fidelity_estimate == pytest.approx(expected_tail.fidelity_estimate)
    assert tail.relative_frobenius_error == pytest.approx(
        expected_tail.relative_frobenius_error, abs=1e-7
    )


@pytest.mark.parametrize(
    "method", [ClassicalShadows(), PauliSampling()], ids=["shadows", "pauli"]
)
def test_device_routes(method, opaque_engine):
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    reference, opaque = _pair(
        ansatz, opaque_engine, cls=HQSMPO, head_synthesis="cnot_ladder"
    )
    plans = [
        model.plan(
            {"ZZZZ": 1.0, "XIXI": 0.5},
            2,
            method=method,
            n_shots=200,
            seed=3,
            out_folder=False,
        )
        for model in (reference, opaque)
    ]
    # Identical circuits, so the same counts serve both plans.
    frequencies = run_plan(plans[0], StatevectorExactSimulator())
    expected, result = (
        method.estimate(plan, frequencies, seed=7).value for plan in plans
    )
    assert result == pytest.approx(expected)


def test_only_the_stim_engine_imports_stim():
    """Every Clifford simulation goes through the stabilizers engine."""
    root = pathlib.Path(mpstab.__file__).parent
    allowed = root / "engines" / "stabilizers" / "stim.py"
    offenders = []
    for path in root.rglob("*.py"):
        if path == allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "stim" or name.startswith("stim.") for name in names):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


@pytest.mark.parametrize(
    "method",
    [
        PauliSampling(),
        ClassicalShadows(tail_handling="fold"),
    ],
    ids=["pauli", "shadows-fold"],
)
def test_residual_routes_on_the_native_stabilizers_engine(method):
    """A non-trivial resynthesis residual folds the same through any engine."""
    ansatz = HardwareEfficient(nqubits=4, nlayers=2)
    models = [
        HQSMPO(ansatz, stab_engine=engine, head_synthesis="extraction")
        for engine in (StimEngine(), NativeStabilizersEngine())
    ]
    cut = len(models[0].magic_gates) - 2
    residual = models[0].resynthesize_head(cut).residual_circuit
    assert not models[0].stab_engine.is_identity(residual)
    plans = [
        model.plan(
            {"ZZII": 1.0, "IXXI": 0.5},
            cut,
            method=method,
            n_shots=200,
            seed=3,
            out_folder=False,
        )
        for model in models
    ]
    frequencies = run_plan(plans[0], StatevectorExactSimulator())
    expected, result = (method.estimate(p, frequencies, seed=7).value for p in plans)
    assert result == pytest.approx(expected, abs=1e-12)
