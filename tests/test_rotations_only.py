"""
`HSMPO.rotations_only` / `HSynthSMPO.rotations_only`: a lazy constructor that
skips the eager MPS precompute in `__post_init__`, for when only the dressed
rotations or a resynthesized circuit are needed.
"""
import time

import numpy as np
import pytest
from qibo import set_backend

from mpstab.engines import NativeTensorNetworkEngine, QuimbEngine, StimEngine
from mpstab.engines.stabilizers.native import NativeStabilizersEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient

set_backend("numpy")


def _small_entangled_circuit(n=4):
    from qibo import gates, Circuit

    circ = Circuit(n)
    for q in range(n):
        circ.add(gates.H(q))
    for q in range(n - 1):
        circ.add(gates.CZ(q, q + 1))
    for q in range(n):
        circ.add(gates.RY(q, theta=float(np.random.uniform(0.1, 1.2))))
    return circ


def test_rotations_only_sets_expected_attributes():
    ansatz = HardwareEfficient(nqubits=4, nlayers=2)

    full = HSMPO(ansatz=ansatz)
    lazy = HSMPO.rotations_only(ansatz)

    assert lazy.ansatz is ansatz
    assert lazy.max_bond_dimension is None
    assert isinstance(lazy.stab_engine, StimEngine)
    assert isinstance(lazy.tn_engine, QuimbEngine)
    assert lazy.nqubits == full.nqubits == 4

    # Same partitioning (deterministic, replacement_probability=0.0).
    assert len(lazy.magic_gates) == len(full.magic_gates)
    assert [g.name for _, g in lazy.magic_gates] == [g.name for _, g in full.magic_gates]
    assert len(lazy.clifford_circuit.queue) == len(full.clifford_circuit.queue)


def test_rotations_only_skips_mps_precompute():
    ansatz = HardwareEfficient(nqubits=4, nlayers=2)
    lazy = HSMPO.rotations_only(ansatz)

    assert not hasattr(lazy, "original_circuit_mps")
    assert not hasattr(lazy, "mps")
    assert not hasattr(lazy, "_mps_engine_type")

    with pytest.raises(AttributeError):
        lazy.expectation("Z" * ansatz.nqubits)


def test_rotations_only_accepts_plain_qibo_circuit():
    circuit = _small_entangled_circuit(4)
    lazy = HSynthSMPO.rotations_only(circuit)

    assert isinstance(lazy.ansatz, CircuitAnsatz)
    assert lazy.nqubits == 4
    assert len(lazy._dressed_rotations()) == len(lazy.magic_gates)


def test_rotations_only_dressed_rotations_match_full_construction():
    ansatz = HardwareEfficient(nqubits=5, nlayers=3)

    full = HSynthSMPO(ansatz)
    lazy = HSynthSMPO.rotations_only(ansatz)

    assert lazy._dressed_rotations() == full._dressed_rotations()


def test_rotations_only_validates_engine_types():
    ansatz = HardwareEfficient(nqubits=3, nlayers=1)

    with pytest.raises(ValueError):
        HSMPO.rotations_only(ansatz, stab_engine="not-an-engine")
    with pytest.raises(ValueError):
        HSMPO.rotations_only(ansatz, tn_engine="not-an-engine")

    # Non-default but valid engines are accepted and stored as-is.
    lazy = HSMPO.rotations_only(ansatz, stab_engine=NativeStabilizersEngine())
    assert isinstance(lazy.stab_engine, NativeStabilizersEngine)


def test_set_engines_upgrades_lazy_instance():
    # Calling set_engines() on a lazy instance computes original_circuit_mps,
    # "upgrading" it to a fully built one (same code path __post_init__ uses).
    ansatz = HardwareEfficient(nqubits=4, nlayers=2)
    lazy = HSMPO.rotations_only(ansatz)

    lazy.set_engines()

    assert hasattr(lazy, "original_circuit_mps")
    obs = "Z" * ansatz.nqubits
    full = HSMPO(ansatz)
    assert np.allclose(lazy.expectation(obs), full.expectation(obs), atol=1e-6)


def test_rotations_only_is_much_faster_than_full_construction():
    # The point of rotations_only: skip the (in general exponentially large)
    # exact MPS precompute entirely.
    ansatz = HardwareEfficient(nqubits=14, nlayers=3)

    t0 = time.perf_counter()
    HSMPO.rotations_only(ansatz)
    t_lazy = time.perf_counter() - t0

    t0 = time.perf_counter()
    HSMPO(ansatz)
    t_full = time.perf_counter() - t0

    assert t_lazy < t_full
    print(
        f"\n[rotations_only] HardwareEfficient n=14: "
        f"lazy={t_lazy:.4f}s full={t_full:.4f}s speedup={t_full / t_lazy:.1f}x"
    )
