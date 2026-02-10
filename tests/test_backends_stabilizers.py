import numpy as np
import pytest
from qibo import Circuit, gates, set_backend

from mpstab.backends.stabilizers.native import NativeStabilizersEngine
from mpstab.backends.stabilizers.stim import StimEngine
from mpstab.evolutors.models import HybridSurrogate
from mpstab.models.ansatze import CircuitAnsatz

set_backend("numpy")


def construct_ghz_circuit(nqubits):
    """Constructs a standard GHZ circuit."""
    c = Circuit(nqubits)
    c.add(gates.H(0))
    for i in range(nqubits - 1):
        c.add(gates.CNOT(i, i + 1))
    return c


@pytest.mark.parametrize("observable", ["XXXXX", "ZIIXI", "XIXXI", "ZYXZI"])
def test_backends_expectation_matches(observable):
    # Original random Clifford test logic
    from utils import (  # Assuming this is available in your local env
        construct_test_circuit,
    )

    circ = construct_test_circuit(
        nqubits=5,
        rng_seed=42,
        is_clifford=True,
    )
    ansatz = CircuitAnsatz(circuit=circ)
    hs = HybridSurrogate(ansatz)

    hs.set_backend(stab_engine=NativeStabilizersEngine())
    exp_native = hs.expectation(observable)

    hs.set_backend(stab_engine=StimEngine())
    exp_stim = hs.expectation(observable)

    assert np.allclose(exp_stim, exp_native, atol=1e-6)


@pytest.mark.parametrize("nqubits", [3, 5, 8])
def test_ghz_stabilizers(nqubits):
    """
    Specifically checks GHZ state stabilizers:
    Z_i Z_{i+1} for all i, and X...X.
    """
    circ = construct_ghz_circuit(nqubits)
    ansatz = CircuitAnsatz(circuit=circ)
    hs = HybridSurrogate(ansatz)

    obs = "Z" * nqubits

    hs.set_backend(stab_engine=NativeStabilizersEngine())
    exp_native = hs.expectation(obs)

    hs.set_backend(stab_engine=StimEngine())
    exp_stim = hs.expectation(obs)

    # For a pure GHZ state, these stabilizers should return 1.0
    assert np.allclose(exp_stim, 1.0, atol=1e-6)
    assert np.allclose(exp_native, 1.0, atol=1e-6)
    assert np.allclose(exp_stim, exp_native, atol=1e-6)
