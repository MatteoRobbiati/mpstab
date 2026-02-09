import numpy as np
import pytest
from qibo import set_backend
from utils import construct_test_circuit

from mpstab.backends.stabilizers.native import NativeStabilizersEngine
from mpstab.backends.stabilizers.stim import StimEngine
from mpstab.evolutors.models import HybridSurrogate
from mpstab.models.ansatze import CircuitAnsatz

set_backend("numpy")


@pytest.mark.parametrize("observable", ["ZIIXI", "XIXXI", "ZYXZI"])
def test_backends_expectation_matches(observable):

    circ = construct_test_circuit(
        nqubits=5,
        rng_seed=42,
        is_clifford=True,
    )

    ansatz = CircuitAnsatz(circuit=circ)

    hs = HybridSurrogate(ansatz)

    hs.set_backend(stab_engine=StimEngine())
    exp_stim = hs.expectation(observable)

    hs.set_backend(stab_engine=NativeStabilizersEngine())
    exp_native = hs.expectation(observable)

    assert np.allclose(exp_stim, exp_native, atol=1e-6)
