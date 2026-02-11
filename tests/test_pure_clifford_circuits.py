import numpy as np
import pytest
from qibo import set_backend
from utils import (
    DEFAULT_ATOL,
    construct_test_circuit,
    expectation_with_qibo,
    ghz_circuit,
)

from mpstab.backends.stabilizers.native import NativeStabilizersEngine
from mpstab.backends.stabilizers.stim import StimEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import CircuitAnsatz

set_backend("numpy")


@pytest.mark.parametrize("stab_engine", [NativeStabilizersEngine, StimEngine])
@pytest.mark.parametrize("nqubits", [3, 4, 5, 8, 10])
def test_ghz(stab_engine, nqubits):

    obs_str = "Z" * nqubits

    ghz_circ = ghz_circuit(nqubits)
    ans = CircuitAnsatz(qibo_circuit=ghz_circ)
    hs = HSMPO(ansatz=ans)

    exp = hs.expectation(observable=obs_str)
    hs.set_engines(stab_engine=stab_engine())
    qibo_expval = expectation_with_qibo(
        mpstab_ansatz=ans,
        observable_str=obs_str,
    )

    assert np.allclose(exp, qibo_expval, atol=DEFAULT_ATOL)


@pytest.mark.parametrize("stab_engine", [NativeStabilizersEngine, StimEngine])
@pytest.mark.parametrize("observable", ["XXXXX", "ZIIXI", "XIXXI", "ZYXZI"])
def test_backends_expectation_matches(stab_engine, observable):
    circ = construct_test_circuit(
        nqubits=5,
        rng_seed=42,
        is_clifford=True,
    )
    ansatz = CircuitAnsatz(qibo_circuit=circ)
    hs = HSMPO(ansatz)

    hs.set_engines(stab_engine=stab_engine())
    exp_mpstab = hs.expectation(observable)

    exp_qibo = expectation_with_qibo(
        mpstab_ansatz=ansatz,
        observable_str=observable,
    )

    assert np.allclose(exp_qibo, exp_mpstab, atol=1e-6)
