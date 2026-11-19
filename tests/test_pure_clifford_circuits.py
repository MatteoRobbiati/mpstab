import numpy as np
import pytest
from utils import (
    DEFAULT_ATOL,
    construct_test_circuit,
    expectation_with_qibo,
    ghz_circuit,
)

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import CircuitAnsatz


@pytest.mark.parametrize("nqubits", [3, 4, 5, 8, 10])
def test_ghz(nqubits, stab_engine):
    obs_str = "Z" * nqubits
    ans = CircuitAnsatz(qibo_circuit=ghz_circuit(nqubits))
    exp = HSMPO(ansatz=ans, stab_engine=stab_engine).expectation(observable=obs_str)
    qibo_expval = expectation_with_qibo(mpstab_ansatz=ans, observable_str=obs_str)
    assert np.allclose(exp, qibo_expval, atol=DEFAULT_ATOL)


@pytest.mark.parametrize("observable", ["XXXXX", "ZIIXI", "XIXXI", "ZYXZI"])
def test_clifford_expectation_matches_qibo(observable, stab_engine):
    circ = construct_test_circuit(nqubits=5, rng_seed=42, is_clifford=True)
    ansatz = CircuitAnsatz(qibo_circuit=circ)
    exp_mpstab = HSMPO(ansatz, stab_engine=stab_engine).expectation(observable)
    exp_qibo = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)
    assert np.allclose(exp_qibo, exp_mpstab, atol=1e-6)
