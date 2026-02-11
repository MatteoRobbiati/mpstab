import numpy as np
import pytest
from utils import DEFAULT_ATOL, expectation_with_qibo, ghz_circuit

from mpstab.evolutors.models import HybridSurrogate
from mpstab.models.ansatze import CircuitAnsatz, TranspiledAnsatz


@pytest.mark.parametrize("nqubits", [5, 8, 11])
def test_ghz(nqubits):

    obs_str = "Z" * nqubits

    ghz_circ = ghz_circuit(nqubits)
    ans = CircuitAnsatz(circuit=ghz_circ)
    hs = HybridSurrogate(ansatz=ans)

    exp = hs.expectation(observable=obs_str)
    qibo_expval = expectation_with_qibo(
        mpstab_ansatz=ans,
        observable_str=obs_str,
    )

    assert np.allclose(exp, qibo_expval, atol=DEFAULT_ATOL)
