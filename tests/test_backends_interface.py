import numpy as np
import pytest
from qibo import set_backend
from utils import expectation_with_qibo

from mpstab.backends.stabilizers.native import NativeStabilizersEngine
from mpstab.backends.stabilizers.stim import StimEngine
from mpstab.backends.tensor_networks.native import NativeTensorNetworkEngine
from mpstab.backends.tensor_networks.quimb import QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient

set_backend("numpy")


@pytest.mark.parametrize("stab_engine", [NativeStabilizersEngine, StimEngine])
@pytest.mark.parametrize("tn_engine", [NativeTensorNetworkEngine, QuimbEngine])
def test_stab_interfaces_vs_qibo(stab_engine, tn_engine):
    nqubits = 5

    ans = HardwareEfficient(nqubits=nqubits, nlayers=3)
    hs = HSMPO(ansatz=ans)
    # Set the native stabilizers engine

    obs = "Z" * nqubits

    hs.set_engines(
        stab_engine=stab_engine(),
        tn_engine=tn_engine(),
    )
    exp_mpstab = hs.expectation(obs)

    exp_qibo = expectation_with_qibo(
        mpstab_ansatz=ans,
        observable_str=obs,
    )

    np.allclose(exp_mpstab, exp_qibo, atol=1e-6)
