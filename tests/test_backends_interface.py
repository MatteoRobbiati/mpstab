import numpy as np
import pytest
from qibo import set_backend
from utils import obs_string_to_qibo_hamiltonian

from mpstab.backends.stabilizers.native import NativeStabilizersEngine
from mpstab.backends.stabilizers.stim import StimEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient

set_backend("numpy")


@pytest.mark.parametrize("stab_engine", ["native", "stim"])
def test_stab_interfaces_vs_qibo(stab_engine):
    nqubits = 5

    ans = HardwareEfficient(nqubits=nqubits, nlayers=3)
    hs = HSMPO(ansatz=ans)
    # Set the native stabilizers engine

    obs = "Z" * nqubits
    qibo_obs = obs_string_to_qibo_hamiltonian(obs)

    if stab_engine == "native":
        stab_engine = NativeStabilizersEngine()
    elif stab_engine == "stim":
        stab_engine = StimEngine()

    hs.set_engines(stab_engine=stab_engine)
    exp_mpstab = hs.expectation(obs)
    exp_qibo = qibo_obs.expectation_from_state(ans.circuit().state())

    np.allclose(exp_mpstab, exp_qibo, atol=1e-6)
