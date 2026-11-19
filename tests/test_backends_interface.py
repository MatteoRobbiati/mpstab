import numpy as np
import pytest
from utils import expectation_with_qibo

from mpstab.engines import NativeStabilizersEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient


def test_stab_interfaces_vs_qibo(stab_engine, quimb_engine):
    nqubits = 5

    ans = HardwareEfficient(nqubits=nqubits, nlayers=3)
    hs = HSMPO(ansatz=ans)
    # Set the native stabilizers engine

    obs = "Z" * nqubits

    if isinstance(stab_engine, NativeStabilizersEngine):
        # The fast single-pass dressed-rotation extraction (used by
        # _precompute_original_mps whenever there are magic gates, as here)
        # requires StimEngine; NativeStabilizersEngine now raises.
        with pytest.raises(NotImplementedError):
            hs.set_engines(stab_engine=stab_engine, tn_engine=quimb_engine)
        return

    hs.set_engines(stab_engine=stab_engine, tn_engine=quimb_engine)

    exp_mpstab = hs.expectation(obs)

    exp_qibo = expectation_with_qibo(
        mpstab_ansatz=ans,
        observable_str=obs,
    )

    assert np.allclose(exp_mpstab, exp_qibo, atol=1e-6)


def test_native_tensor_network_engine_expectation_supported(native_tn_engine):
    """Test that NativeTensorNetworkEngine is now supported for expectation values."""
    nqubits = 4
    ans = HardwareEfficient(nqubits=nqubits, nlayers=2)
    hs = HSMPO(ansatz=ans)

    # Should not raise - NativeTensorNetworkEngine is now supported for expectation()
    hs.set_engines(tn_engine=native_tn_engine)

    # Expectation calculation should work
    result = hs.expectation("ZZZZ")
    assert isinstance(result, (float, np.floating))


def test_native_tensor_network_engine_dmrg_not_supported(native_tn_engine):
    """Test that DMRG optimization with NativeTensorNetworkEngine raises NotImplementedError."""
    nqubits = 4
    ans = HardwareEfficient(nqubits=nqubits, nlayers=2)
    hs = HSMPO(ansatz=ans)

    hs.set_engines(tn_engine=native_tn_engine)

    # DMRG optimization should raise NotImplementedError
    with pytest.raises(
        NotImplementedError,
        match="DMRG optimization requires QuimbEngine",
    ):
        hs.minimize_expectation(
            observables={"ZZZZ": 1.0},
            method="dmrg",
            bond_dims=[8],
            max_sweeps=2,
            verbosity=0,
        )
