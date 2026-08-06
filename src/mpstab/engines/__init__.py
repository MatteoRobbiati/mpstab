"""
The computational engines a surrogate runs on.

A stabilizers engine backpropagates Pauli observables through Clifford gates; a
tensor-network engine evolves the MPS and contracts expectation values. Each
comes in a pure-Python ``Native`` flavour and one backed by an external simulator
(stim, quimb).
"""

from mpstab.engines.stabilizers.abstract import StabilizersEngine
from mpstab.engines.stabilizers.native import NativeStabilizersEngine
from mpstab.engines.stabilizers.stim import StimEngine
from mpstab.engines.tensor_networks.abstract import TensorNetworkEngine
from mpstab.engines.tensor_networks.native import NativeTensorNetworkEngine
from mpstab.engines.tensor_networks.quimb import QuimbEngine

__all__ = [
    "NativeStabilizersEngine",
    "NativeTensorNetworkEngine",
    "QuimbEngine",
    "StabilizersEngine",
    "StimEngine",
    "TensorNetworkEngine",
]
