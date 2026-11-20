"""
The computational engines a surrogate runs on.

A stabilizers engine backpropagates Pauli observables through Clifford gates
(:class:`StimEngine`, backed by stim); a tensor-network engine evolves the MPS
and contracts expectation values (:class:`QuimbEngine`, backed by quimb). The
abstract classes are the interface a custom engine implements.

:class:`NativeStabilizersEngine` and :class:`NativeTensorNetworkEngine` are
pure-Python implementations of that interface, kept as readable examples of
how each engine works. They are slower than stim and quimb and cover less: no
DMRG and no operator-side conjugation, so no head/tail split in
:class:`~mpstab.evolutors.hqsmpo.HQSMPO`.
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
