"""
The computational engines a surrogate runs on.

A stabilizers engine backpropagates Pauli observables through Clifford gates
(:class:`StimEngine`, backed by stim); a tensor-network engine evolves the MPS
and contracts expectation values (:class:`QuimbEngine`, backed by quimb). The
abstract classes are the interface a custom engine implements.
"""

from mpstab.engines.stabilizers.abstract import StabilizersEngine
from mpstab.engines.stabilizers.stim import StimEngine
from mpstab.engines.tensor_networks.abstract import TensorNetworkEngine
from mpstab.engines.tensor_networks.quimb import QuimbEngine

__all__ = [
    "QuimbEngine",
    "StabilizersEngine",
    "StimEngine",
    "TensorNetworkEngine",
]
