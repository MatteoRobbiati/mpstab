"""
Pure-Python tensor-network engine, with no external tensor library.

:mod:`~mpstab.engines.tensor_networks.native.tensor_network` is the general
contractable-graph layer,
:mod:`~mpstab.engines.tensor_networks.native.circuit_mps` the MPS circuit
simulator built on it, and
:mod:`~mpstab.engines.tensor_networks.native.operators` the gates and observables
it applies.
"""

from mpstab.engines.tensor_networks.native.engine import NativeTensorNetworkEngine
from mpstab.engines.tensor_networks.native.tensor_network import TensorNetwork

__all__ = ["NativeTensorNetworkEngine", "TensorNetwork"]
