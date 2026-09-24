"""
The surrogates: they decide what the stabilizer and tensor-network engines each
handle, and drive the simulation through a circuit.
"""

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.evolutors.hsynthsmpo import HSynthSMPO

__all__ = ["HSMPO", "HSynthSMPO"]
