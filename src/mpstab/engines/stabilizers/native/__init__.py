"""
Pure-Python stabilizers engine, with no external simulator.

:mod:`~mpstab.engines.stabilizers.native.pauli_string` holds the XZ-encoded
Pauli arithmetic and :mod:`~mpstab.engines.stabilizers.native.tableaus` the
per-gate conjugation rules the engine composes.
"""

from mpstab.engines.stabilizers.native.engine import NativeStabilizersEngine

__all__ = ["NativeStabilizersEngine"]
