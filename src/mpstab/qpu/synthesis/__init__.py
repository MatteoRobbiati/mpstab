"""
Step 1 of the QPU pipeline: head synthesis.

A head of dressed Pauli rotations is turned into a hardware-native gate list
plus a pure-Clifford residual applied last::

    target = U(residual) . U(head)
    <target|O|target> = <head 0| residual^dag . O . residual |head 0>

The residual is therefore reabsorbed into the observable with a Clifford
simulator (:meth:`~mpstab.engines.StimEngine.fold_pauli_through_tableau`), never
with dense matrices.

Two synthesisers produce that pair, one module each:

- :mod:`~mpstab.qpu.synthesis.rustiq`: rustiq's Pauli-network synthesis
  (optional dependency, imported lazily).
- :mod:`~mpstab.qpu.synthesis.cnot_ladder`: the textbook decomposition, with an
  identity residual.

All return ``(head, residual_tableau, residual_gates)``, with ``head`` in the
gate-list format of :mod:`~mpstab.qpu.synthesis.utils`.
"""

from mpstab.qpu.synthesis.cnot_ladder import build_cnot_ladder_head_and_residual
from mpstab.qpu.synthesis.rustiq import build_head_and_residual, head_counts_only
from mpstab.qpu.synthesis.utils import (
    CLIFFORD_GATES,
    CLIFFORD_TABLEAUS,
    ROTATION_GATES,
    count_two_qubit_gates,
    head_to_qibo_circuit,
)

__all__ = [
    "CLIFFORD_GATES",
    "CLIFFORD_TABLEAUS",
    "ROTATION_GATES",
    "build_cnot_ladder_head_and_residual",
    "build_head_and_residual",
    "count_two_qubit_gates",
    "head_counts_only",
    "head_to_qibo_circuit",
]
