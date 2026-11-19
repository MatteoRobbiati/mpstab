"""The interface every head synthesiser implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import stim

from mpstab.qpu.synthesis.utils import CLIFFORD_TABLEAUS


class HeadSynthesis(ABC):
    """
    How a head of dressed Pauli rotations becomes a runnable gate list plus a
    pure-Clifford residual, applied last.

    Subclasses are frozen dataclasses holding their own options, so an instance
    is hashable and :meth:`~mpstab.evolutors.hqsmpo.HQSMPO.resynthesize_head`
    can cache resyntheses on it. They implement :meth:`_synthesize`.
    """

    #: Short name, reported as
    #: :attr:`~mpstab.evolutors.hqsmpo.ResynthesizedHead.method`.
    name: ClassVar[str]

    def synthesize(
        self, paulis: list[str], angles: list[float], nqubits: int | None = None
    ) -> tuple:
        """
        Synthesise the rotations ``exp(-i angle P / 2)``, in order.

        Args:
            paulis: dressed Pauli axis strings (qubit-0-leftmost), in circuit
                order.
            angles: matching rotation angles.
            nqubits: circuit width; defaults to the width of the strings, and
                sizes the residual of an empty head.

        Returns:
            ``(head, residual_tableau, residual_gates)``: the head as a gate
            list in the format of :mod:`~mpstab.qpu.synthesis.utils`, and the
            Clifford residual both as a ``stim.Tableau`` and as a gate list.
        """
        paulis, angles = list(paulis), list(angles)
        if nqubits is None:
            nqubits = len(paulis[0]) if paulis else 0
        head, residual = (
            self._synthesize(paulis, angles, nqubits) if paulis else ([], [])
        )
        residual_tableau = stim.Tableau(nqubits)
        for name, qubits in residual:
            residual_tableau.append(CLIFFORD_TABLEAUS[name], qubits)
        return head, residual_tableau, list(residual)

    @abstractmethod
    def _synthesize(self, paulis: list[str], angles: list[float], nqubits: int):
        """``(head, residual_gates)`` for a non-empty head."""
