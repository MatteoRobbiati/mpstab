"""The interface every head synthesiser implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar


class HeadSynthesis(ABC):
    """
    How a head of dressed Pauli rotations becomes a runnable gate list plus a
    pure-Clifford residual, applied last.

    Subclasses are frozen dataclasses holding their own options, so an instance
    is hashable and :meth:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO.resynthesize_head`
    can cache resyntheses on it.
    """

    #: Short name, reported as
    #: :attr:`~mpstab.evolutors.hsynthsmpo.ResynthesizedHead.method`.
    name: ClassVar[str]

    @abstractmethod
    def synthesize(self, paulis: list[str], angles: list[float]) -> tuple:
        """
        Synthesise the rotations ``exp(-i angle P / 2)``, in order.

        Args:
            paulis: dressed Pauli axis strings (qubit-0-leftmost), in circuit
                order.
            angles: matching rotation angles.

        Returns:
            ``(head, residual_tableau, residual_gates)``: the head as a gate
            list in the format of :mod:`~mpstab.qpu.synthesis.utils`, and the
            Clifford residual both as a ``stim.Tableau`` and as a gate list.
        """
