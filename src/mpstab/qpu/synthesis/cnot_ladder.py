"""
Head synthesis by the textbook CNOT-ladder decomposition: no extra dependency,
``2*(w-1)`` CNOTs per weight-``w`` rotation, and an identity residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from mpstab.qpu.synthesis.abstract import HeadSynthesis


@dataclass(frozen=True)
class CnotLadderSynthesis(HeadSynthesis):
    """
    Decompose every dressed rotation into an exact CNOT-ladder circuit.

    Uses the textbook identity ``exp(-i theta P / 2) = U^dag L^dag RZ(theta) L
    U``, where ``U`` rotates every non-identity site of ``P`` into the ``Z``
    frame and ``L`` is a CNOT ladder chaining the parity of the support onto its
    last qubit. Every gate is placed exactly, so nothing is left over and the
    residual is the identity -- at the cost of ``2*(w-1)`` CNOTs per weight-``w``
    rotation and no cancellation across rotations.
    """

    name: ClassVar[str] = "cnot_ladder"

    def _synthesize(self, paulis, angles, nqubits):
        head: list[tuple] = []
        for pauli, angle in zip(paulis, angles):
            support = [q for q, label in enumerate(pauli) if label != "I"]
            if not support:
                continue  # exp(-i theta/2 I) is a global phase, no gate needed.

            for qubit in support:
                if pauli[qubit] == "Y":
                    head.append(("Sd", [qubit]))
                if pauli[qubit] in ("X", "Y"):
                    head.append(("H", [qubit]))

            ladder = list(zip(support[:-1], support[1:]))
            head.extend(("CNOT", [q0, q1]) for q0, q1 in ladder)
            head.append(("RZ", [support[-1]], float(angle)))
            head.extend(("CNOT", [q0, q1]) for q0, q1 in reversed(ladder))

            for qubit in support:
                if pauli[qubit] in ("X", "Y"):
                    head.append(("H", [qubit]))
                if pauli[qubit] == "Y":
                    head.append(("S", [qubit]))

        return head, []
