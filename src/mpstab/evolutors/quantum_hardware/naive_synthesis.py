"""Naive CNOT-ladder Pauli-rotation synthesis, the dependency-free fallback.

:mod:`rustiq` is an optional dependency (see that module's docstring) and is
not on PyPI, so :meth:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO.resynthesize_head`
must still produce a runnable head circuit when it is unavailable. This module
gives the textbook decomposition of a Pauli rotation into single-qubit basis
changes plus a CNOT ladder:

.. math::

    e^{-i\\theta P / 2} = U^\\dagger L^\\dagger \\, R_Z(\\theta) \\, L U,

where ``U`` rotates every non-identity site of ``P`` into the ``Z`` frame
(the same single-qubit Cliffords that rotate a Pauli basis into the Z frame for
measurement, since ``U P U^\\dagger = Z`` and ``U^\\dagger Z U = P`` are the
same statement)
and ``L`` is a CNOT ladder chaining the parity of every support qubit onto the
last one, so that ``L (Z_{q_0} ... Z_{q_{k-1}}) L^\\dagger = Z_{q_{k-1}}``. This
is exact for every rotation, so unlike the rustiq path there is no Clifford
residual to reabsorb: the tail tableau is the identity.

Cost is ``2*(w-1)`` CNOTs per weight-``w`` rotation (the forward ladder plus
its exact reverse), against rustiq's generally sub-linear gate count after
cross-rotation cancellation -- this is the resynthesis saving
:meth:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO.resynthesize_head` gives up when
falling back to this module.
"""

from __future__ import annotations

import stim


def build_naive_head_and_residual(paulis: list[str], angles: list[float]):
    """
    Decompose every dressed rotation into an exact CNOT-ladder circuit.

    Args:
        paulis: dressed Pauli axis strings (qubit-0-leftmost), circuit order.
        angles: matching rotation angles.

    Returns:
        ``(head, tail_tableau, tail_gates)``, the same shape as
        :func:`mpstab.evolutors.quantum_hardware.rustiq_synthesis.build_head_and_residual`:
        ``head`` is a gate list (rotations ``("RZ", [q], theta)`` and
        Cliffords ``("H"/"S"/"Sd"/"CNOT", [qubits])``), ``tail_tableau`` is the
        identity :class:`stim.Tableau` (nothing is left over: every gate above
        is placed exactly), and ``tail_gates`` is empty.
    """
    if not paulis:
        return [], stim.Tableau(0), []

    n = len(paulis[0])
    head: list[tuple] = []
    for pauli, angle in zip(paulis, angles):
        support = [q for q, label in enumerate(pauli) if label != "I"]
        if not support:
            continue  # exp(-i theta/2 I) is a global phase, no gate needed.

        for q in support:
            if pauli[q] == "Y":
                head.append(("Sd", [q]))
            if pauli[q] in ("X", "Y"):
                head.append(("H", [q]))

        ladder = list(zip(support[:-1], support[1:]))
        for q0, q1 in ladder:
            head.append(("CNOT", [q0, q1]))

        head.append(("RZ", [support[-1]], float(angle)))

        for q0, q1 in reversed(ladder):
            head.append(("CNOT", [q0, q1]))

        for q in support:
            if pauli[q] in ("X", "Y"):
                head.append(("H", [q]))
            if pauli[q] == "Y":
                head.append(("S", [q]))

    return head, stim.Tableau(n), []
