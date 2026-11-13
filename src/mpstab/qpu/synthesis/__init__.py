"""
Step 1 of the QPU pipeline: head synthesis.

A head of dressed Pauli rotations is turned into a hardware-native gate list
plus a pure-Clifford residual applied last::

    target = U(residual) . U(head)
    <target|O|target> = <head 0| residual^dag . O . residual |head 0>

The residual is therefore reabsorbed into the observable with a Clifford
simulator (:meth:`~mpstab.engines.StimEngine.fold_pauli_through_tableau`), never
with dense matrices.

A synthesiser is a :class:`~mpstab.qpu.synthesis.abstract.HeadSynthesis`
holding its own options. Three are available, one module each:

- :class:`~mpstab.qpu.synthesis.rustiq.RustiqSynthesis`: rustiq's Pauli-network
  synthesis (optional dependency, imported lazily).
- :class:`~mpstab.qpu.synthesis.extraction.ExtractionSynthesis`: Clifford
  extraction, dependency-free and connectivity-aware.
- :class:`~mpstab.qpu.synthesis.cnot_ladder.CnotLadderSynthesis`: the textbook
  decomposition, with an identity residual.

Each module also exposes its synthesiser as a plain function returning
``(head, residual_tableau, residual_gates)``, with ``head`` in the gate-list
format of :mod:`~mpstab.qpu.synthesis.utils`.
"""

from mpstab.qpu.synthesis.abstract import HeadSynthesis
from mpstab.qpu.synthesis.cnot_ladder import (
    CnotLadderSynthesis,
    build_cnot_ladder_head_and_residual,
)
from mpstab.qpu.synthesis.extraction import (
    EXTRACTION_STRATEGIES,
    ExtractionSynthesis,
    build_extraction_head_and_residual,
    connectivity_distances,
)
from mpstab.qpu.synthesis.rustiq import (
    RustiqSynthesis,
    build_head_and_residual,
    head_counts_only,
)
from mpstab.qpu.synthesis.utils import (
    CLIFFORD_GATES,
    CLIFFORD_TABLEAUS,
    ROTATION_GATES,
    count_two_qubit_gates,
    head_to_qibo_circuit,
)

#: The synthesisers by name, for the string shorthand of :func:`resolve_synthesis`.
SYNTHESES = {
    cls.name: cls for cls in (RustiqSynthesis, ExtractionSynthesis, CnotLadderSynthesis)
}


def resolve_synthesis(synthesis) -> HeadSynthesis:
    """
    A :class:`~mpstab.qpu.synthesis.abstract.HeadSynthesis` from either an
    instance, returned as is, or its name, giving the default options.

    Raises:
        ValueError: on an unknown name.
    """
    if isinstance(synthesis, HeadSynthesis):
        return synthesis
    if synthesis not in SYNTHESES:
        raise ValueError(
            f"Unknown head synthesis {synthesis!r}, expected a HeadSynthesis or "
            f"one of {tuple(SYNTHESES)}."
        )
    return SYNTHESES[synthesis]()


__all__ = [
    "CLIFFORD_GATES",
    "CLIFFORD_TABLEAUS",
    "CnotLadderSynthesis",
    "EXTRACTION_STRATEGIES",
    "ExtractionSynthesis",
    "HeadSynthesis",
    "ROTATION_GATES",
    "RustiqSynthesis",
    "SYNTHESES",
    "build_cnot_ladder_head_and_residual",
    "build_extraction_head_and_residual",
    "build_head_and_residual",
    "connectivity_distances",
    "count_two_qubit_gates",
    "head_counts_only",
    "head_to_qibo_circuit",
    "resolve_synthesis",
]
