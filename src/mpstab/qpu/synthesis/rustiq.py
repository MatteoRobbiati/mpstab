"""
Head synthesis with rustiq's Pauli-network synthesis, which cancels gates across
rotations.

``rustiq`` is optional and not on PyPI: this module imports without it, and
synthesising raises an ``ImportError`` with the install hint
``pip install "mpstab[rustiq]"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from mpstab.qpu.synthesis.abstract import HeadSynthesis
from mpstab.qpu.synthesis.utils import place_rotations

try:
    from rustiq import Metric, pauli_network_synthesis
except ImportError:  # optional dependency
    Metric = pauli_network_synthesis = None


def _require_rustiq():
    """Raise an ``ImportError`` with the install hint if rustiq is missing."""
    if pauli_network_synthesis is None:
        raise ImportError(
            "rustiq is required for Pauli-network resynthesis. Install it with: "
            'pip install "mpstab[rustiq]"'
        )


@dataclass(frozen=True)
class RustiqSynthesis(HeadSynthesis):
    """
    rustiq's Pauli-network synthesis, which cancels gates across rotations.

    ``rustiq.pauli_network_synthesis`` called with ``fix_clifford=False``
    returns a skeleton that is a prefix of the one it returns with
    ``fix_clifford=True``. The prefix becomes the head; what the longer version
    appends is the residual. The residual gate list is non-deterministic across
    runs (``fix_clifford=True`` draws randomly) even though the Clifford it
    implements is unique, so compare Cliffords (e.g. with
    :meth:`~mpstab.engines.StabilizersEngine.clifford_images`) rather than gate
    lists.

    Args:
        metric: what rustiq minimises, ``"count"`` or ``"depth"``.
        preserve_order: keep the (non-commuting) rotation order.
    """

    metric: str = "count"
    preserve_order: bool = True

    name: ClassVar[str] = "rustiq"

    def synthesize(self, paulis, angles, nqubits=None):
        _require_rustiq()  # fail with the install hint even on an empty head
        return super().synthesize(paulis, angles, nqubits)

    def _synthesize(self, paulis, angles, nqubits):
        _require_rustiq()
        metric = Metric(self.metric)
        skeleton = pauli_network_synthesis(
            paulis, metric, self.preserve_order, fix_clifford=False
        )
        fixed = pauli_network_synthesis(
            paulis, metric, self.preserve_order, fix_clifford=True
        )
        if fixed[: len(skeleton)] != skeleton:
            raise RuntimeError(
                "rustiq returned a fix_clifford=True skeleton that does not extend "
                "the fix_clifford=False one; cannot split off a Clifford residual."
            )
        head = place_rotations(skeleton, paulis, angles, nqubits)
        return head, fixed[len(skeleton) :]
