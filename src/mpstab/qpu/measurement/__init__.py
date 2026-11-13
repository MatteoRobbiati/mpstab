"""
Step 2 of the QPU pipeline: measuring the tail-folded observable.

Each method lives in one module, holding both halves of it: building the
:class:`~mpstab.qpu.measurement.abstract.MeasurementPlan` (circuits and shots,
fixed before anything runs) and estimating an
:class:`~mpstab.qpu.measurement.abstract.ExpectationResult` from the measured
frequencies.

- :mod:`~mpstab.qpu.measurement.pauli`: sampled Pauli strings, grouped into
  qubit-wise-commuting settings.
- :mod:`~mpstab.qpu.measurement.shadows`: classical shadows, contracted with the
  observable MPO.
- :mod:`~mpstab.qpu.measurement.tnice`: the same data as ``"shadows"``,
  estimated with the MPS-optimised dual of TN-ICE.

:mod:`~mpstab.qpu.measurement.pauli_expansion` turns the observable MPO into
Pauli strings and variances, for the methods above.
"""

from mpstab.qpu.measurement.abstract import ExpectationResult, MeasurementPlan
from mpstab.qpu.measurement.pauli import (
    PauliMeasurementPlan,
    QWCGroup,
    allocate_shots_by_variance,
    build_measurement_plan,
    build_pauli_plan,
    estimate_pauli,
    group_qwc,
)
from mpstab.qpu.measurement.shadows import build_shadow_plan, estimate_shadows
from mpstab.qpu.measurement.tnice import estimate_tnice


def estimate(plan, frequencies, **kwargs) -> ExpectationResult:
    """
    Dispatch to :func:`~mpstab.qpu.measurement.pauli.estimate_pauli`,
    :func:`~mpstab.qpu.measurement.shadows.estimate_shadows` or
    :func:`~mpstab.qpu.measurement.tnice.estimate_tnice` by ``plan.method``.
    ``kwargs`` are forwarded only to ``estimate_tnice``, the only one of the
    three that takes any.
    """
    if plan.method == "pauli":
        return estimate_pauli(plan, frequencies)
    if plan.method == "shadows":
        return estimate_shadows(plan, frequencies)
    if plan.method == "tnice":
        return estimate_tnice(plan, frequencies, **kwargs)
    raise ValueError(
        f"Unknown plan method {plan.method!r}, expected 'pauli', 'shadows' or 'tnice'."
    )


__all__ = [
    "ExpectationResult",
    "MeasurementPlan",
    "PauliMeasurementPlan",
    "QWCGroup",
    "allocate_shots_by_variance",
    "build_measurement_plan",
    "build_pauli_plan",
    "build_shadow_plan",
    "estimate",
    "estimate_pauli",
    "estimate_shadows",
    "estimate_tnice",
    "group_qwc",
]
