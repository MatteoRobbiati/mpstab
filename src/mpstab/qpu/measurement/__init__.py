"""
Step 2 of the QPU pipeline: measuring the tail-folded observable.

A method is a :class:`~mpstab.qpu.measurement.abstract.MeasurementMethod`
holding its own options, with both halves of the job: building the
:class:`~mpstab.qpu.measurement.abstract.MeasurementPlan` (circuits and shots,
fixed before anything runs) and estimating an
:class:`~mpstab.qpu.measurement.abstract.ExpectationResult` from the measured
frequencies. One module each:

- :class:`~mpstab.qpu.measurement.pauli.PauliSampling` (``"pauli"``): sampled
  Pauli strings, grouped into qubit-wise-commuting settings.
- :class:`~mpstab.qpu.measurement.shadows.ClassicalShadows` (``"shadows"``):
  random Pauli bases, contracted with the observable MPO.
- :class:`~mpstab.qpu.measurement.tnice.TNICE` (``"tnice"``): the same data as
  ``"shadows"``, estimated with the MPS-optimised dual of TN-ICE.

:mod:`~mpstab.qpu.measurement.pauli_expansion` turns the observable MPO into
Pauli strings and variances, for the methods above.
"""

from mpstab.qpu.measurement.abstract import (
    ExpectationResult,
    MeasurementMethod,
    MeasurementPlan,
)
from mpstab.qpu.measurement.pauli import (
    PauliMeasurementPlan,
    PauliSampling,
    QWCGroup,
    allocate_shots_by_variance,
    build_measurement_plan,
    build_pauli_plan,
    estimate_pauli,
    group_qwc,
)
from mpstab.qpu.measurement.shadows import (
    ClassicalShadows,
    build_shadow_plan,
    estimate_shadows,
)
from mpstab.qpu.measurement.tnice import TNICE, estimate_tnice

#: The methods by name, for the string shorthand of :func:`resolve_method`.
METHODS = {cls.name: cls for cls in (PauliSampling, ClassicalShadows, TNICE)}


def resolve_method(method, **options) -> MeasurementMethod:
    """
    A :class:`~mpstab.qpu.measurement.abstract.MeasurementMethod` from either an
    instance, returned as is, or its name, built with ``options``.

    Raises:
        ValueError: on an unknown name.
        TypeError: if ``options`` come with an instance, which already holds
            its own, or are not options of the named method.
    """
    if isinstance(method, MeasurementMethod):
        if options:
            raise TypeError(
                f"{type(method).__name__} already holds its options; got "
                f"{sorted(options)} as well."
            )
        return method
    if method not in METHODS:
        raise ValueError(
            f"Unknown method {method!r}, expected a MeasurementMethod or one of "
            f"{tuple(METHODS)}."
        )
    return METHODS[method](**options)


def estimate(plan, frequencies, seed=None, **options) -> ExpectationResult:
    """
    Recombine a plan's frequencies with the method that built it, chosen by
    ``plan.method``: the entry point for counts measured offline.

    Args:
        plan: a :class:`~mpstab.qpu.measurement.abstract.MeasurementPlan`.
        frequencies: one ``{bitstring: count}`` dict per circuit, in
            ``plan.circuits`` order.
        seed: ``"tnice"`` only: RNG seed for the train/test split.
        options: options of the method, e.g. ``n_sweeps`` for ``"tnice"``.
    """
    return resolve_method(plan.method, **options).estimate(plan, frequencies, seed=seed)


__all__ = [
    "ClassicalShadows",
    "ExpectationResult",
    "METHODS",
    "MeasurementMethod",
    "MeasurementPlan",
    "PauliMeasurementPlan",
    "PauliSampling",
    "QWCGroup",
    "TNICE",
    "allocate_shots_by_variance",
    "build_measurement_plan",
    "build_pauli_plan",
    "build_shadow_plan",
    "estimate",
    "estimate_pauli",
    "estimate_shadows",
    "estimate_tnice",
    "group_qwc",
    "resolve_method",
]
