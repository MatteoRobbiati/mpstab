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

:mod:`~mpstab.qpu.measurement.pauli_expansion` turns the observable MPO into
Pauli strings and variances, for the methods above.
"""

from mpstab.qpu.measurement.abstract import (
    ExpectationResult,
    MeasurementMethod,
    MeasurementPlan,
)
from mpstab.qpu.measurement.pauli import PauliSampling
from mpstab.qpu.measurement.shadows import ClassicalShadows

#: The methods by name, for the string shorthand of :func:`resolve_method`.
METHODS = {cls.name: cls for cls in (PauliSampling, ClassicalShadows)}


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
        seed: forwarded to the method's ``estimate``; the built-in methods
            ignore it, since their estimates involve no randomness.
        options: options of the method, e.g. ``tail_handling`` for
            ``"shadows"``.
    """
    return resolve_method(plan.method, **options).estimate(plan, frequencies, seed=seed)


__all__ = [
    "ClassicalShadows",
    "ExpectationResult",
    "MeasurementMethod",
    "MeasurementPlan",
    "PauliSampling",
    "estimate",
    "resolve_method",
]
