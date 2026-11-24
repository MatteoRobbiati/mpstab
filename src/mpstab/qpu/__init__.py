"""
Running part of an expectation value on a QPU.

:class:`~mpstab.evolutors.hqsmpo.HQSMPO` splits a circuit's dressed
rotations into a head, run on the device, and a tail, folded classically into
the observable. Getting from that split to a number is a three-step pipeline,
one subpackage each:

1. :mod:`~mpstab.qpu.synthesis` turns the head into a runnable circuit plus a
   pure-Clifford residual, folded into the observable.
2. :mod:`~mpstab.qpu.measurement` fixes the circuits and shots that measure the
   folded observable, and turns the measured frequencies into an
   :class:`~mpstab.qpu.measurement.abstract.ExpectationResult`.
3. :mod:`~mpstab.qpu.execution` runs those circuits on a backend, or exports
   them as OpenQASM to run on a device.

Synthesisers and measurement methods are objects holding their own options,
passed to :class:`~mpstab.evolutors.hqsmpo.HQSMPO`::

    from mpstab import HQSMPO
    from mpstab.qpu import ClassicalShadows, ExtractionSynthesis

    model = HQSMPO(ansatz, head_synthesis=ExtractionSynthesis(connectivity="line"))
    result = model.expectation_at_cut(H, cut_index, method=ClassicalShadows(), n_shots=10_000)

Importing this package never requires the optional ``rustiq`` dependency:
:mod:`~mpstab.qpu.synthesis.rustiq` only needs it when it synthesises.
"""

from mpstab.qpu.execution import circuit_to_qasm, plan_to_qasm, run_plan, save_circuits
from mpstab.qpu.measurement import (
    ClassicalShadows,
    ExpectationResult,
    MeasurementMethod,
    MeasurementPlan,
    PauliSampling,
    estimate,
)
from mpstab.qpu.synthesis import (
    CnotLadderSynthesis,
    ExtractionSynthesis,
    HeadSynthesis,
    RustiqSynthesis,
)

__all__ = [
    "ClassicalShadows",
    "CnotLadderSynthesis",
    "ExpectationResult",
    "ExtractionSynthesis",
    "HeadSynthesis",
    "MeasurementMethod",
    "MeasurementPlan",
    "PauliSampling",
    "RustiqSynthesis",
    "circuit_to_qasm",
    "estimate",
    "plan_to_qasm",
    "run_plan",
    "save_circuits",
]
