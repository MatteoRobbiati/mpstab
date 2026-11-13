"""
Running part of an expectation value on a QPU.

:class:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO` splits a circuit's dressed
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
passed to :class:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO`::

    from mpstab import HSynthSMPO
    from mpstab.qpu import ClassicalShadows, ExtractionSynthesis

    model = HSynthSMPO(ansatz, head_synthesis=ExtractionSynthesis(connectivity="line"))
    result = model.expectation_at_cut(H, cut_index, method=ClassicalShadows(), n_shots=10_000)

Importing this package never requires the optional ``rustiq`` dependency, which
:mod:`~mpstab.qpu.synthesis.rustiq` imports lazily.
"""

from mpstab.qpu.execution import (
    QiboSimulator,
    circuit_to_qasm,
    execute_plan,
    plan_to_qasm,
    save_circuits,
)
from mpstab.qpu.measurement import (
    TNICE,
    ClassicalShadows,
    ExpectationResult,
    MeasurementMethod,
    MeasurementPlan,
    PauliSampling,
    estimate,
)
from mpstab.qpu.synthesis import (
    CnotLadderSynthesis,
    EdgeGreedySynthesis,
    ExtractionSynthesis,
    HeadSynthesis,
    RustiqSynthesis,
    SteinerSynthesis,
)

__all__ = [
    "ClassicalShadows",
    "CnotLadderSynthesis",
    "EdgeGreedySynthesis",
    "ExpectationResult",
    "ExtractionSynthesis",
    "HeadSynthesis",
    "MeasurementMethod",
    "MeasurementPlan",
    "PauliSampling",
    "QiboSimulator",
    "RustiqSynthesis",
    "SteinerSynthesis",
    "TNICE",
    "circuit_to_qasm",
    "estimate",
    "execute_plan",
    "plan_to_qasm",
    "save_circuits",
]
