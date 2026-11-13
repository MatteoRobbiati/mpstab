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
    ExpectationResult,
    MeasurementPlan,
    build_pauli_plan,
    build_shadow_plan,
    estimate,
)
from mpstab.qpu.synthesis import (
    build_cnot_ladder_head_and_residual,
    build_head_and_residual,
    head_to_qibo_circuit,
)

__all__ = [
    "ExpectationResult",
    "MeasurementPlan",
    "QiboSimulator",
    "build_cnot_ladder_head_and_residual",
    "build_head_and_residual",
    "build_pauli_plan",
    "build_shadow_plan",
    "circuit_to_qasm",
    "estimate",
    "execute_plan",
    "head_to_qibo_circuit",
    "plan_to_qasm",
    "save_circuits",
]
