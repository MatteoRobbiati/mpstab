"""
Running a head/tail split on a quantum device.

The pipeline is four steps, one module each:

1. :mod:`~mpstab.quantum_hardware.synthesis` turns the head's dressed rotations
   into a runnable circuit plus a pure-Clifford residual.
2. :mod:`~mpstab.quantum_hardware.pauli_expansion` turns the tail-folded
   observable MPO back into Pauli strings with coefficients.
3. :mod:`~mpstab.quantum_hardware.plan` groups those into measurement settings
   and fixes the circuits and shot counts.
4. :mod:`~mpstab.quantum_hardware.estimate` turns the measured frequencies into
   an :class:`~mpstab.quantum_hardware.estimate.ExpectationResult`.

:mod:`~mpstab.quantum_hardware.backend` holds the default simulator backend.
Importing this package never requires the optional ``rustiq`` dependency, which
:mod:`~mpstab.quantum_hardware.synthesis` imports lazily.
"""

from mpstab.quantum_hardware.backend import QiboSimulator
from mpstab.quantum_hardware.estimate import (
    ExpectationResult,
    estimate,
    estimate_pauli,
    estimate_shadows,
)
from mpstab.quantum_hardware.pauli_expansion import (
    PauliEnsemble,
    enumerate_pauli_coefficients,
    fold_pool_through_tableau,
    mpo_site_arrays,
    mpo_to_pauli_mps,
    pool_pauli_terms,
    sample_pauli_strings,
    shadow_variance_from_mpo,
    top_k_pauli_strings,
    truncation_error_estimate,
)
from mpstab.quantum_hardware.plan import (
    MeasurementPlan,
    PauliMeasurementPlan,
    QWCGroup,
    allocate_shots_by_variance,
    build_measurement_plan,
    build_pauli_plan,
    build_shadow_plan,
    group_qwc,
)
from mpstab.quantum_hardware.synthesis import (
    build_head_and_residual,
    build_naive_head_and_residual,
    count_two_qubit_gates,
    head_counts_only,
    head_to_qibo_circuit,
)

__all__ = [
    "ExpectationResult",
    "MeasurementPlan",
    "PauliEnsemble",
    "PauliMeasurementPlan",
    "QWCGroup",
    "QiboSimulator",
    "allocate_shots_by_variance",
    "build_head_and_residual",
    "build_measurement_plan",
    "build_naive_head_and_residual",
    "build_pauli_plan",
    "build_shadow_plan",
    "count_two_qubit_gates",
    "enumerate_pauli_coefficients",
    "estimate",
    "estimate_pauli",
    "estimate_shadows",
    "fold_pool_through_tableau",
    "group_qwc",
    "head_counts_only",
    "head_to_qibo_circuit",
    "mpo_site_arrays",
    "mpo_to_pauli_mps",
    "pool_pauli_terms",
    "sample_pauli_strings",
    "shadow_variance_from_mpo",
    "top_k_pauli_strings",
    "truncation_error_estimate",
]
