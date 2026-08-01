# Everything needed to run HSynthSMPO's head/tail measurement split on real
# hardware: resynthesizing the head circuit (rustiq_synthesis, naive_synthesis),
# extracting Pauli terms from a tail-folded MPO (tail), turning terms into
# circuits and a shot allocation (plan), and turning frequencies back into a
# result (estimate). The optional rustiq dependency is isolated with lazy
# imports inside rustiq_synthesis -- importing this package itself never
# requires it.
from .estimate import ExpectationResult, estimate, estimate_pauli, estimate_shadows
from .naive_synthesis import build_naive_head_and_residual
from .plan import (
    MeasurementPlan,
    PauliMeasurementPlan,
    QiboSimulator,
    QWCGroup,
    allocate_shots_by_variance,
    build_measurement_plan,
    build_pauli_plan,
    build_shadow_plan,
    group_qwc,
)
from .rustiq_synthesis import (
    build_head_and_residual,
    fold_observable,
    head_counts_only,
    head_to_qibo_circuit,
)
from .tail import (
    PauliEnsemble,
    enumerate_pauli_coefficients,
    fold_pool_through_tableau,
    hamming_weight,
    mpo_site_arrays,
    mpo_to_pauli_mps,
    pool_pauli_terms,
    sample_pauli_strings,
    shadow_variance_from_mpo,
    top_k_pauli_strings,
    truncation_error_estimate,
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
    "enumerate_pauli_coefficients",
    "estimate",
    "estimate_pauli",
    "estimate_shadows",
    "fold_observable",
    "fold_pool_through_tableau",
    "group_qwc",
    "hamming_weight",
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
