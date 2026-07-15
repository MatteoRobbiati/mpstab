"""HSynthSMPO: exact expectation values via foldable low-level-rustiq resynthesis.

Resynthesizes every dressed Pauli rotation into hardware-native gates with the
low-level ``rustiq.pauli_network_synthesis`` API, and reabsorbs the resulting
pure-Clifford residual exactly into the observable via ``StimEngine`` -- no
tensor-network truncation involved (contrast with ``mpo_tail_approximation``,
which folds a partial tail as an *approximate*, truncated MPO).

Requires the optional rustiq extra:  pip install "mpstab[rustiq]"
"""

import numpy as np
from qibo import set_backend

from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import HardwareEfficient

set_backend("numpy")
np.random.seed(42)

NQUBITS = 6
NLAYERS = 2
OBSERVABLE = "Z" * NQUBITS

ansatz = HardwareEfficient(nqubits=NQUBITS, nlayers=NLAYERS)

# rotations_only skips the eager MPS precompute in __post_init__ -- only the
# dressed rotations are needed here, so construction stays cheap regardless of
# system size (see docs/guides/rustiq_resynthesis.rst).
hs = HSynthSMPO.rotations_only(ansatz)
n_dressed = len(hs.magic_gates)
print(f"qubits={NQUBITS}  layers={NLAYERS}  dressed rotations={n_dressed}")

# ---------------------------------------------------------------------------
# 1. Exact expectation value: resynthesize ALL dressed rotations, fold the
#    resulting pure-Clifford tail exactly into the observable.
# ---------------------------------------------------------------------------
expval = hs.expectation_from_rustiq_fold(OBSERVABLE)
print(f"\nexpectation_from_rustiq_fold({OBSERVABLE!r}) = {expval:+.6f}")

# ---------------------------------------------------------------------------
# 2. The actual hardware-native circuit you would run.
# ---------------------------------------------------------------------------
head_circuit = hs.foldable_head_circuit(cut_index=n_dressed)
print(f"resynthesized head circuit: {len(head_circuit.queue)} gates")

# ---------------------------------------------------------------------------
# 3. Gate-count profiling: resynthesis cost of increasingly large prefixes of
#    the dressed-rotation chain (cheap -- a single rustiq call per cut, no
#    rotation placement).
# ---------------------------------------------------------------------------
print(f"\n{'cut':>5} | {'head total':>10} {'head 2Q':>8} {'original 2Q':>12}")
for cut in sorted({0, n_dressed // 2, n_dressed}):
    counts = hs.foldable_head_gate_counts(cut)
    print(
        f"{cut:>5} | {counts['synthesized_head_total_gates']:>10} "
        f"{counts['synthesized_head_2q_gates']:>8} "
        f"{counts['original_circuit_2q_gates']:>12}"
    )

print(
    "\nNote: cut < n_dressed is only meaningful for gate-count profiling here. "
    "Combining a partial rustiq resynthesis with an exact expectation value "
    "isn't possible in general -- the rotations left out are still "
    "non-Clifford, so their Heisenberg conjugation isn't a single Pauli. "
    "For that regime (folding a partial tail into the observable), see "
    "mpo_tail_approximation in examples/hsynthsmpo_mpo_approximation.py, "
    "which uses a truncated tensor-network operator instead."
)
