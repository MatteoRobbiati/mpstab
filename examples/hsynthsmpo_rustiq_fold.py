"""HSynthSMPO: measuring a fully-resynthesized head via the "pauli" route.

Resynthesizes every dressed Pauli rotation into hardware-native gates with the
low-level ``rustiq.pauli_network_synthesis`` API (falling back to a
dependency-free CNOT-ladder decomposition if ``rustiq`` isn't installed), and
measures the resulting circuit with a finite shot budget via
``expectation_at_cut``. The resynthesis Clifford residual (if any) is folded
into the sampled Pauli terms automatically -- see the module docstring of
``mpstab.evolutors.hsynthsmpo`` for why that fold has to happen before QWC
grouping. ``expectation_from_split`` (exact, no resynthesis, no shot noise) is
used here only as the reference the shot-based estimate is checked against.

With the optional rustiq extra installed (``pip install "mpstab[rustiq]"``)
the head is gate-count-optimized; otherwise the naive fallback is used, at
the cost of that saving (see ``resynthesize_head``'s docstring).
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
# 1. Resynthesize ALL dressed rotations into a hardware-native circuit.
# ---------------------------------------------------------------------------
resynth = hs.resynthesize_head(cut_index=n_dressed)
print(f"\nresynthesis method: {resynth.method}")
print(
    f"resynthesized head circuit: {resynth.n_gates} gates ({resynth.n_two_qubit_gates} two-qubit)"
)

# ---------------------------------------------------------------------------
# 2. Measure it with a finite shot budget, and compare against the exact
#    (no-resynthesis, no shot-noise) reference.
# ---------------------------------------------------------------------------
exact = hs.expectation_from_split(OBSERVABLE, cut_index=n_dressed)
result = hs.expectation_at_cut(
    OBSERVABLE, cut_index=n_dressed, method="pauli", n_shots=20000, seed=0
)
print(
    f"\nexpectation_from_split({OBSERVABLE!r}) = {exact:+.6f}  (exact, reference only)"
)
print(f"expectation_at_cut({OBSERVABLE!r}, method='pauli') = {result!r}")
print(
    f"|estimate - exact| = {abs(result.value - exact):.3e}  (total_error = {result.total_error:.3e})"
)

# ---------------------------------------------------------------------------
# 3. Gate-count profiling: resynthesis cost of increasingly large prefixes of
#    the dressed-rotation chain.
# ---------------------------------------------------------------------------
print(f"\n{'cut':>5} | {'head total':>10} {'head 2Q':>8} {'tail trivial':>12}")
for cut in sorted({0, n_dressed // 2, n_dressed}):
    r = hs.resynthesize_head(cut)
    import stim

    trivial = r.tail_tableau == stim.Tableau(hs.nqubits)
    print(f"{cut:>5} | {r.n_gates:>10} {r.n_two_qubit_gates:>8} {str(trivial):>12}")

print(
    "\nNote: cut < n_dressed leaves rotations out of the resynthesis; those "
    "are still non-Clifford dressed rotations and are folded (approximately, "
    "as a truncated tensor-network operator) into the tail MPO by the "
    "'pauli'/'shadows' routes exactly as an un-resynthesized cut would be. "
    "For a look at that approximation on its own, see "
    "examples/hsynthsmpo_mpo_approximation.py, which uses "
    "HSynthSMPO.tail_truncation."
)
