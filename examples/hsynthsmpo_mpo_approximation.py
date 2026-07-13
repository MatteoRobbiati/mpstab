"""HSynthSMPO: how folding rotations into the observable MPO introduces error.

The HSynthSMPO splits the chain of dressed Pauli rotations at a cut index:

  * the *head* rotations (closer to the initial state) are resynthesized with
    Qiskit's Rustiq backend and applied exactly to build the state MPS;
  * the *tail* rotations (closer to the observable) are folded into the
    observable as an MPO by Heisenberg conjugation R^dag . O . R.

Each conjugation grows the observable MPO bond dimension, so with a finite bond
cap the folded MPO must be truncated -- this is the *only* source of
approximation in the tail. This script sweeps the cut index (hence the tail
length) at a fixed bond cap and shows the approximation growing as more
rotations are pushed into the MPO.

Requires the optional transpilation extra:  pip install "mpstab[transpilation]"
"""

import numpy as np
from qibo import set_backend

from mpstab import HSynthSMPO
from mpstab.models.ansatze import HardwareEfficient

set_backend("numpy")
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. An entangling ansatz with several non-Clifford (magic) rotations.
# ---------------------------------------------------------------------------
NQUBITS = 8
NLAYERS = 3
MAX_BOND = 4  # deliberately tight so the MPO truncation is visible
OBSERVABLE = "Z" * NQUBITS

ansatz = HardwareEfficient(nqubits=NQUBITS, nlayers=NLAYERS)
hs = HSynthSMPO(ansatz, max_bond_dimension=MAX_BOND)

n_dressed = len(hs.magic_gates)
print(f"qubits={NQUBITS}  layers={NLAYERS}  bond cap={MAX_BOND}")
print(f"dressed (magic) rotations: {n_dressed}")
print(f"observable: {OBSERVABLE}\n")

# ---------------------------------------------------------------------------
# 2. Sweep the cut index.
#
#    cut_index = k  ->  k rotations go into the (exact) state circuit,
#                       (n_dressed - k) rotations are folded into the MPO tail.
#
#    So SMALLER cut_index  ==  LONGER MPO tail  ==  MORE approximation.
# ---------------------------------------------------------------------------
header = (
    f"{'cut':>4} {'tail':>5} {'2q_head':>8} "
    f"{'expval':>11} {'exp_err':>11} {'rel_F_err':>11} "
    f"{'approx_bd':>10} {'exact_bd':>9}"
)
print(header)
print("-" * len(header))

for cut_index in range(n_dressed, -1, -1):
    info = hs.mpo_tail_approximation(
        OBSERVABLE, cut_index=cut_index, reference_max_bond=None
    )
    counts = hs.count_two_qubit_gates(cut_index)
    print(
        f"{cut_index:>4} {n_dressed - cut_index:>5} "
        f"{counts['synthesized_head_2q_gates']:>8} "
        f"{info['expval_approx']:>11.5f} "
        f"{info['expval_abs_error']:>11.3e} "
        f"{info['relative_frobenius_error']:>11.3e} "
        f"{info['approx_max_bond']:>10} {info['reference_max_bond']:>9}"
    )

print(
    "\nReading the table:\n"
    "  * cut = n_dressed (top row): the whole circuit is resynthesized into the\n"
    "    state, the MPO tail is empty -> exact operator (rel_F_err = 0).\n"
    "  * cut = 0 (bottom row): every rotation is folded into the MPO. The exact\n"
    "    MPO bond (exact_bd) blows up while approx_bd stays at the cap, so the\n"
    "    truncation -- and the resulting error -- is largest here.\n"
    "  * exp_err is the impact on the actual expectation value; it is typically\n"
    "    much smaller than the operator-level rel_F_err, because only the part of\n"
    "    the MPO overlapping the state matters."
)
