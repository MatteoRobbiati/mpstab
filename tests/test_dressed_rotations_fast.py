"""
Correctness and speed of the fast single-pass dressed-rotation extraction
(`mpstab.evolutors.utils.dressed_rotations`, used by both
`HSMPO._precompute_original_mps` and `HSynthSMPO._dressed_rotations`).

The "old" implementation (kept as `HSMPO._clifford_subcircuit` +
`HSMPO._conjugate_generator`, no longer called from the hot loops) re-simulates
the Clifford prefix from scratch for every magic gate: O(nmagic^2). The fast
path simulates the Clifford circuit once into a running stim.TableauSimulator:
O(nmagic + len(clifford_circuit)). Output must be bit-identical.
"""
import time

import pytest
from qibo import set_backend

from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import QAE, QFT, QPE

set_backend("numpy")


def _old_dressed_rotations(hs):
    """The pre-optimization algorithm: re-backpropagate from scratch per gate."""
    rotations = []
    for k, magic_gate in hs.magic_gates:
        clifford_subcircuit = hs._clifford_subcircuit(hs.clifford_circuit, k)
        generator, sign = hs._conjugate_generator(magic_gate, clifford_subcircuit)
        rotations.append((generator, hs._gate_angle(magic_gate) * sign))
    return rotations


_LIBRARY_ANSATZE = [
    ("QFT", lambda: QFT(nqubits=10)),
    ("QPE", lambda: QPE(n_counting=6)),
    ("QAE", lambda: QAE(n_counting=6)),
]


@pytest.mark.parametrize(
    "label,factory", _LIBRARY_ANSATZE, ids=[a[0] for a in _LIBRARY_ANSATZE]
)
def test_dressed_rotations_fast_matches_old(label, factory):
    ansatz = factory()
    hs = HSynthSMPO.rotations_only(ansatz)

    old = _old_dressed_rotations(hs)
    new = hs._dressed_rotations()

    assert len(old) == len(new) == len(hs.magic_gates)
    for (old_generator, old_angle), (new_generator, new_angle) in zip(old, new):
        assert old_generator == new_generator
        assert old_angle == new_angle


def test_dressed_rotations_fast_is_faster():
    # Large enough that the O(nmagic^2) old path is clearly, robustly slower,
    # while still completing quickly.
    ansatz = QFT(nqubits=18)
    hs = HSynthSMPO.rotations_only(ansatz)

    t0 = time.perf_counter()
    old = _old_dressed_rotations(hs)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    new = hs._dressed_rotations()
    t_new = time.perf_counter() - t0

    assert old == new
    assert t_new < t_old, f"fast path ({t_new:.3f}s) not faster than old ({t_old:.3f}s)"
    print(
        f"\n[dressed_rotations] QFT n=18, nmagic={len(hs.magic_gates)}: "
        f"old={t_old:.3f}s new={t_new:.3f}s speedup={t_old / t_new:.1f}x"
    )
