"""
Correctness and speed of the single-pass dressed-rotation extraction
(`HSMPO._dressed_rotations`), and of the rotation conjugation it relies on
(`HSMPO._conjugate_rotation`).

`_dressed_rotations` simulates the Clifford circuit once into a running
stim.TableauSimulator: O(nmagic + len(clifford_circuit)). The straightforward
alternative, re-backpropagating each magic gate's generator through its Clifford
prefix from scratch, is O(nmagic^2). `_reference_dressed_rotations` below spells
that alternative out, and the two must agree exactly.
"""

import time
from functools import reduce

import numpy as np
import pytest
import stim
from qibo import set_backend

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.evolutors.utils import gate_angle, gate_generator
from mpstab.models.ansatze import QAE, QFT, QPE
from mpstab.pauli import PAULI_LABELS, PAULI_MATRICES, validate_pauli_string

set_backend("numpy")


def _reference_dressed_rotations(hs):
    """Re-backpropagate every magic gate's generator from scratch, per gate."""
    rotations = []
    for breakpoint_index, magic_gate in hs.magic_gates:
        prefix = hs._clifford_subcircuit(hs.clifford_circuit, breakpoint_index)
        generator, sign = hs.stab_engine.backpropagate(
            gate_generator(magic_gate, hs.nqubits), prefix
        )
        rotations.append((generator, gate_angle(magic_gate) * sign))
    return rotations


_LIBRARY_ANSATZE = [
    ("QFT", lambda: QFT(nqubits=10)),
    ("QPE", lambda: QPE(n_counting=6)),
    ("QAE", lambda: QAE(n_counting=6)),
]


@pytest.mark.parametrize(
    "label,factory", _LIBRARY_ANSATZE, ids=[a[0] for a in _LIBRARY_ANSATZE]
)
def test_dressed_rotations_match_reference(label, factory):
    ansatz = factory()
    hs = HSynthSMPO.rotations_only(ansatz)

    reference = _reference_dressed_rotations(hs)
    fast = hs._dressed_rotations()

    assert len(reference) == len(fast) == len(hs.magic_gates)
    for (ref_generator, ref_angle), (fast_generator, fast_angle) in zip(
        reference, fast
    ):
        assert ref_generator == fast_generator
        assert ref_angle == fast_angle


def test_dressed_rotations_is_faster_than_reference():
    # Large enough that the O(nmagic^2) reference is clearly, robustly slower,
    # while still completing quickly.
    ansatz = QFT(nqubits=18)
    hs = HSynthSMPO.rotations_only(ansatz)

    t0 = time.perf_counter()
    reference = _reference_dressed_rotations(hs)
    t_reference = time.perf_counter() - t0

    t0 = time.perf_counter()
    fast = hs._dressed_rotations()
    t_fast = time.perf_counter() - t0

    assert reference == fast
    assert t_fast < t_reference, (
        f"single-pass path ({t_fast:.3f}s) not faster than the per-gate "
        f"reference ({t_reference:.3f}s)"
    )
    print(
        f"\n[dressed_rotations] QFT n=18, nmagic={len(hs.magic_gates)}: "
        f"reference={t_reference:.3f}s fast={t_fast:.3f}s "
        f"speedup={t_reference / t_fast:.1f}x"
    )


def _rotation(generator: str, angle: float) -> np.ndarray:
    """Dense ``exp(-i angle/2 G)`` for a Pauli string ``G``, qubit 0 leftmost."""
    dense = reduce(np.kron, [PAULI_MATRICES[label] for label in generator])
    return np.cos(angle / 2) * np.eye(2 ** len(generator)) - 1j * np.sin(
        angle / 2
    ) * dense


@pytest.mark.parametrize("seed", range(5))
def test_conjugate_rotation_matches_dense_random_clifford(seed):
    tableau = stim.Tableau.random(3)
    unitary = tableau.to_unitary_matrix(endian="big")
    rng = np.random.default_rng(seed)
    generator = "".join(rng.choice(list(PAULI_LABELS[1:]), size=3))
    angle = float(rng.uniform(-np.pi, np.pi))

    conjugated, conjugated_angle = HSMPO._conjugate_rotation(
        (generator, angle), tableau
    )

    validate_pauli_string(conjugated, 3)
    assert abs(conjugated_angle) == pytest.approx(abs(angle))
    np.testing.assert_allclose(
        _rotation(conjugated, conjugated_angle),
        unitary @ _rotation(generator, angle) @ unitary.conj().T,
        atol=1e-12,
    )


def test_conjugate_rotation_moves_sign_into_angle():
    # H Y H^dag = -Y, so exp(-i a/2 Y) becomes exp(+i a/2 Y).
    tableau = stim.Tableau.from_named_gate("H")
    assert HSMPO._conjugate_rotation(("Y", 0.3), tableau) == ("Y", -0.3)
