"""
Correctness and speed of the single-pass dressed-rotation extraction
(`HSMPO._dressed_rotations`), and the Clifford comparisons every
stabilizers engine provides (`StabilizersEngine.is_identity`).

`_dressed_rotations` simulates the Clifford circuit once into a running
stim.TableauSimulator: O(nmagic + len(clifford_circuit)). The straightforward
alternative, re-backpropagating each magic gate's generator through its Clifford
prefix from scratch, is O(nmagic^2). `_reference_dressed_rotations` below spells
that alternative out, and the two must agree exactly.
"""

import time

import numpy as np
import pytest
from qibo import Circuit, gates

from mpstab.evolutors.hqsmpo import HQSMPO
from mpstab.evolutors.utils import gate_angle, gate_generator
from mpstab.models.ansatze import QAE, QFT, QPE


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
    hs = HQSMPO.rotations_only(ansatz)

    reference = _reference_dressed_rotations(hs)
    fast = hs._dressed_rotations()

    assert len(reference) == len(fast) == len(hs.magic_gates)
    for (ref_generator, ref_angle), (fast_generator, fast_angle) in zip(
        reference, fast
    ):
        assert ref_generator == fast_generator
        assert ref_angle == fast_angle


def test_dressed_rotations_is_faster_than_reference():
    # Large enough that the O(nmagic^2) reference is clearly, robustly slower
    # (~50x at n=10), while still completing in seconds: n=18 took ~3 minutes.
    ansatz = QFT(nqubits=10)
    hs = HQSMPO.rotations_only(ansatz)

    t0 = time.perf_counter()
    reference = _reference_dressed_rotations(hs)
    t_reference = time.perf_counter() - t0

    t0 = time.perf_counter()
    fast = hs._dressed_rotations()
    t_fast = time.perf_counter() - t0

    assert reference == list(fast)
    assert t_fast < t_reference, (
        f"single-pass path ({t_fast:.3f}s) not faster than the per-gate "
        f"reference ({t_reference:.3f}s)"
    )
    print(
        f"\n[dressed_rotations] QFT n=10, nmagic={len(hs.magic_gates)}: "
        f"reference={t_reference:.3f}s fast={t_fast:.3f}s "
        f"speedup={t_reference / t_fast:.1f}x"
    )


def _random_clifford_circuit(nqubits, n_gates, rng):
    single = [gates.H, gates.S, gates.SDG, gates.SX, gates.SXDG, gates.X, gates.Z]
    circuit = Circuit(nqubits)
    for _ in range(n_gates):
        if rng.random() < 0.4:
            control, target = rng.choice(nqubits, size=2, replace=False)
            circuit.add(gates.CNOT(int(control), int(target)))
        else:
            circuit.add(single[rng.integers(len(single))](int(rng.integers(nqubits))))
    return circuit


def test_identity_and_images(stab_engine):
    circuit = Circuit(2)
    assert stab_engine.is_identity(circuit)
    circuit.add(gates.CNOT(0, 1))
    assert not stab_engine.is_identity(circuit)
    circuit.add(gates.CNOT(0, 1))
    assert stab_engine.is_identity(circuit)


@pytest.mark.parametrize("seed", range(3))
def test_random_clifford_times_its_inverse_is_identity(seed, stab_engine):
    circuit = _random_clifford_circuit(3, 15, np.random.default_rng(seed))
    assert not stab_engine.is_identity(circuit)
    assert stab_engine.is_identity(circuit + circuit.invert())
