import copy

import numpy as np
from qibo import Circuit, gates

from mpstab import HSMPO


def _evolved(tn_engine):
    circuit = Circuit(4)
    for q in range(4):
        circuit.add(gates.RY(q, theta=0.3 + 0.1 * q))
    for q in range(3):
        circuit.add(gates.CNOT(q, q + 1))
        circuit.add(gates.RZ(q + 1, theta=0.7))
    hsmpo = HSMPO(circuit)
    hsmpo.set_engines(tn_engine=tn_engine)
    return hsmpo.original_circuit_mps


def test_copy_state_is_independent(tn_engine):
    original = _evolved(tn_engine)
    reference = copy.deepcopy(original)
    paulis = ["ZIII", "XXIY", "IZZI"]
    before = tn_engine.pauli_expvals(original, paulis)

    copied = tn_engine.copy_state(original)
    tn_engine.pauli_rot(copied, "XYZX", 0.9, max_bond_dimension=None)
    tn_engine.pauli_rot(reference, "XYZX", 0.9, max_bond_dimension=None)

    np.testing.assert_allclose(
        tn_engine.pauli_expvals(original, paulis), before, atol=1e-14
    )
    np.testing.assert_allclose(
        tn_engine.pauli_expvals(copied, paulis),
        tn_engine.pauli_expvals(reference, paulis),
        atol=1e-14,
    )
    # and the other way round: evolving the original leaves the copy alone
    after_copy = tn_engine.pauli_expvals(copied, paulis)
    tn_engine.pauli_rot(original, "ZZII", 0.4, max_bond_dimension=None)
    np.testing.assert_allclose(
        tn_engine.pauli_expvals(copied, paulis), after_copy, atol=1e-14
    )
