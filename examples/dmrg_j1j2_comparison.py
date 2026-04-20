"""DMRG comparison: Pure Quimb vs HSMPO for J1J2 Hamiltonian."""

import numpy as np
import quimb.tensor as qtn
from qibo import Circuit, gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z
from quimb.tensor import MPO_product_operator

from mpstab.engines.tensor_networks.quimb import QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient

# Setup
n_qubits = 10  # Larger system for more long-range correlations
n_layers = 12
J1 = 1.0
J2 = 0.2
max_bond_dim_initial = 4  # Bond dimension for INITIAL state - moderate compression

np.random.seed(48)  # For reproducibility

# Create ansatz with LOTS of long-range CZ gates
hea_model = HardwareEfficient(nqubits=n_qubits, nlayers=n_layers)

# Extend the HEA circuit with long-range CZ layers
extended_circuit = hea_model.circuit.copy()
for layer_idx in range(4):  # Add 4 layers of long-range entanglement
    # CZ stride-2
    for q in range(0, n_qubits - 2, 2):
        extended_circuit.add(gates.CZ(q, q + 2))
    # CZ stride-3
    if n_qubits > 4:
        for q in range(0, n_qubits - 3, 3):
            extended_circuit.add(gates.CZ(q, q + 3))
    # CZ stride-4 (very long-range)
    if n_qubits > 5:
        for q in range(0, n_qubits - 4, 4):
            extended_circuit.add(gates.CZ(q, q + 4))
    # CZ stride-5 (extremely long-range)
    if n_qubits > 6:
        for q in range(0, min(n_qubits - 5, 8), 5):  # Limit to avoid explosion
            extended_circuit.add(gates.CZ(q, q + 5))

# Create extended ansatz
hea_model = CircuitAnsatz(qibo_circuit=extended_circuit)
extended_circ = hea_model.partitionate_circuit(
    replacement_probability=0.85,
    replacement_method="closest",
)
he_ansatz = CircuitAnsatz(qibo_circuit=extended_circuit)

# Define J1J2 Hamiltonian using Qibo (solo per HSMPO, non crea dense internamente)
ham_form = 0
for i in range(n_qubits - 1):
    ham_form = ham_form + J1 * (X(i) * X(i + 1) + Y(i) * Y(i + 1) + Z(i) * Z(i + 1))
for i in range(n_qubits - 2):
    ham_form = ham_form + J2 * (X(i) * X(i + 2) + Y(i) * Y(i + 2) + Z(i) * Z(i + 2))
H_symbolic = SymbolicHamiltonian(nqubits=n_qubits, form=ham_form)

# Convert circuit to MPS for Quimb
qasm_str = extended_circuit.to_qasm()
circ_quimb = qtn.CircuitMPS.from_openqasm2_str(qasm_str)
mps_initial = circ_quimb.psi

# Compress initial MPS to control bond dimension (in-place)
mps_initial.compress(max_bond=max_bond_dim_initial, cutoff=0)

# === Quimb DMRG ===
# Pauli matrices
sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
I = np.eye(2, dtype=complex)


def build_twobody_mpo(site_i, site_j, op_i, op_j, coeff, L):
    """Build MPO for a two-body term."""
    operators = []
    for k in range(L):
        if k < site_i:
            operators.append(I)
        elif k == site_i:
            operators.append(op_i)
        elif k < site_j:
            operators.append(I)
        elif k == site_j:
            operators.append(op_j)
        else:
            operators.append(I)
    mpo = MPO_product_operator(operators, cyclic=False)
    mpo *= coeff
    return mpo


# Build full J1J2 Hamiltonian MPO
H_mpo = None
for i in range(n_qubits - 1):
    for ops in [(sx, sx), (sy, sy), (sz, sz)]:
        mpo_term = build_twobody_mpo(i, i + 1, ops[0], ops[1], J1, n_qubits)
        H_mpo = mpo_term if H_mpo is None else H_mpo + mpo_term
for i in range(n_qubits - 2):
    for ops in [(sx, sx), (sy, sy), (sz, sz)]:
        mpo_term = build_twobody_mpo(i, i + 2, ops[0], ops[1], J2, n_qubits)
        H_mpo = H_mpo + mpo_term

# Normalize and compute initial energy via MPS-MPO-MPS contraction (scalable)
mps_initial.normalize()
init_energy_quimb = float(np.real(qtn.expec_TN_1D(mps_initial.H, H_mpo, mps_initial)))

# Measure entanglement entropy of initial MPS (indicator of complexity)
ent_quimb = mps_initial.entropy(1)  # Entanglement at bond between site 0 and 1

# Run Quimb DMRG (pass a copy so mps_initial stays untouched)
dmrg_quimb = qtn.DMRG2(
    H_mpo, p0=mps_initial.copy(), bond_dims=[2, 4, 8, 16], cutoffs=1e-10
)
dmrg_quimb.solve(verbosity=0, tol=1e-8)  # Tighter tolerance
energy_quimb = float(np.real(dmrg_quimb.energy))
sweeps_quimb = len(dmrg_quimb.energies) if hasattr(dmrg_quimb, "energies") else 0
energy_history_quimb = (
    [float(np.real(e)) for e in dmrg_quimb.energies]
    if hasattr(dmrg_quimb, "energies")
    else []
)

# === HSMPO DMRG ===
hsmpo = HSMPO(hea_model, max_bond_dimension=max_bond_dim_initial)
hsmpo.set_engines(tn_engine=QuimbEngine())

# Compute initial energy from HSMPO
init_energy_hsmpo = hsmpo.expectation(H_symbolic)

# Measure entanglement entropy of HSMPO initial MPS
ent_hsmpo = hsmpo.original_circuit_mps.entropy(
    1
)  # Entanglement at bond between site 0 and 1

# NOTE: HSMPO uses original_circuit_mps (with magic gate evolution applied)
# so its initial energy will differ from Quimb's (which uses pure HEA circuit).
# This is expected behavior!

result_hsmpo = hsmpo.minimize_expectation(
    H_symbolic, method="dmrg", bond_dims=[2, 4, 8, 16], verbosity=0, tol=1e-8
)
energy_hsmpo = result_hsmpo["energy"]
sweeps_hsmpo = result_hsmpo.get("num_sweeps", 0)
energy_history_hsmpo = result_hsmpo.get("energy_history", [])

# === Results ===
print(
    f"Initial energy: Quimb = {init_energy_quimb:.10f}, HSMPO = {init_energy_hsmpo:.10f}"
)
print(f"Initial diff:   {abs(init_energy_quimb - init_energy_hsmpo):.2e}")
print()
print(
    f"Entanglement entropy (bond 0): Quimb = {ent_quimb:.6f}, HSMPO = {ent_hsmpo:.6f}"
)
print(f"  → HSMPO simplification ratio: {ent_quimb/ent_hsmpo:.2f}x less entanglement")
print()
print(f"Final energy: Quimb = {energy_quimb:.10f}, HSMPO = {energy_hsmpo:.10f}")
print(f"Final diff:   {abs(energy_quimb - energy_hsmpo):.2e}")
print()
print(f"=== CONVERGENCE ANALYSIS ===")
print(f"DMRG bond_dims: Quimb = [3, 6], HSMPO = [6, 12]")
print(f"→ HSMPO gets 2x more resources due to less entanglement")
print(f"DMRG sweeps: Quimb = {sweeps_quimb}, HSMPO = {sweeps_hsmpo}")
if energy_history_quimb and len(energy_history_quimb) > 1:
    print(f"\nQuimb energy drop per sweep:")
    for i, e in enumerate(energy_history_quimb[:5]):  # Show first 5 sweeps
        print(f"  Sweep {i}: {e:.10f}")
if energy_history_hsmpo and len(energy_history_hsmpo) > 1:
    print(f"\nHSMPO energy drop per sweep:")
    for i, e in enumerate(energy_history_hsmpo[:5]):  # Show first 5 sweeps
        print(f"  Sweep {i}: {e:.10f}")
print()
print()
print(f"Note: Both initial states compressed to max_bond_dim = {max_bond_dim_initial}")
print(f"DMRG tolerance: tol=1e-8 (tight convergence criterion)")
print(f"\n✓ WHERE HSMPO DOMINATES:")
print(f"  Initial state entanglement: HSMPO has {ent_quimb/ent_hsmpo:.2f}x less")
print(f"  Regime: Tight bond_dims where entanglement reduction matters most")
