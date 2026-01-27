import random
import numpy as np
from qibo import Circuit, gates

from mpstab.analysis_scripts.utils import ( 
    execute_benchmark_circuit, 
    generate_partitionated_circuit,
    initialize_backend,
)

def get_clifford_random_params(nparams: int, seed: int):
    """Generate random parameters that are multiples of pi/2 to ensure Clifford circuits."""
    np.random.seed(seed)
    integers = np.random.randint(0, 4, size=nparams)
    return integers * (np.pi / 2)

def generate_ghz_circuit(nqubits: int) -> Circuit:

    c = Circuit(nqubits)
    c.add(gates.H(0))
    for q in range(nqubits - 1):
        c.add(gates.CNOT(q, q + 1))
    return c

def compare_backends(
    nqubits: int = 4,
    nlayers: int = 2,
    seed: int = 42,
    circuit_type: str = "random_clifford",
    max_bond_dim: int = 16
):
    print(f"\n--- Testing {circuit_type} with seed {seed} on {nqubits} qubits ---")

    np.random.seed(seed)
    random.seed(seed)

    # Prepare initial state (Clifford-compatible)
    # Use RY with angles 0, pi/2, pi, 3pi/2 to stay within the Clifford group
    initial_state = Circuit(nqubits)
    if circuit_type == "random_clifford":
        for q in range(nqubits):
            angle = np.random.randint(0, 4) * (np.pi / 2)
            initial_state.add(gates.RY(q, angle))
    else:
        # For GHZ, start from the clean zero state
        initial_state = None

    # Generate circuit
    if circuit_type == "ghz":
        circuit = generate_ghz_circuit(nqubits)
        # GHZ has no variational parameters, so skip parameter setting
    else:
        # random_clifford uses HardwareEfficient
        # magic_fraction=0.0 means replacement_probability=1.0 -> all Clifford
        circuit = generate_partitionated_circuit(
            nqubits=nqubits,
            nlayers=nlayers,
            magic_fraction=0.0, 
        )
        # Set discrete parameters (multiples of pi/2) to ensure Stim doesn't fail
        params = get_clifford_random_params(len(circuit.get_parameters()), seed)
        circuit.set_parameters(params)

    # Define observable
    obs_str = "ZX" * (nqubits // 2)
    if nqubits % 2:
        obs_str += "Z"
    
    if circuit_type == "ghz":
        obs_str = "Z" * nqubits

    print(f"Observable: {obs_str}")

    print("Running mpstab...")
    bk_mpstab = initialize_backend("mpstab", platform=None, max_bond_dim=max_bond_dim)
    res_mpstab, time_mpstab, _ = execute_benchmark_circuit(
        circuit=circuit,
        observable=obs_str,
        backend="mpstab",
        max_bond_dim=max_bond_dim,
        initial_state=initial_state,
        replacement_probability=1.0, # 1.0 = 100% Clifford replacement
        backend_obj=bk_mpstab
    )

    print("Running stim...")
    bk_stim = initialize_backend("stim", platform=None, max_bond_dim=None)
    res_stim, time_stim, _ = execute_benchmark_circuit(
        circuit=circuit,
        observable=obs_str,
        backend="stim",
        max_bond_dim=None,
        initial_state=initial_state,
        replacement_probability=1.0,
        backend_obj=bk_stim
    )

    print(f"Result mpstab: {res_mpstab:.6f} (time: {time_mpstab:.4f}s)")
    print(f"Result stim:   {res_stim:.6f}   (time: {time_stim:.4f}s)")
    
    # Note: Stim returns exact results (often integer or float x.0). 
    # Mpstab with sufficient bond_dim is exact for Clifford circuits.
    assert np.isclose(res_mpstab, res_stim, atol=1e-8), \
        f"MISMATCH! mpstab={res_mpstab}, stim={res_stim}"
    
    print("✅ SUCCESS: Results match.")


if __name__ == "__main__":
    compare_backends(nqubits=200, nlayers=3, seed=123, circuit_type="random_clifford")
    
    compare_backends(nqubits=100, seed=999, circuit_type="ghz")
    
    compare_backends(nqubits=500, nlayers=5, seed=555, circuit_type="random_clifford")