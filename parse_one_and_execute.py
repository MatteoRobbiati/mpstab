import argparse
import json
import os

import numpy as np
from qibo import hamiltonians, set_backend, symbols
from qiskit_ibm_runtime import RuntimeDecoder

from tncdr.evolutors.models import HybridSurrogate
from tncdr.targets.ansatze import FloquetAnsatz, TranspiledAnsatz

# Set Qibo backend
set_backend("numpy")


def compute_and_save(path: str, index: int, max_bond_dim: int, simulation_method: str):
    # Create results directory with bond dimension in name
    if simulation_method == "statevector":
        results_dir = os.path.join(path, f"results/statevector")
    elif simulation_method == "hsmpo":
        results_dir = os.path.join(path, f"results/hybrid_{max_bond_dim}MAXBD")
    else:
        raise ValueError(
            " The selected simulation method is not supported, please select one between 'statevector' or 'hsmpo' "
        )

    os.makedirs(results_dir, exist_ok=True)

    # Load experiment configuration
    config_file = os.path.join(path, "config.json")
    with open(config_file) as f:
        config = json.load(f)

    # Load circuit parameters for given index
    param_file = os.path.join(path, "circuits", f"params_circuit{index}.npy")
    params = np.load(param_file)

    # Build Floquet ansatz and set parameters
    floq_ansatz = FloquetAnsatz(
        nqubits=config["nqubits"],
        nlayers=config["nlayers"],
        b=config["b"],
        theta=config["theta"],
    )
    floq_ansatz.circuit.set_parameters(params)
    tran_ansatz = TranspiledAnsatz(original_circuit=floq_ansatz.circuit)

    # Construct observable Pauli string
    q = config["nqubits"] // 2
    obs = "I" * q + "X" + "I" * (config["nqubits"] - q - 1)

    # Construct Qibo hamiltonian if needed and execute on statevector mode
    if simulation_method == "statevector":
        form = 1
        for i, pauli in enumerate(obs):
            form *= getattr(symbols, pauli)(i)
        ham = hamiltonians.SymbolicHamiltonian(form=form)
        exact_val = ham.expectation(tran_ansatz.circuit().state())

    elif simulation_method == "hsmpo":
        # Compute exact expectation via HybridSurrogate
        hs = HybridSurrogate(tran_ansatz, max_bond_dimension=max_bond_dim)
        exact_val, _ = hs.expectation_from_partition(
            observable=obs, replacement_probability=0.0
        )

    # Prepare and save output array: [hardware_value, exact_value]
    output = np.array(exact_val)

    out_file = os.path.join(results_dir, f"expectation_circuit{index}.npy")
    np.save(out_file, output)
    print(f"Saved expectation values to {out_file}: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save expectation value for a single circuit with adjustable bond dimension"
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Base path to the experiment directory"
    )
    parser.add_argument(
        "--index", type=int, required=True, help="Index of the circuit to process"
    )
    parser.add_argument(
        "--max-bond-dim",
        type=int,
        default=48,
        help="Maximum bond dimension for the HybridSurrogate",
    )
    parser.add_argument(
        "--simulation_method",
        type=str,
        default="hsmpo",
        help="Simulation method. Can be 'hsmpo' or 'statevector'",
    )
    args = parser.parse_args()

    compute_and_save(args.path, args.index, args.max_bond_dim, args.simulation_method)
