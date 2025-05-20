import click

import numpy as np

from qibo import set_backend

from tncdr.targets.ansatze import HardwareEfficient
from tncdr.targets.entropies import stabilizer_renyi_entropy


@click.command()
@click.option("--nqubits", type=int, required=True, help="Number of qubits.")
@click.option("--nlayers", type=int, required=True, help="Number of layers.")
@click.option(
    "--qibo_backend",
    type=str,
    required=True,
    default="numpy",
    help="Qibo backend to use.",
)
def main(nqubits, nlayers, qibo_backend):
    """
    Execute the circuit with the specified parameters and compute SRE.
    """
    # Set the Qibo backend
    set_backend(qibo_backend)

    # Initialize the ansatz
    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)

    # Draw the circuit
    ansatz.circuit.draw()

    # Set all zeros at the beginning
    ansatz.circuit.set_parameters([np.pi] * ansatz.nparams)

    # Execute the circuit and collect the final state
    state = ansatz.circuit(nshots=1000).state()

    # Compute the Stabilizer Rényi Entropy (SRE) for the initialized state
    sre = stabilizer_renyi_entropy(state=state, alpha=2)
    print(f"\nSRE for circuit with all zero as parameters (pure Clifford): {sre}")

    # Fill circuit's parameters with random angles
    ansatz.random_unitary(random_seed=42)
    state = ansatz.circuit(nshots=1000).state()

    sre = stabilizer_renyi_entropy(state=state, alpha=2)
    print(f"SRE for circuit with all random parameters (non-Clifford): {sre}")

    # Fill the circuit with quasi-clifford angles and compute SRE
    ansatz.random_quasi_clifford_unitary(cliff_fraction=0.7, random_seed=42)
    state = ansatz.circuit(nshots=1000).state()
    sre = stabilizer_renyi_entropy(state=state, alpha=2)
    print(f"SRE for circuit for a quasi-Clifford circuit: {sre}")


if __name__ == "__main__":
    main()
