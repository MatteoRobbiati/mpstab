import json
import os
import random

import click
import matplotlib.pyplot as plt
import numpy as np
from qibo import get_backend, hamiltonians, set_backend, symbols
from scipy.stats import median_abs_deviation
from tncdr.models.ansatze import FloquetAnsatz, TranspiledAnsatz
from tncdr.models.mitigation_methods import TNCDR
from tncdr.models.utils import build_noise_model


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return float(obj)
        return super().default(obj)


@click.command()
@click.option(
    "--nqubits",
    type=int,
    default=9,
    show_default=True,
    help="Number of qubits in the problem.",
)
@click.option(
    "--nlayers",
    type=int,
    default=2,
    show_default=True,
    help="Number of layers in the ansatz.",
)
@click.option(
    "--b",
    type=float,
    default=0.4 * np.pi,
    show_default=True,
    help="Magic parameter of the ansatz.",
)
@click.option(
    "--theta",
    type=float,
    default=0.5 * np.pi,
    show_default=True,
    help="Rotation angle theta for the Floquet ansatz.",
)
@click.option(
    "--replacement-probability",
    type=float,
    default=0.5,
    show_default=True,
    help="Replacement probability for TNCDR.",
)
@click.option(
    "--ncircuits",
    type=int,
    default=20,
    show_default=True,
    help="Number of noisy circuits to generate for training.",
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
@click.option(
    "--local-pauli-noise-sigma",
    type=float,
    default=0.003,
    show_default=True,
    help="Local Pauli noise sigma parameter for the noise model.",
)
@click.option(
    "--max-bond-dimension",
    type=int,
    default=128,
    show_default=True,
    help="Maximum bond dimension for TNCDR method.",
)
@click.option(
    "--nruns",
    type=int,
    default=10,
    show_default=True,
    help="Number of times the whole mitigation routine is run.",
)
@click.option(
    "--plot",
    default=True,
    help="Enable or disable saving of the training scatter plot.",
)
@click.option(
    "--save-results",
    default=True,
    help="Enable or disable dumping of results and hyperparameters to JSON.",
)
def main(
    nqubits,
    nlayers,
    b,
    theta,
    replacement_probability,
    ncircuits,
    random_seed,
    local_pauli_noise_sigma,
    max_bond_dimension,
    nruns,
    plot,
    save_results,
):
    """
    CLI for running TNCDR experiment with hyperparameter parsing via click.
    """

    # Set Qibo backend
    # TODO: parse it, if necessary
    set_backend("numpy")

    # Collect initial parameters
    params = locals().copy()

    # Fix all random seeds
    np.random.seed(random_seed)
    random.seed(random_seed)
    backend = get_backend()
    backend.set_seed(random_seed)

    # Build target observable
    # namely "IIIXIII", a list of identities with an X in the central qubit
    obs = "".join("X" if i == int(nqubits / 2) else "I" for i in range(nqubits))
    # and the correspondent Qibo Hamiltonian
    form = 1
    for i, pauli in enumerate(obs):
        form *= getattr(symbols, pauli)(i)
    ham = hamiltonians.SymbolicHamiltonian(form=form)

    # Initialize and transpile ansatz
    ansatz = FloquetAnsatz(
        nqubits=nqubits,
        nlayers=nlayers,
        b=b,
        target_qubit=int(nqubits / 2),
        theta=theta,
        density_matrix=True,
    )

    ansatz = TranspiledAnsatz(original_circuit=ansatz.circuit)

    # Computing exact expectation value of the original circuit
    exact_expval = ham.expectation(ansatz.circuit().state())

    # Lists where we save all results
    mit_values, noisy_values, noise_maps = [], [], []

    # Repeat the experiment nruns times
    for _ in range(nruns):

        # Build noise model
        noise_model = build_noise_model(
            nqubits=nqubits,
            local_pauli_noise_sigma=local_pauli_noise_sigma,
        )

        # Run TNCDR
        training_data, fit_map = TNCDR(
            observable=obs,
            ansatz=ansatz,
            noise_model=noise_model,
            replacement_probability=replacement_probability,
            ncircuits=ncircuits,
            random_seed=np.random.randint(0, 1000000),
            max_bond_dimension=max_bond_dimension,
        )

        # Compute noisy and mitigated expectation values
        noisy_values.append(
            ham.expectation(noise_model.apply(ansatz.circuit)().state())
        )
        mit_values.append(fit_map[0] * noisy_values[-1] + fit_map[1])

        # Save noise map
        noise_maps.append(fit_map)

    # Update params with results
    params.update(
        exact_expval=exact_expval,
        median_mit_value=np.median(mit_values),
        median_abs_deviation_mit_value=median_abs_deviation(mit_values),
    )

    # Save to folder if requested
    if save_results:
        folder_name = (
            f"results/floquet/"
            f"{nqubits}q_{nlayers}l_rprob{replacement_probability}_"
            f"nc{ncircuits}_seed{random_seed}_sigma{local_pauli_noise_sigma}_"
            f"mbd{max_bond_dimension}"
        )
        os.makedirs(folder_name, exist_ok=True)

        # Save noisy and mitigated values
        np.save(file=os.path.join(folder_name, "mit_values"), arr=np.array(mit_values))
        np.save(
            file=os.path.join(folder_name, "noisy_values"), arr=np.array(noisy_values)
        )
        np.save(file=os.path.join(folder_name, "fit_maps"), arr=np.array(noise_maps))

        # Dump JSON
        json_path = os.path.join(folder_name, "results.json")
        with open(json_path, "w") as jf:
            json.dump(params, jf, indent=4, cls=NumpyEncoder)
        click.echo(f"Results saved to {json_path}")

        # Plot and save if requested
        if plot:
            fig, ax = plt.subplots(figsize=(5, 5 * 6 / 8))
            x = np.linspace(
                min(training_data["noisy_expvals"]),
                max(training_data["noisy_expvals"]),
                100,
            )
            ax.scatter(
                training_data["noisy_expvals"],
                training_data["exact_expvals"],
                label="Training data",
                color="purple",
            )
            ax.plot(
                x,
                fit_map[0] * x + fit_map[1],
                ls="-",
                lw=1.5,
                label="TNCDR fit",
                color="black",
            )
            ax.set_xlabel("Noisy expvals")
            ax.set_ylabel("Exact expvals")
            ax.legend()
            plt.tight_layout()
            plot_path = os.path.join(folder_name, "training_plot.pdf")
            fig.savefig(plot_path)
            click.echo(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
