import os
import json
import click
import random
import numpy as np
from scipy.stats import median_abs_deviation

# Qibo and tncdr imports
from qibo import Circuit, gates, symbols, hamiltonians, set_backend
from qibo.models.error_mitigation import CDR, vnCDR  # vnCDR imported if needed later

from tncdr.targets.ansatze import HardwareEfficient
from tncdr.targets.noise_utils import build_noise_model
from tncdr.mitigation.methods import TNCDR, density_matrix_circuit

# Custom JSON encoder for NumPy objects
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return float(obj)
        return super().default(obj)

@click.command()
@click.option('--nqubits', default=5, type=int, help='Number of qubits.')
@click.option('--nlayers', default=3, type=int, help='Number of layers in the ansatz.')
@click.option('--nshots', default=10000, type=int, help='Number of shots per circuit.')
@click.option('--ncircuits', default=20, type=int, help='Number of circuits (training samples).')
@click.option('--ansatz', default='HardwareEfficient', type=str, help='Ansatz name (currently only HardwareEfficient supported).')
@click.option('--observable', default=None, type=str,
              help='Observable as a string (e.g. "ZZZZZ"). Defaults to "Z" repeated nqubits times if not provided.')
@click.option('--readout_bitflip_probability', default=0.005, type=float, 
              help='Probability of applying bit-flip in the readout noise simulation.')
@click.option('--local_pauli_noise_sigma', default=0.005, type=float, 
              help='Local Pauli noise sigma: the noise rate is sampled from abs(N(0, sigma)).')
@click.option('--mitigation_method', default='TNCDR', type=str,
              help='Error mitigation method to use (e.g., TNCDR, CDR).')
@click.option('--mitigation_args', default='{}', type=str,
              help='JSON string of additional arguments for the mitigation method. '
                   'For TNCDR, e.g.: \'{"npartitions": 2, "magic_gates_per_partition": 1, "max_bond_dimension": null}\'')
@click.option('--random_seed', default=42, type=int, help='Seed of the random number generator.')
@click.option('--nruns', default=20, type=int, help='Number of times the exercise is repeated.')
def main(
    nqubits, 
    nlayers, 
    nshots, 
    ncircuits, 
    ansatz, 
    observable,
    readout_bitflip_probability,
    local_pauli_noise_sigma, 
    mitigation_method, 
    mitigation_args, 
    random_seed,
    nruns,
):

    # Parse extra mitigation arguments.
    try:
        mitigation_args_dict = json.loads(mitigation_args)
    except json.JSONDecodeError:
        click.echo("Error: mitigation_args must be a valid JSON string.")
        return

    mitigation_method_upper = mitigation_method.upper()

    # Constructing the output results dict
    out_results = locals().copy()

    # Base folder: results/nqubits_nlayers_ansatz_observable
    base_folder_name = f"{nqubits}qubits_{nlayers}layers_{ansatz}_{observable}"
    base_folder_path = os.path.join("results", base_folder_name)
    
    # Create folder for mitigation method.
    method_folder_path = os.path.join(base_folder_path, mitigation_method_upper)
    
    # Now, for each mitigation argument, create a subfolder.
    mitigation_subfolder = method_folder_path + f"/ncircuits_{ncircuits}"
    if mitigation_args_dict:
        for i, (key, value) in enumerate(mitigation_args_dict.items()):
            if i == 0:
                mitigation_subfolder += f"{key}_{value}"
            else:
                mitigation_subfolder += f"_{key}_{value}"
    
    # Add an extra slash (i.e. final directory) after the mitigation method.
    folder_path = mitigation_subfolder
    os.makedirs(folder_path, exist_ok=True)
    
    # Set the Qibo backend.
    set_backend("numpy")
    if observable is None:
        observable = "Z" * nqubits

    # Construct Hamiltonian.
    form = 1
    for i, pauli in enumerate(observable):
        form *= getattr(symbols, pauli)(i)
    ham = hamiltonians.SymbolicHamiltonian(form=form)
    
    # Build the ansatz (only HardwareEfficient is supported here).
    if ansatz != "HardwareEfficient":
        click.echo("Only 'HardwareEfficient' ansatz is supported.")
        return
    ansatz_instance = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)

    # Fix random seed.
    np.random.seed(random_seed)
    random.seed(random_seed)

    # The whole experiment will be repeated nruns times.
    exact_values, noisy_values, mit_values = [], [], []
    for i in range(nruns):
        click.echo(f"Running experiment {i+1}/{nruns}")
        
        # Update parameters in the ansatz.
        ansatz_instance.circuit.set_parameters(np.random.randn(ansatz_instance.nparams))

        # Build the initial state circuit.
        init_circ = Circuit(nqubits=nqubits)
        for q in range(nqubits):
            init_circ.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))

        # Build the noise model.
        noise_model = build_noise_model(
            nqubits=nqubits, 
            readout_bit_flip_prob=readout_bitflip_probability, 
            local_pauli_noise_sigma=local_pauli_noise_sigma,
        )

        # Define common parameters (used by all mitigation methods).
        common_params = {
            "noise_model": noise_model,
            "nshots": nshots
        }

        # Define mitigation methods mapping.
        methods = {
            "TNCDR": {
                "func": TNCDR,
                "params": {
                    "observable": observable,  # TNCDR expects the observable as a string.
                    "ansatz": ansatz_instance,
                    "initial_state": init_circ,
                    "ncircuits": ncircuits,
                    "random_seed": np.random.randint(0, 1000000),
                    # TNCDR-specific parameters will be provided via mitigation_args.
                }
            },
            "CDR": {
                "func": CDR,
                "params": {
                    "circuit": density_matrix_circuit(init_circ + ansatz_instance.circuit),
                    "observable": ham,
                    "n_training_samples": ncircuits,
                    "replacement_gates": [(gates.RY, {"theta": n * np.pi / 2}) for n in range(4)],
                    "target_non_clifford_gates": [gates.RY],
                    "full_output": True
                }
            }
        }

        if mitigation_method_upper not in methods:
            click.echo(f"Mitigation method '{mitigation_method}' not supported.")
            return

        method_info = methods[mitigation_method_upper]
        method_func = method_info["func"]
        method_params = method_info["params"]

        # Merge common parameters and extra mitigation arguments.
        method_params.update(common_params)
        method_params.update(mitigation_args_dict)

        # Run the selected error mitigation method.
        mitigation_output = method_func(**method_params)

        # Compute extra info common to all methods: the exact and noisy expectation values.
        exact_circ = init_circ + ansatz_instance.circuit
        exact_value = ham.expectation(exact_circ().state())
        noisy_init_circ = noise_model.apply(density_matrix_circuit(init_circ))
        noisy_main_circ = noise_model.apply(density_matrix_circuit(ansatz_instance.circuit))
        noisy_outcome = (noisy_init_circ + noisy_main_circ)()
        noisy_value = ham.expectation(noisy_outcome.state())

        # Process the result.
        if mitigation_method_upper == "TNCDR":
            # For TNCDR, assume the output is a tuple where index 1 contains [slope, intercept].
            mit_map_params = mitigation_output[1]
            mit_value = mit_map_params[0] * noisy_value + mit_map_params[1]
        elif mitigation_method_upper == "CDR":
            mit_value = mitigation_output[0]
        else:
            mit_value = None

        exact_values.append(exact_value)
        noisy_values.append(noisy_value)
        mit_values.append(mit_value)

    # Convert lists to arrays.
    exact_values = np.array(exact_values)
    noisy_values = np.array(noisy_values)
    mit_values = np.array(mit_values)

    # Saving results into arrays.
    np.save(os.path.join(folder_path, "exact_values.npy"), exact_values)
    np.save(os.path.join(folder_path, "noisy_values.npy"), noisy_values)
    np.save(os.path.join(folder_path, "mit_values.npy"), mit_values)

    abs_dist_noisy_exact = np.abs(exact_values - noisy_values)
    abs_dist_mit_exact = np.abs(exact_values - mit_values)

    # Prepare the final output.
    out_results.update(
        {
            "mitigation_output": mitigation_output,
            "median_abs_dist_exact_noisy": float(np.median(abs_dist_noisy_exact)),
            "mad_abs_dist_exact_noisy": float(median_abs_deviation(abs_dist_noisy_exact)),
            "median_abs_dist_exact_mit": float(np.median(abs_dist_mit_exact)),
            "mad_abs_dist_exact_mit": float(median_abs_deviation(abs_dist_mit_exact)),
        }
    )

    # Dump output into a JSON file using the custom encoder.
    results_file = os.path.join(folder_path, "results.json")
    with open(results_file, "w") as f:
        json.dump(out_results, f, indent=4, cls=NumpyEncoder)

    click.echo(f"Results dumped to: {results_file}")
    click.echo("Mitigation output:")
    click.echo(json.dumps(mitigation_output, indent=4, cls=NumpyEncoder))

if __name__ == '__main__':
    main()
