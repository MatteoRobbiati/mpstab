import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder

from tncdr.targets.ansatze import FloquetAnsatz


def qibo_to_qiskit(qibocirc):
    """Convert a Qibo circuit into a Qiskit QuantumCircuit."""
    qasm = qibocirc.to_qasm()
    return qiskit.QuantumCircuit.from_qasm_str(qasm)


def main():
    parser = argparse.ArgumentParser(
        description="Run TNCDR on a set of circuits located in a given path"
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        help="Path to a folder containing circuits dataset: results.json, original_circuit_params.npy, params_circuit*.npy",
    )
    args = parser.parse_args()

    base_path = Path(args.experiment_path)
    results_dir = base_path / "results"
    results_dir.mkdir(exist_ok=True)

    # load config
    with open(base_path / "results.json") as f:
        config = json.load(f)

    # prepare ansatz and original circuit
    ansatz = FloquetAnsatz(
        nqubits=config["nqubits"],
        nlayers=config["nlayers"],
        b=config["b"],
        theta=config["theta"],
    )
    original_qibo_circ = ansatz.circuit
    original_params = np.load(base_path / "circuits" / "original_circuit_params.npy")
    original_qibo_circ.set_parameters(original_params)
    original_qiskit_circ = qibo_to_qiskit(original_qibo_circ)

    # detect training parameter files
    param_files = sorted((base_path / "circuits").glob("params_circuit*.npy"))
    n_training = len(param_files)
    if n_training == 0:
        raise RuntimeError("No training parameter files found in " + str(base_path))

    # load training circuits
    training_circuits = []
    for p in param_files:
        params = np.load(p)
        qc = deepcopy(original_qibo_circ)
        qc.set_parameters(params)
        training_circuits.append(qibo_to_qiskit(qc))

    # assemble all circuits: original first
    all_circuits = [original_qiskit_circ] + training_circuits

    print(all_circuits)

    # IBM runtime setup
    service = QiskitRuntimeService(
        instance="cern/internal/tncdr", channel="ibm_quantum"
    )
    backend = service.least_busy(operational=True)

    # observable
    q = config["nqubits"] // 2
    obs_label = "I" * q + "X" + "I" * (config["nqubits"] - q - 1)
    observables = [SparsePauliOp.from_list([(obs_label, 1.0)])] * len(all_circuits)

    # transpile to ISA and build pubs
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    pubs = []
    for circ, obs in zip(all_circuits, observables):
        isa_circ = pm.run(circ)
        isa_obs = obs.apply_layout(isa_circ.layout)
        pubs.append((isa_circ, isa_obs))

    # run estimator (TNCDR) on batch
    estimator = Estimator(backend)
    t0 = time.time()
    job = estimator.run(pubs)
    tn_job_id = job.job_id()
    tn_elapsed = time.time() - t0
    print(f"TNCDR job id: {tn_job_id}")
    print(f"TNCDR elapsed: {tn_elapsed:.2f}s")
    result_tncdr = job.result()

    # save TNCDR results
    with open(results_dir / "tncdr_results.json", "w") as f:
        json.dump(result_tncdr, f, cls=RuntimeEncoder)

    # collect metadata
    metadata = {
        "backend": backend.name,
        "total_circuits": len(all_circuits),
        "training_circuits": n_training,
        "tncdr_job_id": tn_job_id,
        "tncdr_elapsed": tn_elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # save metadata
    with open(results_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {results_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
