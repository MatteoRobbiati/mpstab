import json
import os
import time
import random
from typing import Any, Dict, List

import numpy as np
import scipy.stats as sp

from qibo import Circuit, gates

from mpstab.evolutors.models import HybridSurrogate
from mpstab.targets.ansatze import HardwareEfficient

def run_experiment(
    max_bond_dim: int | None,
    replacement_probability: float,
    nqubits: int,
    nlayers: int,
    nruns: int = 10,
    rng_seed: int = 42,
    set_initial_state: bool = True,
) -> None:
    
    input_arguments = dict(locals())

    random.seed(rng_seed)
    np.random.seed(rng_seed)

    obs_str = "ZX" * int(nqubits/2)
    if nqubits%2 != 0:
        obs_str += "Z"

    base_folder = (
        f"results/hdw_efficient_ansatz_"
        f"{nqubits}qubits_{nlayers}layers"
    )
    run_folder = (
        f"{max_bond_dim}max_bd_"
        f"{replacement_probability}repl_prob"
    )
    output_dir = os.path.join(base_folder, run_folder)
    os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, List[Any]] = {
        "measured_observable": obs_str,
        "times": [],
        "expvals": [],
        "magic_gates": [],
    }

    for _ in range(nruns):

        if set_initial_state:
            initial_state = Circuit(nqubits)
            for q in range(nqubits):
                initial_state.add(gates.RY(q, theta=np.random.uniform(-np.pi, np.pi)))

        ansatz = HardwareEfficient(
            nqubits=nqubits,
            nlayers=nlayers,
        )

        evolutor = HybridSurrogate(
            ansatz=ansatz,
            max_bond_dimension=max_bond_dim,
            initial_state=initial_state,
        )

        start_time = time.time()
        expval, partitions = evolutor.expectation_from_partition(
            observable=obs_str,
            replacement_probability=replacement_probability,
            return_partitions=True,
        )
        elapsed_time = time.time() - start_time

        results["magic_gates"].append(len(partitions["magic_gates"]))
        results["times"].append(elapsed_time)
        results["expvals"].append(expval)

    results.update({"median_time": np.median(results["times"])})
    results.update({"mad_time": sp.median_abs_deviation(results["times"])})
    results.update({"ave_magic_gates": np.mean(results["magic_gates"])})

    results.update({"input_arguments": input_arguments})

    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
