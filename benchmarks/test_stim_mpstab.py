import time
import csv
import os
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

def run_benchmark_cycle(
    nqubits: int,
    nlayers: int,
    seed: int,
    circuit_type: str,
    nruns: int = 10,
    max_bond_dim: int = 16
):
    """Esegue il benchmark per un numero specifico di qubit e calcola la mediana."""
    
    times_mpstab = []
    times_stim = []
    results_mpstab = []
    results_stim = []

    # Inizializziamo i backend una volta per questa configurazione
    bk_mpstab = initialize_backend("mpstab", platform=None, max_bond_dim=max_bond_dim)
    bk_stim = initialize_backend("stim", platform=None, max_bond_dim=None)

    for run in range(nruns):
        # Cambiamo seed ogni run per variabilità, ma manteniamo lo stesso per entrambi i backend
        current_seed = seed + run
        np.random.seed(current_seed)
        random.seed(current_seed)

        # 1. Preparazione Stato Iniziale
        initial_state = Circuit(nqubits)
        if circuit_type == "random_clifford":
            for q in range(nqubits):
                angle = np.random.randint(0, 4) * (np.pi / 2)
                initial_state.add(gates.RY(q, angle))
        else:
            initial_state = None

        # 2. Generazione Circuito
        if circuit_type == "ghz":
            circuit = generate_ghz_circuit(nqubits)
        else:
            circuit = generate_partitionated_circuit(
                nqubits=nqubits,
                nlayers=nlayers,
                replacement_probability=1, 
            )
            params = get_clifford_random_params(len(circuit.get_parameters()), current_seed)
            circuit.set_parameters(params)

        # 3. Definizione Osservabile
        if circuit_type == "ghz":
            obs_str = "Z" * nqubits
        else:
            obs_str = "ZX" * (nqubits // 2) + ("Z" if nqubits % 2 else "")

        # --- Esecuzione MPSTAB ---
        res_m, t_m, _ = execute_benchmark_circuit(
            circuit=circuit, observable=obs_str, backend="mpstab",
            max_bond_dim=max_bond_dim, initial_state=initial_state,
            replacement_probability=1.0, backend_obj=bk_mpstab
        )
        times_mpstab.append(t_m)
        results_mpstab.append(res_m)

        # --- Esecuzione STIM ---
        res_s, t_s, _ = execute_benchmark_circuit(
            circuit=circuit, observable=obs_str, backend="stim",
            max_bond_dim=None, initial_state=initial_state,
            replacement_probability=1.0, backend_obj=bk_stim
        )
        times_stim.append(t_s)
        results_stim.append(res_s)

        # Verifica correttezza (solo al primo run per risparmiare log)
        if run == 0:
            if not np.isclose(res_m, res_s, atol=1e-8):
                print(f"⚠️ [WARNING] Mismatch trovato per {circuit_type} (N={nqubits})!")
                print(f"   MPSTAB: {res_m:.6f}, STIM: {res_s:.6f}")

    return {
        "nqubits": nqubits,
        "circuit_type": circuit_type,
        "median_mpstab": np.median(times_mpstab),
        "median_stim": np.median(times_stim)
    }

if __name__ == "__main__":
    # Parametri globali
    nlayers = 3
    nruns = 10
    base_seed = 42
    output_file = "benchmark_results.csv"
    qubit_range = range(0, 1000, 50)
    circuit_types = ["random_clifford", "ghz"]

    # Preparazione file CSV
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["circuit_type", "nqubits", "backend", "median_time_seconds"])

        for c_type in circuit_types:
            print(f"\n🚀 Inizio benchmark per circuito: {c_type}")
            
            for n in qubit_range:
                print(f"Processing N={n}...", end="\r")
                
                stats = run_benchmark_cycle(
                    nqubits=n, 
                    nlayers=nlayers, 
                    seed=base_seed, 
                    circuit_type=c_type, 
                    nruns=nruns
                )

                # Salviamo i dati per entrambi i backend
                writer.writerow([c_type, n, "mpstab", f"{stats['median_mpstab']:.6f}"])
                writer.writerow([c_type, n, "stim", f"{stats['median_stim']:.6f}"])
                
                # Svuota il buffer per non perdere dati in caso di crash
                f.flush()

    print(f"\n✅ Benchmark completato. Risultati salvati in: {output_file}")
