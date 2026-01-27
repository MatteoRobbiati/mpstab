#!/usr/bin/env bash

# Fallisce se un comando fallisce o se una variabile è non definita
set -euo pipefail

PYTHON=python
SCRIPT=run_configuration.py

# --- Parametri dell'esperimento ---
NLAYERS=3
NRUNS=10
P=1.0          # Replacement probability 1.0 per avere solo circuiti Clifford
BOND_DIM=100   # Bond dimension per mpstab
BACKENDS=("stim" "mpstab")

# --- Generazione Lista Qubit ---
# Genera la sequenza: 20, 40, 60, ..., 1000
NQUBITS_LIST=($(seq 20 20 1000))

echo "Inizio Benchmark locale: da 20 a 1000 qubit (passo 20)"
echo "Backend: ${BACKENDS[*]}"
echo "--------------------------------------------------------"

# --- Loop di Esecuzione ---
for N in "${NQUBITS_LIST[@]}"; do
  for BE in "${BACKENDS[@]}"; do
    
    echo "[$(date +%T)] Esecuzione: N=$N, Backend=$BE..."

    # Esecuzione del comando python
    ${PYTHON} ${SCRIPT} \
      --backend "${BE}" \
      --replacement-probability "${P}" \
      --nqubits "${N}" \
      --nlayers "${NLAYERS}" \
      --max-bond-dim "${BOND_DIM}" \
      --nruns "${NRUNS}" \
      --set-initial-state "False"
    
  done
  echo "--------------------------------------------------------"
done

echo "Benchmark completato."