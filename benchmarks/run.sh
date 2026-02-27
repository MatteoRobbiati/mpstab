#!/usr/bin/env bash
#SBATCH --job-name=quantum_sweep
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

PYTHON=python
SCRIPT=run_configuration.py

# --- Liste dei parametri (Modificabili) ---
QUBITS_LIST=(20 30 40)
LAYERS_LIST=(5 10)
PROB_LIST=(0.5 0.8)
BD_LIST=(32 64 128)
NRUNS=10

# --- Generazione della matrice delle configurazioni ---
# Creiamo un array temporaneo con tutte le combinazioni possibili
CONFIGS=()
for NQ in "${QUBITS_LIST[@]}"; do
    for L in "${LAYERS_LIST[@]}"; do
        for P in "${PROB_LIST[@]}"; do
            for BD in "${BD_LIST[@]}"; do
                # Aggiungiamo i due backend per ogni combinazione
                CONFIGS+=("$NQ:$L:$P:$BD:mpstab:quimb,stim")
                CONFIGS+=("$NQ:$L:$P:$BD:quimb:numpy")
            done
        done
    done
done

# --- Gestione SLURM Array ---
# Calcoliamo il numero totale di task (partendo da 0)
TOTAL_TASKS=${#CONFIGS[@]}
MAX_INDEX=$((TOTAL_TASKS - 1))

# Se lo script è lanciato senza sbatch, stampa il comando per lanciarlo correttamente
if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
    echo "Per eseguire questo sweep di $TOTAL_TASKS job, usa:"
    echo "sbatch --array=0-$MAX_INDEX $0"
    exit 0
fi

# Estraiamo i parametri per il task corrente
CURRENT_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

NQ=$(echo $CURRENT_CONFIG | cut -d: -f1)
L=$(echo $CURRENT_CONFIG | cut -d: -f2)
P=$(echo $CURRENT_CONFIG | cut -d: -f3)
BD=$(echo $CURRENT_CONFIG | cut -d: -f4)
BACKEND=$(echo $CURRENT_CONFIG | cut -d: -f5)
PLATFORM=$(echo $CURRENT_CONFIG | cut -d: -f6)

echo "Job $SLURM_ARRAY_TASK_ID/$MAX_INDEX: NQ=$NQ, L=$L, P=$P, BD=$BD, BE=$BACKEND"

# --- Esecuzione ---
${PYTHON} ${SCRIPT} \
  --backend "${BACKEND}" \
  --platform "${PLATFORM}" \
  --replacement-probability "${P}" \
  --nqubits "${NQ}" \
  --nlayers "${L}" \
  --nruns "${NRUNS}" \
  --max-bond-dimension "${BD}"