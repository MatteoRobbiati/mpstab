#!/usr/bin/env bash
#SBATCH --job-name=bench_tn
#SBATCH --output=logs/job_%a.out
#SBATCH --error=logs/job_%a.err
#SBATCH --array=0-289

set -euo pipefail

PYTHON=python
SCRIPT=run_configuration.py

# --- Parameters ---
NLAYERS=3
NRUNS=10
P=0
BACKENDS=("qibotn" "mpstab")

# --- Task Mapping Logic ---
TASKS=()
for N in $(seq 5 5 100); do
  LIMIT=$(( N / 2 ))
  [ "$LIMIT" -gt 10 ] && LIMIT=10
  
  # k va da 2 a LIMIT (esponente)
  for ((k=2; k<=LIMIT; k++)); do
    BD=$(( 2**k )) # Calcolo 2^k
    for BE in "${BACKENDS[@]}"; do
      TASKS+=("$N:$BD:$BE")
    done
  done
done

CURRENT_TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
NQUBITS=$(echo $CURRENT_TASK | cut -d: -f1)
BD=$(echo $CURRENT_TASK | cut -d: -f2)
BACKEND=$(echo $CURRENT_TASK | cut -d: -f3)

PLATFORM="None"
[ "$BACKEND" == "qibotn" ] && PLATFORM="quimb"

# --- Execution ---
${PYTHON} ${SCRIPT} \
  --backend "${BACKEND}" \
  --platform "${PLATFORM}" \
  --max-bond-dim "${BD}" \
  --replacement-probability "${P}" \
  --nqubits "${NQUBITS}" \
  --nlayers "${NLAYERS}" \
  --nruns "${NRUNS}"