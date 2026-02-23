#!/usr/bin/env bash

set -euo pipefail

PYTHON=python
SCRIPT=run_configuration.py

# --- Parameters ---
NLAYERS=7
NRUNS=10
P=0.8
# BACKEND=mpstab
# PLATFORM=quimb,stim
BACKEND=quimb
PLATFORM=numpy

NQUBITS=40

# # --- Task Mapping Logic ---
# TASKS=()
# for N in $(seq 5 5 20); do
#   LIMIT=$(( N / 2 ))
#   [ "$LIMIT" -gt 10 ] && LIMIT=10
  
#   for ((k=2; k<=LIMIT; k++)); do
#     BD=$(( 2**k ))
#     for BE in "${BACKENDS[@]}"; do
#       TASKS+=("$N:$BD:$BE")
#     done
#   done
# done

# # --- Execution Loop ---
# for CURRENT_TASK in "${TASKS[@]}"; do
#     NQUBITS=$(echo $CURRENT_TASK | cut -d: -f1)
#     BD=$(echo $CURRENT_TASK | cut -d: -f2)
#     BACKEND=$(echo $CURRENT_TASK | cut -d: -f3)

    # --- Platform Logic ---
    # if [ "$BACKEND" == "mpstab" ]; then
    #     PLATFORM="quimb,stim"
    # elif [ "$BACKEND" == "quimb" ]; then
    #     PLATFORM="numpy"
    # elif [ "$BACKEND" == "qibo" ]; then
    #     PLATFORM="numpy"
    # fi

    echo "Running: Backend=$BACKEND, Platform=$PLATFORM, NQ=$NQUBITS" #, BD=$BD"

    ${PYTHON} ${SCRIPT} \
      --backend "${BACKEND}" \
      --platform "${PLATFORM}" \
      --replacement-probability "${P}" \
      --nqubits "${NQUBITS}" \
      --nlayers "${NLAYERS}" \
      --nruns "${NRUNS}"