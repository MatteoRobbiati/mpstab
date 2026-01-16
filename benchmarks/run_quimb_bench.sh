#!/usr/bin/env bash

# Path to your Python script
SCRIPT="quimb_scripts.py"

# Fixed parameters (edit as needed)
REPLACEMENT_PROB=0.75
NQUBITS=12
NLAYERS=5
NRUNS=10
RNG_SEED=42
SET_INITIAL_STATE=true
BACKEND="both"
USE_TRANSPILED=false
REPLACEMENT_METHOD="closest"

# Values to loop over
MAX_BOND_DIMS=(2 4 8 16 32 64)

for MAX_BOND_DIM in "${MAX_BOND_DIMS[@]}"; do
  echo "Running with max_bond_dim=${MAX_BOND_DIM}"

  python3 "${SCRIPT}" \
    --max-bond-dim "${MAX_BOND_DIM}" \
    --replacement-probability "${REPLACEMENT_PROB}" \
    --nqubits "${NQUBITS}" \
    --nlayers "${NLAYERS}" \
    --nruns "${NRUNS}" \
    --rng-seed "${RNG_SEED}" \
    --set-initial-state "${SET_INITIAL_STATE}" \
    --backend "${BACKEND}"
done
