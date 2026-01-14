#!/usr/bin/env bash

# Path to your Python script
SCRIPT="run_configuration.py"

# Fixed parameters (edit as needed)
REPLACEMENT_PROB=0.75
NQUBITS=20
NLAYERS=4
NRUNS=10
RNG_SEED=42
SET_INITIAL_STATE=true

# Values to loop over
MAX_BOND_DIMS=(2 4 16 32 64 128 256)

for MAX_BOND_DIM in "${MAX_BOND_DIMS[@]}"; do
  echo "Running with max_bond_dim=${MAX_BOND_DIM}"

  python "${SCRIPT}" \
    --max-bond-dim "${MAX_BOND_DIM}" \
    --replacement-probability "${REPLACEMENT_PROB}" \
    --nqubits "${NQUBITS}" \
    --nlayers "${NLAYERS}" \
    --nruns "${NRUNS}" \
    --rng_seed "${RNG_SEED}" \
    --set_initial_state "${SET_INITIAL_STATE}"
done