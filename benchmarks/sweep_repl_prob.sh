#!/usr/bin/env bash

SCRIPT="run_configuration.py"

# Fixed parameters
MAX_BOND_DIM=256
NQUBITS=15
NLAYERS=4
NRUNS=10
RNG_SEED=42
SET_INITIAL_STATE=true

# Replacement probability sweep
REPL_PROBS=(0.5 0.6 0.7 0.8 0.9 0.99)

for REPL_PROB in "${REPL_PROBS[@]}"; do
  echo "Running with replacement_probability=${REPL_PROB}"

  python "${SCRIPT}" \
    --max-bond-dim "${MAX_BOND_DIM}" \
    --replacement-probability "${REPL_PROB}" \
    --nqubits "${NQUBITS}" \
    --nlayers "${NLAYERS}" \
    --nruns "${NRUNS}" \
    --rng_seed "${RNG_SEED}" \
    --set_initial_state "${SET_INITIAL_STATE}"
done