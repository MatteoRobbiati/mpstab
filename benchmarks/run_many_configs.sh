#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=run_configuration.py

# -------------------------------
# Fixed parameters
# -------------------------------
NLAYERS=3
NRUNS=10

# Sweep parameters
NQUBITS_LIST=(4 6 8 10 12)
REPLACEMENT_PROBS=(0 0.25 0.5 0.75 0.99)

# -------------------------------
# Main sweep
# -------------------------------
for NQUBITS in "${NQUBITS_LIST[@]}"; do

  # Max bond dimension = 2^(nqubits/2)
  MAX_EXP=$(( NQUBITS / 2 ))

  # Generate bond dimensions: 2^1, 2^2, ..., 2^(nqubits/2)
  BOND_DIMS=()
  for ((e=1; e<=MAX_EXP; e++)); do
    BOND_DIMS+=( $((2**e)) )
  done

  for P in "${REPLACEMENT_PROBS[@]}"; do
    for BD in "${BOND_DIMS[@]}"; do

      echo "===================================================="
      echo "Qubits            : ${NQUBITS}"
      echo "Bond dimension    : ${BD}"
      echo "Replacement prob  : ${P}"
      echo "===================================================="

      echo "--- Running qibotn ---"
      ${PYTHON} ${SCRIPT} \
        --backend qibotn \
        --platform quimb \
        --max-bond-dim "${BD}" \
        --replacement-probability "${P}" \
        --nqubits "${NQUBITS}" \
        --nlayers "${NLAYERS}" \
        --nruns "${NRUNS}"

      echo "--- Running mpstab ---"
      ${PYTHON} ${SCRIPT} \
        --backend mpstab \
        --platform None \
        --max-bond-dim "${BD}" \
        --replacement-probability "${P}" \
        --nqubits "${NQUBITS}" \
        --nlayers "${NLAYERS}" \
        --nruns "${NRUNS}"

    done
  done
done
