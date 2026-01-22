#!/usr/bin/env bash
set -euo pipefail

BOND_DIMS=(4 8 16 32 64)

for BD in "${BOND_DIMS[@]}"; do
  echo "========================================"
  echo "Running qibotn | max bond dimension = ${BD}"
  echo "========================================"

  python run_configuration.py \
    --backend qibotn \
    --platform quimb \
    --max-bond-dim "${BD}" \
    --replacement-probability 0.75 \
    --nqubits 12 \
    --nlayers 4 \
    --nruns 20

  echo "========================================"
  echo "Running mpstab | max bond dimension = ${BD}"
  echo "========================================"

  python run_configuration.py \
    --backend mpstab \
    --platform None \
    --max-bond-dim "${BD}" \
    --replacement-probability 0.75 \
    --nqubits 12 \
    --nlayers 4 \
    --nruns 20
done
