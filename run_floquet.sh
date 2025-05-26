#!/usr/bin/env bash

# Run TNCDR with default hyperparameters
python floquet.py \
  --nqubits 5 \
  --nlayers 2 \
  --b 1.2566370614359172 \
  --theta 1.5707963267948966 \
  --replacement-probability 0.5 \
  --ncircuits 20 \
  --random-seed 42 \
  --local-pauli-noise-sigma 0.002 \
  --max-bond-dimension 128 \
  --nruns 5 \
  --plot true \
  --save-results true
