#!/usr/bin/env bash
#SBATCH --job-name=tncdr
#SBATCH --output=tncdr_test.out

# run your Python command
python floquet.py \
  --nqubits 7 \
  --nlayers 2 \
  --b 1.2566370614359172 \
  --theta 1.5707963267948966 \
  --replacement-probability 0.5 \
  --ncircuits 10 \
  --random-seed 42 \
  --local-pauli-noise-sigma 0.0000001\
  --max-bond-dimension 128 \
  --nruns 10 \
  --plot true \
  --save-results true
