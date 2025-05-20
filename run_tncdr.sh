#!/bin/bash

python mitigation_run.py \
  --nqubits 3 \
  --nlayers 3 \
  --nshots 1000 \
  --ncircuits 30 \
  --ansatz HardwareEfficient \
  --observable IIZ \
  --readout_bitflip_probability 0.001 \
  --local_pauli_noise_sigma 0.001 \
  --mitigation_method TNCDR \
  --mitigation_args '{"replacement_probability": 0.5, "max_bond_dimension": 128}' \
  --random_seed 42 \
  --nruns 20 \
  --plot true
