#!/bin/bash

python mitigation_run.py \
  --nqubits 5 \
  --nlayers 1 \
  --nshots 2000 \
  --ncircuits 20 \
  --ansatz HardwareEfficient \
  --observable ZZZZZ \
  --readout_bitflip_probability 0.001 \
  --local_pauli_noise_sigma 0.001 \
  --mitigation_method TNCDR \
  --mitigation_args '{"replacement_probability": 0.25, "max_bond_dimension": null}' \
  --random_seed 42 \
  --nruns 10 \
  --plot true

