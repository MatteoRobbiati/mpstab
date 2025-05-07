#!/bin/bash

python mitigation_run.py \
  --nqubits 4 \
  --nlayers 4 \
  --nshots 10000 \
  --ncircuits 20 \
  --ansatz HardwareEfficient \
  --observable IIIZ \
  --readout_bitflip_probability 0.0001 \
  --local_pauli_noise_sigma 0.0001 \
  --mitigation_method TNCDR \
  --mitigation_args '{"replacement_probability": 0.7, "max_bond_dimension": null}' \
  --random_seed 42 \
  --nruns 10 \
  --plot true

