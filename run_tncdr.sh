#!/bin/bash

python mitigation_run.py \
  --nqubits 5 \
  --nlayers 3 \
  --nshots 2000 \
  --ncircuits 20 \
  --ansatz HardwareEfficient \
  --observable ZZZZZZZ \
  --readout_bitflip_probability 0.005 \
  --local_pauli_noise_sigma 0.005 \
  --mitigation_method TNCDR \
  --mitigation_args '{"npartitions": 3, "magic_gates_per_partition": 1, "max_bond_dimension": null}' \
  --random_seed 42 \
  --nruns 10

