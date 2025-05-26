#!/usr/bin/env bash
#SBATCH --job-name=tncdr
#SBATCH --output=tncdr_%A_%a.out   
#SBATCH --array=0-53

# --- parameter grids ---
nqubits_list=(3 5 7 9 11 13)
nlayers_list=(2 4 6)
ncircuits_list=(10 30 100)

# total counts
NQ=${#nqubits_list[@]}      # 6
NL=${#nlayers_list[@]}      # 3
NC=${#ncircuits_list[@]}    # 3

# map SLURM_ARRAY_TASK_ID -> (i,j,k)
idx=$SLURM_ARRAY_TASK_ID
i=$(( idx / (NL*NC) ))                  # which nqubits
rem=$(( idx % (NL*NC) ))
j=$(( rem / NC ))                       # which nlayers
k=$(( rem % NC ))                       # which ncircuits

nqubits=${nqubits_list[i]}
nlayers=${nlayers_list[j]}
ncircuits=${ncircuits_list[k]}

echo "Running task $idx: nqubits=$nqubits nlayers=$nlayers ncircuits=$ncircuits"

# run your Python command
python floquet.py \
  --nqubits "$nqubits" \
  --nlayers "$nlayers" \
  --b 1.2566370614359172 \
  --theta 1.5707963267948966 \
  --replacement-probability 0.5 \
  --ncircuits "$ncircuits" \
  --random-seed 42 \
  --local-pauli-noise-sigma 0.001\
  --max-bond-dimension 128 \
  --nruns 10 \
  --plot true \
  --save-results true
