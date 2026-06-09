#!/usr/bin/env bash

# Usage:
#   bash run_simple.sh <settings.json> [extra_python_args...]

set -euo pipefail

# Parse arguments
SETTINGS="${1:?Usage: bash run_simple.sh <settings.json>}"

if [[ ! -f "$SETTINGS" ]]; then
    echo "ERROR: settings file not found: $SETTINGS" >&2
    exit 1
fi

# Read sweep parameters from JSON
if ! command -v jq &>/dev/null; then
    echo "ERROR: 'jq' is required but not found in PATH." >&2
    exit 1
fi
N_QUBITS_LIST=($(jq -r '.sweep.n_qubits[]' "$SETTINGS"))
DEPTH_LIST=($(jq -r '.sweep.depth[]' "$SETTINGS"))
ENGINE_LIST=($(jq -r '.execution.engines[]' "$SETTINGS"))
VERBOSE_OUTPUT=$(jq -r '.output.verbose // false' "$SETTINGS")

if [[ "$VERBOSE_OUTPUT" == "false" ]]; then
    export QIBO_LOG_LEVEL=3
fi


N_Q=${#N_QUBITS_LIST[@]}
N_D=${#DEPTH_LIST[@]}
N_E=${#ENGINE_LIST[@]}
TOTAL=$(( N_Q * N_D * N_E ))
if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: sweep produces zero tasks.  Check sweep.n_qubits, sweep.depth, and execution.engines in $SETTINGS." >&2
    exit 1
fi
echo "Sweep dimensions: n_qubits=${N_Q} × depth=${N_D} × engines=${N_E} = ${TOTAL} tasks"

# Setup 

VENV_PATH="${VENV_PATH:-./.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    source "${VENV_PATH}/bin/activate"
fi

OUTPUT_DIR="$(jq -r '.output.filename | split("/")[:-1] | join("/")' "$SETTINGS")"
OUTPUT_FMT="$(jq -r '.output.format // "jsonl"' "$SETTINGS")"
OUTPUT_BASE="$(jq -r '.output.filename | split("/")[-1] | split(".")[0]' "$SETTINGS")"
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_BASE}.${OUTPUT_FMT}"
mkdir -p "${OUTPUT_DIR}"

# Run all tasks sequentially
TASK_ID=0
for (( e_idx=0; e_idx < N_E; e_idx++ )); do
    for (( d_idx=0; d_idx < N_D; d_idx++ )); do
        for (( q_idx=0; q_idx < N_Q; q_idx++ )); do
            N_QUBITS="${N_QUBITS_LIST[$q_idx]}"
            DEPTH="${DEPTH_LIST[$d_idx]}"
            ENGINE="${ENGINE_LIST[$e_idx]}"
            
            echo
            echo "Task ${TASK_ID}: n_qubits=${N_QUBITS}, depth=${DEPTH}, engine=${ENGINE}"
            
            python main.py \
                --settings  "${SETTINGS}"   \
                --n_qubits  "${N_QUBITS}"   \
                --depth     "${DEPTH}"      \
                --engine    "${ENGINE}"     \
                --run_id    "${TASK_ID}"    \
                --output    "${OUTPUT_FILE}" \
            
            echo "Task ${TASK_ID} finished successfully."
            TASK_ID=$(( TASK_ID + 1 ))
        done
    done
done

echo "All ${TOTAL} tasks completed."