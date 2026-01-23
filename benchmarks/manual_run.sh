#!/usr/bin/env bash

NQUBITS=5
EXP=2          # Esponente per la bond dimension (2^4 = 16)
BACKEND_ID=1   # 1 per qibotn, 2 per mpstab
NLAYERS=3
NRUNS=1
# ==========================================

# Calcolo dinamico
BD=$(( 2**EXP ))

if [ "$BACKEND_ID" -eq 1 ]; then
    BACKEND="qibotn"
    PLATFORM="quimb"
else
    BACKEND="mpstab"
    PLATFORM="None"
fi

# Riepilogo Parametri
echo "===================================================="
echo "   PREPARAZIONE SIMULAZIONE"
echo "===================================================="
echo "  Backend:      $BACKEND"
echo "  Platform:     $PLATFORM"
echo "  Qubits:       $NQUBITS"
echo "  Bond Dim:     $BD (2^$EXP)"
echo "  Layers:       $NLAYERS"
echo "  Runs:         $NRUNS"
echo "  Repl. Prob:   0"
echo "===================================================="
echo "Avvio in corso..."
echo ""

# Esecuzione
python run_configuration.py \
  --backend "$BACKEND" \
  --platform "$PLATFORM" \
  --max-bond-dim "$BD" \
  --replacement-probability 0 \
  --nqubits "$NQUBITS" \
  --nlayers "$NLAYERS" \
  --nruns "$NRUNS"