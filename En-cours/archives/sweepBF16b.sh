#!/usr/bin/env bash
set -euo pipefail

export GGML_METAL_DEVICE_INDEX=4,6
export GGML_METAL_FA_VEC=1

# A ADAPTER après la phase 1
export GGML_METAL_PP_BF16_NR0=8
export GGML_METAL_PP_BF16_NSG=4

export MODEL_BF16_4B=/Volumes/NM790-II/Qwen3-4B-BF16.gguf


MODELS=(
  "$MODEL_BF16_4B"
)

DEC_NR0_LIST=(2 4 8)
DEC_NSG_LIST=(2 4)
THRESH_LIST=(1024 2048 4096)

LOGDIR=./logs_bf16_decode_sweep
mkdir -p "$LOGDIR"

for model in "${MODELS[@]}"; do
  mname="$(basename "$model" .gguf)"
  for dec_nr0 in "${DEC_NR0_LIST[@]}"; do
    for dec_nsg in "${DEC_NSG_LIST[@]}"; do
      for thr in "${THRESH_LIST[@]}"; do
        export GGML_METAL_DECODE_BF16_NR0="$dec_nr0"
        export GGML_METAL_DECODE_BF16_NSG="$dec_nsg"
        export GGML_METAL_DECODE_BF16_SHAPE_THRESHOLD="$thr"

        logfile="$LOGDIR/${mname}.dec_nr0_${dec_nr0}.dec_nsg_${dec_nsg}.thr_${thr}.log"

        echo "=== MODEL=$mname DEC_NR0=$dec_nr0 DEC_NSG=$dec_nsg THR=$thr ==="
        ./build/bin/llama-bench -ngl 99 -fa 1 -ub 32 -b 512 -p 512 -n 128 --mmap 0 -m "$model" \
          2>&1 | tee "$logfile"
      done
    done
  done
done
