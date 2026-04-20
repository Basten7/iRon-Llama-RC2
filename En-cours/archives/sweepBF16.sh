#!/usr/bin/env bash
set -euo pipefail

export GGML_METAL_DEVICE_INDEX=3,4
export GGML_METAL_FA_VEC=1

export MODEL_BF16_06B=~/Models/Qwen3-0.6B-BF16.gguf
export MODEL_BF16_17B=~/Models/Qwen3-1.7B-BF16.gguf
export MODEL_BF16_4B=~/Models/Qwen3-4B-128K-BF16.gguf
export MODEL_BF16_14B=~/Models/Qwen3-14B-BF16.gguf

MODELS=(
  "$MODEL_BF16_06B"
  "$MODEL_BF16_17B"
  "$MODEL_BF16_4B"
  "$MODEL_BF16_14B"
)

PP_NR0_LIST=(4 8)
PP_NSG_LIST=(2 4)

LOGDIR=./logs_bf16_pp_sweep
mkdir -p "$LOGDIR"

unset GGML_METAL_DECODE_BF16_NR0
unset GGML_METAL_DECODE_BF16_NSG
unset GGML_METAL_DECODE_BF16_SHAPE_THRESHOLD

for model in "${MODELS[@]}"; do
  mname="$(basename "$model" .gguf)"
  for pp_nr0 in "${PP_NR0_LIST[@]}"; do
    for pp_nsg in "${PP_NSG_LIST[@]}"; do
      export GGML_METAL_PP_BF16_NR0="$pp_nr0"
      export GGML_METAL_PP_BF16_NSG="$pp_nsg"

      logfile="$LOGDIR/${mname}.pp_nr0_${pp_nr0}.pp_nsg_${pp_nsg}.log"

      echo "=== MODEL=$mname PP_NR0=$pp_nr0 PP_NSG=$pp_nsg ==="
      ./build/bin/llama-bench -ngl 99 -fa 1 -ub 32 -b 512 -p 512 -n 128 --mmap 0 -m "$model" \
        2>&1 | tee "$logfile"
    done
  done
done
