export GGML_METAL_DEVICE_INDEX=3,4
export GGML_METAL_FA_VEC=1
MODEL="/Volumes/NM790-II/Qwen3-4B-F32.gguf"

for pp_nr0 in 2 4 8; do
  for pp_nsg in 1 2 4 8; do
    export GGML_METAL_PP_F32_NR0=$pp_nr0
    export GGML_METAL_PP_F32_NSG=$pp_nsg
    echo "===== F32 PP NR0=$pp_nr0 NSG=$pp_nsg ====="
    ./build/bin/llama-cli -st -p 'expliques-moi Tensorflow' -ngl 99 --no-mmap --jinja --ctx-size 80144 -fa 1 -ub 32 -b 512 -n 512 -m "$MODEL"
  done
done

for dec_nr0 in 2 4 6 8; do
  for dec_nsg in 1 2 4; do
    export GGML_METAL_DECODE_F32_NR0=$dec_nr0
    export GGML_METAL_DECODE_F32_NSG=$dec_nsg
    export GGML_METAL_DECODE_F32_SHAPE_THRESHOLD=2048
    echo "===== F32 TG NR0=$dec_nr0 NSG=$dec_nsg ====="
    ./build/bin/llama-cli -st -p 'expliques-moi Tensorflow' -ngl 99 --no-mmap --jinja --ctx-size 80144 -fa 1 -ub 32 -b 512 -n 512 -m "$MODEL"
  done
done
