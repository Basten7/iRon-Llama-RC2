export GGML_METAL_DEVICE_INDEX=1,2
export GGML_METAL_FA_VEC=1
MODEL="/Volumes/NM790-II/Qwen3-4B-Q5_0.gguf"

export GGML_METAL_DECODE_F32_NR0=4
export GGML_METAL_DECODE_F32_NSG=4
export GGML_METAL_PP_F32_NR0=4
export GGML_METAL_PP_F32_NSG=4
export GGML_METAL_DECODE_F32_SHAPE_THRESHOLD=2048
export GGML_METAL_DECODE_BF16_NR0=4
export GGML_METAL_DECODE_BF16_NSG=4
export GGML_METAL_PP_BF16_NR0=4
export GGML_METAL_PP_BF16_NSG=4
export GGML_METAL_DECODE_BF16_SHAPE_THRESHOLD=2048
export GGML_METAL_PP_F16_NR0=4
export GGML_METAL_PP_F16_NSG=4
export GGML_METAL_DECODE_F16_NR0=4
export GGML_METAL_DECODE_F16_NSG=4
export GGML_METAL_DECODE_F16_SHAPE_THRESHOLD=2048
export GGML_METAL_DECODE_Q50_SHAPE_THRESHOLD=2048
export GGML_METAL_PP_Q50_NR0=8
export GGML_METAL_PP_Q50_NSG=1
export GGML_METAL_DECODE_Q50_NR0=2
export GGML_METAL_DECODE_Q50_NSG=2

# for pp_nr0 in 2 4 6 8; do
#   for pp_nsg in 1 2 4 8; do
#     export GGML_METAL_PP_Q50_NR0=$pp_nr0
#     export GGML_METAL_PP_Q50_NSG=$pp_nsg
#     echo "===== Q5_0 PP NR0=$pp_nr0 NSG=$pp_nsg ====="
#     ./build/bin/llama-bench -ngl 99 -fa 1 -ub 32 -b 512 -p 256 -n 0 --mmap 0 -m "$MODEL"
#     #./build/bin/llama-cli -st -p 'expliques-moi Tensorflow' -ngl 99 --no-mmap --jinja --ctx-size 80144 -fa 1 -ub 32 -b 512 -n 512 -m "$MODEL"
#   done
# done

for dec_nr0 in 2 4 6 8; do
  for dec_nsg in 1 2 4; do
    export GGML_METAL_DECODE_Q50_NR0=$dec_nr0
    export GGML_METAL_DECODE_Q50_NSG=$dec_nsg
    export GGML_METAL_DECODE_Q50_SHAPE_THRESHOLD=2048
    echo "===== Q5_0 TG NR0=$dec_nr0 NSG=$dec_nsg ====="
    ./build/bin/llama-bench -ngl 99 -fa 1 -ub 32 -b 512 -p 0 -n 256 --mmap 0 -m "$MODEL"
    #./build/bin/llama-cli -st -p 'expliques-moi Tensorflow' -ngl 99 --no-mmap --jinja --ctx-size 80144 -fa 1 -ub 32 -b 512 -n 512 -m "$MODEL"
  done
done

# for dec_nr0 in 2 4 6 8; do
#   for dec_nsg in 1 2 4; do
#     export GGML_METAL_DECODE_Q50_ID_NR0=$dec_nr0
#     export GGML_METAL_DECODE_Q50_ID_NSG=$dec_nsgQ50
#     echo "===== Q6_K_M TG DECODE_Q50_ID_NR0=$dec_nr0 DECODE_Q50_ID_NSG=$dec_nsg ====="
#     ./build/bin/llama-bench -ngl 99 -fa 1 -ub 32 -b 512 -p 0 -n 256 --mmap 0 -m "$MODEL"
#     #./build/bin/llama-cli -st -p 'expliques-moi Tensorflow' -ngl 99 --no-mmap --jinja --ctx-size 80144 -fa 1 -ub 32 -b 512 -n 512 -m "$MODEL"
#   done
# done
