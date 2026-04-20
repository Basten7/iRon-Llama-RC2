# iRon-Llama-RC2  
  
## 1. Base Commit  
- Upstream base: ggml-org/llama.cpp  
- Reference llama.cpp commit: 41ea26144 HEAD@{2026-01-31 18:46:12 +0100}: (last stable before regression)  
- Fork: iRon-Llama-RC2 (MGPU-focused branch)  
⸻  
## 2. Optimizations Implemented  
  
### Metal V3 / AMD Focus  
- Multi-GPU (MGPU) weight splitting: OK  
- KV cache splitting across GPUs: OK  
- Decode (TG) optimization priority over Prefill (PP)  
- Fine-grained offload policy tuning  
- Kernel tuning:  
    - NR0 / NSG tuning for Q4_K / Q6_K / F16 / BF16 / F32  
    - Decode-specific kernels (mul_mv / mul_mv_id)  
- Flash Attention EXT:  
    - Vectorized path tuning (FA_VEC)  
    - Threadgroup sizing (FA_NSG)  
- Shape threshold tuning:  
    - F16 / F32 thresholds  
    - BF16 integration  
  
### Scheduling / Runtime  
- MGPU stabilization (disable async → partial re-enable)  
- Copy path tracing instrumentation  
- Improved decode detection heuristics  
⸻  
## 3. Performance Results  
  
### Example (Metal, W6800X Duo)  
- Prefill: ~60–400 tokens/s (model dependent)  
- Decode: ~10–70 tokens/s  
  
### Observations  
- Decode is the main bottleneck  
- 2 GPUs often outperform 3–4 GPUs due to overhead  
- BF16 performs best in Prefill  
- Shape thresholds strongly impact F16/F32 performance  
⸻  
## 4. Comparison with Vulkan  
  
### Metal (macOS)  
- Better integration with AMD GPUs  
- Lower latency for small batches  
- Strong BF16/F16 paths  
  
### Vulkan (Linux / MoltenVK)  
- More stable multi-GPU scaling  
- Less optimized kernels for AMD vs Metal V3  
- No BF16 on many devices  
  
### Conclusion  
- Metal = best single-node performance on macOS  
- Vulkan = better portability and MGPU scaling consistency  
⸻  
## 5. Recommended Hardware  
  
### Optimal Setup  
- Mac Pro 2019 (Intel Xeon)  
- 2× to 4× Radeon PRO W6800X Duo  
- Optional: Vega II Duo (compatibility fallback)  
  
### Notes  
- Avoid >3 GPUs unless MGPU overhead is optimized  
- Ensure sufficient VRAM (≥128 GB recommended for large models)  
⸻  
## 6. Installation Procedure  
  
```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout <base_commit>

```
```


```
```
cmake -B build -DGGML_METAL=ON
cmake --build build -j

```
  
⸻  
## 7. Environment Variables & CLI Usage  
  
### Key Environment Variables  
  
```
# Device selection
export GGML_METAL_DEVICE_INDEX=1,2

# PP and Decode tuning

```
```


```
```
export GGML_METAL_PP_F32_NR0=4
export GGML_METAL_PP_F32_NSG=4
export GGML_METAL_DECODE_F32_NR0=4
export GGML_METAL_DECODE_F32_NSG=4

```
```
export GGML_METAL_DECODE_F32_SHAPE_THRESHOLD=2048

export GGML_METAL_PP_F16_NR0=4
export GGML_METAL_PP_F16_NSG=4
export GGML_METAL_DECODE_F16_NR0=4
export GGML_METAL_DECODE_F16_NSG=4
export GGML_METAL_DECODE_F16_SHAPE_THRESHOLD=2048

export GGML_METAL_PP_BF16_NR0=4
export GGML_METAL_PP_BF16_NSG=4
export GGML_METAL_DECODE_BF16_NR0=4
export GGML_METAL_DECODE_BF16_NSG=4
export GGML_METAL_DECODE_BF16_SHAPE_THRESHOLD=2048

export GGML_METAL_DECODE_Q6K_SHAPE_THRESHOLD=2048
export GGML_METAL_PP_Q6K_NR0=4
export GGML_METAL_PP_Q6K_NSG=2
export GGML_METAL_DECODE_Q6K_NR0=2
export GGML_METAL_DECODE_Q6K_NSG=2
export GGML_METAL_DECODE_Q6K_ID_NR0=2
export GGML_METAL_DECODE_Q6K_ID_NSG=2

export GGML_METAL_PP_Q50_NR0=8
export GGML_METAL_PP_Q50_NSG=1
export GGML_METAL_DECODE_Q50_NR0=2
export GGML_METAL_DECODE_Q50_NSG=4

```
```
export GGML_METAL_DECODE_Q50_SHAPE_THRESHOLD=2048


```
```
export GGML_METAL_DECODE_Q4K_SHAPE_THRESHOLD=2048
export GGML_METAL_PP_Q4K_NR0=8
export GGML_METAL_PP_Q4K_NSG=2
export GGML_METAL_DECODE_Q4K_NR0=2
export GGML_METAL_DECODE_Q4K_NSG=1
export GGML_METAL_DECODE_Q4K_ID_NR0=2
export GGML_METAL_DECODE_Q4K_ID_NSG=1

export GGML_METAL_PP_Q40_NR0=2
export GGML_METAL_PP_Q40_NSG=1
export GGML_METAL_DECODE_Q40_NR0=2
export GGML_METAL_DECODE_Q40_NSG=1
export GGML_METAL_DECODE_ID_Q40_NR0=2
export GGML_METAL_DECODE_ID_Q40_NSG=1

```
```
export GGML_METAL_DECODE_Q40_SHAPE_THRESHOLD=2048

```
```

# Flash Attention
export GGML_METAL_FA_VEC=1
export GGML_METAL_FA_NSG=4

# MGPU tuning
export GGML_METAL_MGPU_DECODE_OFFLOAD_MIN_BS=0
export GGML_METAL_MGPU_SMALL_MAT_OFFLOAD_MAX_BS=4

```
  
⸻  
### Benchmark Command  
  
```
./build/bin/llama-bench \
-ngl 99 \
-fa 1 \
-ub 32 \
-b 32 \
-p 512 \
-n 128 \
-m <model.gguf>

```
  
⸻  
### Server Command  
  
```
./build/bin/llama-server \
-ngl 99 \
-fa on \
-ub 32 \
-b 64 \
-m <model.gguf> \
--ctx-size 8192 \
-np 1

```
  
⸻  
  
  
  
  
  
## Summary  
  
iRon-Llama-RC2 focuses on pushing Metal V3 + AMD GPUs to their limits, prioritizing decode optimization, kernel tuning, and MGPU orchestration. While single-node Metal performance is strong, MGPU scaling remains an active optimization area.  
