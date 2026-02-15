# llama.cpp for Mac INtel with AMD GPUs (macOS Intel + AMD dGPU) – W6800X Duo / Xeon

This repository is a **llama.cpp (ggml-metal) focused release** tuned for **macOS Intel (MacPro7,1)** with **discrete AMD GPUs** (e.g. **Radeon PRO W6800X Duo**) and **Intel Xeon** CPUs.

The goal is to restore/ensure:

- **Correct GPU device selection** via `GGML_METAL_DEVICE_INDEX`
- **Correct + fast mmap loading on discrete GPUs** by mirroring mmapped weights into **VRAM (StorageModePrivate)** when possible
- **Stable text generation** on AMD dGPU by avoiding known non-determinism in Metal concurrency paths on discrete devices
- **VRAM budget & fallback**: if VRAM budget is exceeded, the backend falls back to shared mmap (slow but correct) instead of crashing

> Target hardware: Mac Pro (Intel) + AMD dGPU (RDNA2 / W6800X Duo).  
> Metal feature note: simdgroup matrix multiply ops are not available on these Macs; this release focuses on correctness + bandwidth-aware memory placement.

---

## Highlights of this version

### ✅ GPU Device Selection

Select which GPU to use (macOS):

```bash
export GGML_METAL_DEVICE_INDEX=2   # example index
```

### ✅ Discrete GPU mmap → VRAM Private Mirror (Fast mmap)

When loading a model with mmap (default), mmapped weights are first mapped as a **shared buffer** and then **one-shot blit copied** into a **private VRAM mirror** (when enabled and budget allows). Kernels read from the private mirror for bandwidth and stability.

### ✅ VRAM Budget + Fallback

To avoid VRAM thrashing/OOM:

- private mirrors are created only up to a computed budget
- when budget is exceeded, the backend **falls back** to shared mmap (slow but correct) and logs a warning

Configurable example for a 32 Go GPU W6800x:

```bash
export GGML_METAL_VRAM_RESERVE_MB=2048    # default reserve margin
export GGML_METAL_VRAM_BUDGET_MB=32000    # optional hard cap override
```
Configurable example for a 16 Go GPU RX6800xt:

```bash
export GGML_METAL_VRAM_RESERVE_MB=1024    # default reserve margin
export GGML_METAL_VRAM_BUDGET_MB=16000    # optional hard cap override
```

### ✅ Stability on AMD dGPU

On discrete AMD GPUs, some concurrency paths can cause **non-deterministic outputs** (garbled text).  
For stability:

```bash
export GGML_METAL_CONCURRENCY_DISABLE=1
```

---


### 📦 Installation

```bash
git clone https://github.com/Basten7/iRon-Llama-RC1.git
cd iRon-Llama-RC1
```

### 📦 Build for MacOS + Metal V3 

## ➡️ Prerequisites (Metal-only build)
```bash
brew install cmake git libomp glslang
```
➡️ Prerequisites (Metal + Vulkan build)
```bash
brew install cmake git libomp glslang molten-vk shaderc vulkan-loader vulkan-headers
```
➡️ Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL_MGPU=ON -DOpenMP_ROOT="$(brew --prefix)/opt/libomp" && cmake --build build -j
```

## 📦 Quick Validation

### 1) Verify GPU selection

```bash
export GGML_METAL_DEVICE_INDEX=2
./build/bin/llama-cli -ngl 99 -c 128 --jinja --no-mmap \
  -m ~/Models/Qwen3-30B-A3B-Q4_K_M.gguf -p "test"
```

Expected log snippet:

- `using device index X from GGML_METAL_DEVICE_INDEX`
- `GPU name: AMD Radeon PRO W6800X Duo` (or your selected GPU)

### 2) Fast + correct generation (recommended defaults for AMD dGPU)

```bash
export GGML_METAL_DEVICE_INDEX=2
export GGML_METAL_CONCURRENCY_DISABLE=1
unset GGML_METAL_MMAP_PRIVATE_DISABLE

./build/bin/llama-cli -ngl 99 -c 4096 --jinja \
  -m ~/Models/Qwen3-0.6B-F32.gguf \
  -p "Explain TensorFlow in French."
```

---

## Performance Notes

### mmap vs no-mmap

On discrete GPUs, `--no-mmap` is often fast because weights are copied into GPU-friendly buffers.  
This release aims to make mmap fast too by enabling a **private VRAM mirror**.

- Default: mmap + private mirror (fast on dGPU)
- Debug fallback (very slow on dGPU, reads over PCIe):

  ```bash
  export GGML_METAL_MMAP_PRIVATE_DISABLE=1
  ```

### VRAM Tuning

Your device logs `recommendedMaxWorkingSetSize`. Budget is derived from it:

- `budget = recommendedMaxWorkingSetSize - VRAM_RESERVE_MB`

Typical values:

- `VRAM_RESERVE_MB=2048` (safe)
- `VRAM_RESERVE_MB=1024` (aggressive / max perf, less safety margin)

---

## Benchmark

### llama-bench

```bash
export GGML_METAL_DEVICE_INDEX=2
export GGML_METAL_CONCURRENCY_DISABLE=1

./build/bin/llama-bench -ngl 99 \
  -m ~/Models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --mmap 1
```

---

## Server & Long Context (KV cache)

### Long-context slowdown is expected

As context grows, attention cost increases because each new token attends over an ever larger KV cache. This can drop token/s significantly at very large ctx sizes.

### Flash-Attn on macOS Intel + AMD dGPU (Recommended: `-fa auto`)

On macOS Intel with discrete AMD GPUs (e.g. W6800X Duo), forcing Flash-Attn can be highly sensitive to the KV cache
type and kernel availability. The most reliable choice is:

- **Use `-fa auto`**: enables Flash-Attn only when the backend supports it safely for the current configuration.

Example (recommended):

```bash
export GGML_METAL_DEVICE_INDEX=4
export GGML_METAL_CONCURRENCY_DISABLE=1
unset GGML_METAL_MMAP_PRIVATE_DISABLE
export GGML_METAL_VRAM_RESERVE_MB=2048

./build/bin/llama-server -ngl 99 -sm layer --port 8086 --jinja \
  --ctx-size 62144 \
  -fa auto \
  -m ~/Models/Qwen3-Coder-30B-A3B-Q4_0.gguf
```

Observed behavior on this setup:

- `-fa auto` provides strong prompt-processing throughput (e.g. ~80 tok/s in prompt eval on 30B-class models).
- Forcing Flash-Attn while using quantized KV cache (`--cache-type-v q8_0`) may become very slow and/or fall back to CPU.

---

## Environment Variables Reference

| Variable | Purpose | Suggested |
|---------:|---------|-----------|
| `GGML_METAL_DEVICE_INDEX` | Select GPU device index on macOS | set to your W6800X |
| `GGML_METAL_CONCURRENCY_DISABLE` | Disable concurrency for stability on discrete AMD GPUs | `1` |
| `GGML_METAL_ORDERED_COMMIT` | Ordered command buffer commits (debug) | optional |
| `GGML_METAL_MMAP_PRIVATE_DISABLE` | Disable mmap→private VRAM mirror (debug, slow on dGPU) | unset |
| `GGML_METAL_VRAM_RESERVE_MB` | VRAM safety margin | `2048` |
| `GGML_METAL_VRAM_BUDGET_MB` | Override total VRAM budget | optional |

---

## Known Limitations

- Metal simdgroup matrix multiply ops are not supported on these Intel + AMD dGPU Macs.
- Very large context sizes will still slow down generation due to KV cache bandwidth.
- Some concurrency paths on discrete GPUs can cause non-deterministic outputs; this repo defaults to stable behavior via recommended env vars.

---
#RC2  Support for Flash Attention

./build/bin/llama-bench -ngl 99 --mmap 0 -m ~/Models/Qwen3-Coder-30B-A3B-Instruct-1M-Q4_K_S.gguf  -fa 1,0

ggml_metal_device_init: using device index 2 from GGML_METAL_DEVICE_INDEX

ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices

ggml_metal_library_init: using embedded metal library

ggml_metal_library_init: loaded in 0.038 sec

ggml_metal_rsets_init: creating a residency set collection (keep_alive = 180 s)

ggml_metal_device_init: GPU name:   AMD Radeon PRO W6800X Duo

ggml_metal_device_init: GPU family: MTLGPUFamilyCommon3 (3003)

ggml_metal_device_init: GPU family: MTLGPUFamilyMetal3  (5001)

ggml_metal_device_init: simdgroup reduction   = true

ggml_metal_device_init: simdgroup matrix mul. = false

ggml_metal_device_init: has unified memory    = false

ggml_metal_device_init: has bfloat            = false

ggml_metal_device_init: has tensor            = false

ggml_metal_device_init: use residency sets    = true

ggml_metal_device_init: use shared buffers    = false

ggml_metal_device_init: recommendedMaxWorkingSetSize  = 34342.96 MB

| model                          |       size |     params | backend    | threads | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Small  |  16.25 GiB |    30.53 B | Metal,BLAS |      12 |  1 |           pp512 |        255.73 ± 0.08 |
| qwen3moe 30B.A3B Q4_K - Small  |  16.25 GiB |    30.53 B | Metal,BLAS |      12 |  1 |           tg128 |         72.25 ± 0.42 |
| qwen3moe 30B.A3B Q4_K - Small  |  16.25 GiB |    30.53 B | Metal,BLAS |      12 |  0 |           pp512 |        189.06 ± 0.03 |
| qwen3moe 30B.A3B Q4_K - Small  |  16.25 GiB |    30.53 B | Metal,BLAS |      12 |  0 |           tg128 |         76.96 ± 0.16 |

build: a85804025 (7926)

## License

This repository is based on upstream **llama.cpp** (ggml). Please refer to upstream licensing terms and preserve notices in derived works.
