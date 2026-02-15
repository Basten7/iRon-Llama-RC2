TEST OK pour perf et cohérence: ./build/bin/llama-bench -ngl 99 --mmap 0 -m /Models/Qwen3-Coder-30B-A3B-Instruct-1M-Q4_K_S.gguf -fa 1,0 ggml_metal_device_init: using device index 2 from GGML_METAL_DEVICE_INDEX ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices ggml_metal_library_init: using embedded metal library ggml_metal_library_init: loaded in 0.038 sec ggml_metal_rsets_init: creating a residency set collection (keep_alive = 180 s) ggml_metal_device_init: GPU name: AMD Radeon PRO W6800X Duo ggml_metal_device_init: GPU family: MTLGPUFamilyCommon3 (3003) ggml_metal_device_init: GPU family: MTLGPUFamilyMetal3 (5001) ggml_metal_device_init: simdgroup reduction = true ggml_metal_device_init: simdgroup matrix mul. = false ggml_metal_device_init: has unified memory = false ggml_metal_device_init: has bfloat = false ggml_metal_device_init: has tensor = false ggml_metal_device_init: use residency sets = true ggml_metal_device_init: use shared buffers = false ggml_metal_device_init: recommendedMaxWorkingSetSize = 34342.96 MB | model | size | params | backend | threads | fa | test | t/s | | ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --------------: | -------------------: | | qwen3moe 30B.A3B Q4_K - Small | 16.25 GiB | 30.53 B | Metal,BLAS | 12 | 1 | pp512 | 255.73 ± 0.08 | | qwen3moe 30B.A3B Q4_K - Small | 16.25 GiB | 30.53 B | Metal,BLAS | 12 | 1 | tg128 | 72.25 ± 0.42 | | qwen3moe 30B.A3B Q4_K - Small | 16.25 GiB | 30.53 B | Metal,BLAS | 12 | 0 | pp512 | 189.06 ± 0.03 | | qwen3moe 30B.A3B Q4_K - Small | 16.25 GiB | 30.53 B | Metal,BLAS | 12 | 0 | tg128 | 76.96 ± 0.16 | build: a85804025 (7926) TEST COHéRENCE ./build/bin/llama-cli -ngl 99 --no-mmap -p "génère un tutorial sur Pytorch poour un Datascientist" -c 80144 -m ~/Models/Qwen3-Coder-30B-A3B-Q4_K_M.gguf --jinja -fa auto ggml_metal_device_init: using device index 1 from GGML_METAL_DEVICE_INDEX ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices ggml_metal_library_init: using embedded metal library ggml_metal_library_init: loaded in 0.037 sec ggml_metal_rsets_init: creating a residency set collection (keep_alive = 180 s) ggml_metal_device_init: GPU name: AMD Radeon PRO W6800X Duo ggml_metal_device_init: GPU family: MTLGPUFamilyCommon3 (3003) ggml_metal_device_init: GPU family: MTLGPUFamilyMetal3 (5001) ggml_metal_device_init: simdgroup reduction = true ggml_metal_device_init: simdgroup matrix mul. = false ggml_metal_device_init: has unified memory = false ggml_metal_device_init: has bfloat = false ggml_metal_device_init: has tensor = false ggml_metal_device_init: use residency sets = true ggml_metal_device_init: use shared buffers = false ggml_metal_device_init: recommendedMaxWorkingSetSize = 34342.96 MB Loading model... build : b7926-a85804025 model : Qwen3-Coder-30B-A3B-Q4_K_M.gguf modalities : text available commands: /exit or Ctrl+C stop or exit /regen regenerate the last response /clear clear the chat history /read add a text file > génère un tutorial sur Pytorch poour un Datascientist Ce tutorial vous donne les bases nécessaires pour commencer à utiliser PyTorch dans vos projets de data science ! [ Prompt: 197,3 t/s | Generation: 67,3 t/s ]
Note sur tes chiffres FA

PP512 : +35% environ avec FA (énorme)

TG128 : -6% environ avec FA

Si tu veux optimiser le decode, on pourra :

forcer FA uniquement sur prefill (si tu as un mode), ou

vérifier si ton TG est limité par autre chose (bande passante, scheduling, kernels mul_mv, etc.)

export GGML_METAL_DEVICE_INDEX=1
export GGML_METAL_CONCURRENCY_DISABLE=1
unset GGML_METAL_FORCE_PRIVATE
./build/bin/llama-cli -ngl 99 --no-mmap -c 80144 \
  -m ~/Models/Qwen3-Coder-30B-A3B-Q4_K_M.gguf \
  --jinja -fa auto \
  -p "écris une fonction Python qui charge un CSV et calcule des stats de base"
Parfait : ton commit 4f8d26485 (tag TT__gemv_FA-OK) touche exactement les 4 zones “sensibles” :

ggml-metal-device.m : fix upload/sync AMD

ggml-metal-device.cpp : device/props côté C++

ggml-metal-ops.cpp : fallback flash_attn_ext vec + garde pipeline

tools/cli/cli.cpp : probablement options/params (FA auto, etc.)

git tag -a iron-7926-amd-fa-stable -m "AMD W6800X: stable upload + FA fallback + coherence OK"
git show --name-only --oneline --decorate iron-7926-amd-fa-stable
