# Qwen3-ASR on the Jetson Nano gen1 (Tegra X1, sm_53, CUDA 10.2)

This fork adds the changes needed to run **Qwen3-ASR-0.6B** with CUDA on the
original **Jetson Nano gen1** (Tegra X1, 4× Cortex-A57, 128-core Maxwell GPU
`sm_53`, CUDA 10.2 / JetPack 4.6, 4 GB RAM) and optimizes it to **RTF ≈ 1.1**
for short clips — native Traditional-Chinese + punctuation, zh/en code-switch.

Measured on-device (13.5 s zh-TW clip, q8_0):

| stage | time |
|---|---|
| mel (cuFFT, batched R2C) | 1.7 s |
| audio encoder (GPU) | 4.8 s |
| LLM prefill (GPU) | 2.7 s |
| LLM decode (CPU, 195 ms/tok) | 5.3 s |
| **total** | **14.7 s → RTF ≈ 1.1** |

Starting point was RTF ≈ 4.4; the 3.9× speedup comes from the changes below.

## What this fork changes

1. **cuFFT batched mel** (`src/mel_cufft.cu`, `src/mel_cufft.h`) — replaces the
   reference per-frame O(N²) DFT (≈22 s) with one batched R2C FFT on the GPU
   (≈1.7 s). Numerically faithful (1+N/2 = 201 bins, same power+filterbank+log10).
   Gated by `RS_USE_CUDA`/`__CUDACC__`; the CPU path remains the fallback.
2. **Prefill-on-GPU / decode-on-CPU split** (`src/text_decoder.*`) — a CPU-only
   scheduler + dual KV cache. The big parallel prefill runs on the GPU; the
   batch-1 autoregressive decode runs on the A57 CPU (~195 ms/tok), which on
   Maxwell `sm_53` is ~3.7× faster than per-token GPU weight uploads.
3. **Manual attention instead of flash-attention** (`src/text_decoder.cpp`) —
   `sm_53` has no flash-attention kernel and batch-1 decode violates
   `GGML_KQ_MASK_PAD`; the manual `soft_max_ext` path is correct on every backend.
4. **CUDA-10.2 / sm_53 loader fix** (`src/gguf_loader.cpp`, etc.) — the older
   ggml CUDA backend has a null `buffer_from_host_ptr` iface; fall back to a CPU
   buffer (the scheduler still computes on CUDA).

## Build (Jetson Nano gen1)

CUDA 10.2's `nvcc` cannot compile the C++17 `if constexpr` used by recent ggml,
so pin ggml to a compatible commit and apply the `sm_53` patch:

```sh
# 1. pin ggml to the CUDA-10.2/sm_53-compatible commit
git -C ggml fetch --depth 300 origin && git -C ggml checkout b2a092a7
git -C ggml apply ../patches/ggml-cuda-10.2-sm53.patch

# 2. CUDA 10.2 has no cuda_bf16.h; stage the stub + nvcc compat shims
#    (see scripts/nano_cuda_compat/: cuda_bf16.h, nvcc_compat.h, neon_x4_shim.h)

# 3. configure with gcc-8 (gcc-7.5 lacks <charconv>), sm_53, cuFFT
cmake -B build-nano -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 -DCMAKE_CUDA_HOST_COMPILER=g++-8 \
  -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_ARCHITECTURES=53 -DGGML_CUDA_NO_VMM=ON \
  -DCMAKE_CUDA_FLAGS="--forward-unknown-to-host-compiler -arch=sm_53 -I scripts/nano_cuda_compat -include scripts/nano_cuda_compat/nvcc_compat.h"
cmake --build build-nano -j4
```

A pre-built cross-build image is documented in the RapidSpeech.cpp fork below.

## Model

GGUF (f16 / q8_0 / q4_0) — **q8_0 recommended on the Nano** (half size, same
accuracy/speed; q4_0 is slower on Maxwell and drops to Simplified):

- https://huggingface.co/Luigi/qwen3-asr-0.6b-rapidspeech-gguf

```sh
./build-nano/qwen3-asr-cli -m qwen3-asr-0.6b-q8_0.gguf -f clip_16k.wav -t 4
```

## See also

- **RapidSpeech.cpp fork** (this engine integrated into the multi-model
  RapidSpeech framework, `jetson-nano-gen1` branch; all-GPU RTF ≈ 1.3, shares
  the same cuFFT mel + converter):
  https://github.com/vieenrose/RapidSpeech.cpp/tree/jetson-nano-gen1
- Upstream: https://github.com/predict-woo/qwen3-asr.cpp
