# COLLAB NOTES — Ampere sm_86 Garbage Output Bug

**Branch:** `fix/ampere-sm86-garbage`
**GitHub Issue:** Indras-Mirror/llama.cpp-mtp#4
**Date:** 2026-05-10
**Status:** OPEN — needs sm_86 testing (no 3090/A5000 available locally)

## Summary

All Ampere GPUs (sm_86 / GA10x) produce garbled/infinite output. Ada Lovelace (sm_89 / 4090) works fine. Bug is in this fork's kernel changes — same GGUF + same hardware works correctly on upstream.

## Key Files To Investigate

| File | What | Risk |
|------|------|------|
| `ggml/src/ggml-cuda/fattn-mma-tbq4.cuh` | TBQ4 tile loader, centroid dequant inline | HIGH |
| `ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh` | Template launcher, shmem calc | HIGH |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | Modified — TBQ4 guards (4 locations) | MED |
| `ggml/src/ggml-cuda/fattn.cu` | TBQ4 dispatch + rotation kernel calls | MED |
| `ggml/src/ggml-cuda/tbq4-cuda.cuh` | FWHT, quantize, dequant | MED |
| `src/models/qwen35_mtp.cpp` | MTP tensor sharing | MED |

## Approach

### Phase 1 — Diagnose
1. Add `GGML_CUDA_TBQ4_DISABLE` env var to force F16 KV path even with `-ctk tbq4_0`
2. Add `#if __CUDA_ARCH__ >= 890` guard around TBQ4 fused FA dispatch — fall to GPU-dequant path on sm_86
3. Compare shmem layouts between sm_86 and sm_89

### Phase 2 — Fix
Once isolated, fix the specific kernel.

### Phase 3 — Verify
- A/B test across all 4 configs
- PPL vs upstream turboquant
- Decode speed benchmark

## Build & Test (for Ampere GPU owners)

```bash
git checkout fix/ampere-sm86-garbage
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc) --target llama-server
./build/bin/llama-server -m any-qwen3.6.gguf --port 8098 -c 4096 \
  --flash-attn on -ngl 99 -ctk q4_0 -ctv q4_0 --temp 0
# Test: curl localhost:8098/completion -d '{"prompt":"Capital of France?","max_tokens":20}'
# Expected: "Paris" — if "/////////" or "!!!!!!!!!" → bug reproduced
```

## Contact
- GitHub: @Indras-Mirror (issue #4)
- Relay: `quetza-codetl`, thread `tbq4-coordination`
