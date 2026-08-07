# threadchip/moe-prefetch

Upstream **b10298** (`15586e2d7`) plus three MoE-offload prefill optimizations
cherry-picked (`-x`, original SHAs preserved) from
[thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp)
`fable5/prefetch-experts`, authored with Claude Fable 5 —
see [the video](https://youtu.be/VytSYCDhWQ0).

Both optimizations are **opt-in via environment variables** and produce
**token-identical** output:

| Env var | What it does |
|---|---|
| `GGML_CUDA_REGISTER_HOST=1` | Page-locks mmap'd CPU expert weights so host→device copies go over DMA instead of the driver's bounce buffer |
| `GGML_SCHED_PREFETCH_EXPERTS=1` | Prefetches each layer's experts on a second CUDA stream, overlapping uploads with compute |

## Measured

| rig | model | prefill gain |
|---|---|---|
| RTX 3060 12 GB, PCIe 4.0 (thecodacus) | Qwen3.6-35B-A3B `--n-cpu-moe 26` | **+64 %** (1143 → 1880 t/s) |
| GTX 1080 Ti 11 GB, PCIe 3.0 (threadchip) | same | **+13.4 %** (858 → 972 t/s) |
| GTX 1080 Ti 11 GB, PCIe 3.0 (threadchip) | Laguna-S-2.1 118B `--n-cpu-moe 46` | **+14.2 %** (190 → 217 t/s) |
| GTX 1080 Ti 11 GB, PCIe 3.0 (threadchip) | DeepSeek-V4-Flash **IQ3_XXS** `--n-cpu-moe 99` | **0 %** (98.9 → 98.7 t/s) |

**Quant boundary:** the patches accelerate the expert *upload* path. Q4_K models take that
path and gain; IQ3_XXS apparently never does, and gains nothing. If your MoE runs an IQ-class
quant and shows no benefit, that is expected, not a regression.

**Interaction found in service use:** with `GGML_CUDA_REGISTER_HOST=1`, loading a co-located
**BF16 mmproj** (vision projector) on a Pascal card aborts on the first request. Text-only
(`--no-mmproj` / suppressing auto-detect) avoids it.

The gap is the bus: the pinning patch lifts H2D toward ~20 GB/s, PCIe 3.0 ×16 caps at
15.8 GB/s. The faster your PCIe, the more these patches are worth.

## Maintenance

Rebase this branch onto new upstream releases as needed:

```bash
git fetch origin --tags
git rebase <new-tag> threadchip/moe-prefetch
```

The three commits touch `llama-mmap.cpp`, `ggml-cuda.cu` and `ggml-backend.cpp` narrowly;
conflicts should be rare and small.
