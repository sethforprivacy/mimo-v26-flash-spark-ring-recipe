# GB10 microbenchmarks behind the overlays

These are the scripts behind `vllm-patches-attn2`, `vllm-patches-fp8cfg` and `vllm-patches-roce`, and behind the
NCCL channel choice. The single-GPU ones ran inside the serving image on one Spark's GPU while the lane was idle:

```bash
mkdir -p ~/attn-bench && cp kernel-tuning/*.py ~/attn-bench/
docker run --rm --gpus all -v ~/attn-bench:/w -e PYTHONPATH=/w --entrypoint python3 \
  myllmbox/mimo-v26-flash-cluster-vllm:v2 /w/<script>
```

Every kernel script checks its outputs against the stock launch of the same inputs.

| script | what it measures | result we got (GB10, MiMo TP4 per-rank shapes) |
|---|---|---|
| `attn_bench.py` | TRITON_ATTN_DIFFKV 2D (prefill) launch sweep over `BLOCK_M:TILE:num_warps:num_stages`. `SHAPES=ga:<q>:<kv>[:<seqs>]` (full attention, 16 q heads / 1 kv head per rank) or `swa:...` | full-attention prefill 2.9-3.2x (16K chunk at 16K: 70.5 to 21.9 ms), SWA prefill 1.8x |
| `attn_bench3d.py` | decode / spec-verify (3D split-KV) sweep including the segment count; `2D:` configs for comparison | full-attention verify 1.2-2.3x, SWA verify 1.25-3.5x on the 2D path |
| `attn_patch_test.py` | stock `unified_attention_diffkv()` vs the patched module (`PATCHED_OPS`, default `/w/patched_ops.py`: copy `overlays/vllm-patches-attn2/v1/attention/ops/triton_unified_attention_diffkv.py` there), through the real call | 1.04-3.36x on every tested shape for the first tuned version; worst output difference 0.00098 (bf16 rounding) |
| `attn_split_test.py` + `triton_unified_attention_diffkv_qksplit.py` | the QK-split experiment: stock vs tuned vs tuned with Q.K^T as 128 + 64 (no pad to 256) and wider tiles. The module is the attn2 file with the split off by default and the earlier prefill tiles; copy it to `/w/split_ops.py` | full-attention prefill +11-15 % over the tuned launch with 128:64:8:2 tiles, SWA prefill +8 %, decode unchanged, outputs within 1e-4 |
| `tune_fp8.py` + `benchmark_w8a8_block_fp8.py` | block-FP8 GEMM configs for the four per-rank shapes the boot log reports as untuned on NVIDIA_GB10, via vLLM's own tuner over a pruned space (BLOCK_K=128). Writes the stock-named JSON files to `/w/out` | the four files in `overlays/vllm-patches-fp8cfg/` |
| `cmp_fp8_cold.py` | default vs tuned configs with L2-cold rotating weights (the decode case) | 1.2-1.4x at M = 4-64 on the QKV shapes (default ~145-150 GB/s, tuned 190-215 GB/s); 1.0-1.1x on the dense 8192 shape |
| `attn_fp8kv_test.py` | FP8 KV numerics and speed for `overlays/vllm-patches-fp8kv` (copy its ops file to `/w/fp8kv_ops.py`): bf16 cache vs the dequantized cache vs the FP8 cache, per-tensor scales 1.0 and 0.25 / 0.5 | FP8 path matches the bf16 kernel on the dequantized cache to <= 0.00098; FP8 vs bf16 cache ~3 % relative (the quantization itself) |
| `attn_fp8_dec_sweep.py` + `stable_timer.py` | FP8 spec-verify (3D) launch sweep (TILE / warps / stages / segments) and FP8 prefill-tail (2D) sweep at 250K / 500K / 1M against the bf16 production launch; interleaved, warmed-up timing (short GB10 kernel timings swing with clock state) | verify 64:32:4:2 with 64-128 segments: 1.59 / 1.64 / 1.68x vs bf16 (2.09x at 2 x 500K); prefill tail 128:128:8:2: 0.91x of bf16 |
| `attn_bf16_dec_sweep.py` | the same verify sweep for the bf16 launch (copy `overlays/vllm-patches-attn2/.../triton_unified_attention_diffkv.py` to `/w/bf16_ops.py`) | production launch already at 240-255 GB/s for one sequence; best alternative 0.99-1.03x (1.17x only at 2 x 500K) |
| `nccl_ar_bench.py` + `nccl-bench.sh` | 4-rank bf16 all-reduce latency, eager and CUDA-graph replay, one NCCL knob per config | NCCL's own channel choice was 2.7-5.3x slower than 4 channels at 512 KB-1 MB (graph: 390 / 993 us vs 145 / 188 us) |
| `roce_rd_bench.py` + `roce-rd-bench.sh` | ring-only RoCEnante recursive doubling (two 2-rank b12x runtimes) vs NCCL ring, eager and graph replay, exactness vs NCCL | graph replay per call, RD vs NCCL: 8 KB 19.0 vs 70.6 us, 32 KB 24.8 vs 76.1, 128 KB 42.4 vs 85.1, 512 KB 100.1 vs 135.5, 1 MB 184.6 vs 187.2, 2 MB 328.6 vs 275.1; bit-identical across ranks |
| `roce_rd_stress.py` (`RD_SCRIPT=roce_rd_stress.py roce-rd-bench.sh`) | stale-data stress: 106 all-reduces of 8 KB-1 MB per graph replay with varying compute gaps, 3,000 replays with pauses; every replay must reproduce the first bit for bit | 1.27M all-reduces, 0 mismatches |

Notes:
- The ring scripts (`nccl-bench.sh`, `roce-rd-bench.sh`) run from rank 0's host, with the ring idle. They read
  `launcher/hosts.env` and start one throwaway container per rank, so the recipe must sit at the same path on all
  four nodes. `roce-rd-bench.sh` needs the b12x checkout (`$B12X_ROOT/e4084d2e`) on every node.
- `benchmark_w8a8_block_fp8.py` is vLLM's `benchmarks/kernels/benchmark_w8a8_block_fp8.py` at nightly `ddd6fbca`
  (Apache-2.0, unchanged). `triton_unified_attention_diffkv_qksplit.py` is a modified vLLM file (Apache-2.0); see
  `overlays/NOTICE`.
- Kernel speedups are microbenchmark numbers. For what they did in serving, see the README.
