#!/usr/bin/env python3
"""Tune vLLM's Triton W8A8 block-FP8 GEMM for MiMo-V2.6-Flash TP=4 per-rank shapes on GB10.

Wraps benchmarks/kernels/benchmark_w8a8_block_fp8.py @ ddd6fbca (its DeepSeek shape list and 1280-config space
replaced): the four shapes the boot log reports as untuned on NVIDIA_GB10, a pruned search space (BLOCK_K=128 to
match the 128x128 weight blocks), and the M values this lane runs: verify batches (K=3: 4 tokens/seq, up to 32
seqs) and prefill chunks up to --max-num-batched-tokens 16384. Writes the stock-named JSON files to /w/out.
"""
import os, sys, json, time
import torch
sys.path.insert(0, "/w")
import benchmark_w8a8_block_fp8 as b

SHAPES = [(3392, 4096), (3712, 4096), (8192, 4096), (4096, 4096)]
MS = [int(x) for x in os.environ.get("MS", "1,2,4,8,16,24,32,48,64,96,128,256,512,1024,2048,4096,8192,16384").split(",")]


def space():
    out = []
    for st in (2, 3, 4):
        for bm in (16, 32, 64, 128):
            for bn in (32, 64, 128):
                for w in (4, 8):
                    for g in (1, 16):
                        out.append({"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": g,
                                    "num_warps": w, "num_stages": st})
    return out


def main():
    torch.cuda.init()
    sp = space()
    os.makedirs("/w/out", exist_ok=True)
    for n, k in SHAPES:
        t0 = time.time()
        best = {}
        for m in MS:
            best[m] = b.tune(m, n, k, [128, 128], torch.bfloat16, sp, "fp8")
        b.save_configs(n, k, 128, 128, best, "/w/out", "fp8")
        print(json.dumps({"N": n, "K": k, "seconds": round(time.time() - t0), "best": {str(m): best[m] for m in MS}}),
              flush=True)


if __name__ == "__main__":
    main()
