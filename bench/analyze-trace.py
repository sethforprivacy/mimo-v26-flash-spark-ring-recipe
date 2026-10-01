#!/usr/bin/env python3
"""Summarise a vLLM torch-profiler trace (chrome JSON, .json or .json.gz) of decode steps (2026-09-30).

  python3 analyze-trace.py <trace.json.gz> [--top 25]

GPU kernel time by category (regex on the kernel name), the union of busy time vs the window (idle gaps = launch /
CPU / sync overhead), and the top kernels. Categories are MiMo-V2.6-on-vLLM specific."""
import gzip, json, re, sys
from collections import defaultdict

CATS = [
    ("attention", r"unified_attention|reduce_segments|reshape_and_cache|diffkv"),
    ("nccl", r"nccl"),
    ("roce", r"roce"),  # b12x RoCEnante one-shot kernels (2026-10-01; their names also contain "cute")
    ("moe", r"marlin|moe|expert|topk_softmax|grouped_topk|align_block|count_and_sort"),
    ("fp8_gemm", r"w8a8|fp8|block_fp8|_matmul_kernel|cutlass.*sm120"),
    ("nvfp4_gemm", r"nvfp4|fp4|cute|cutedsl|gemm_kernel"),
    ("gemm_other", r"gemm|gemv|sm\d+_xmma|cublas|splitk"),
    ("sampling_spec", r"rejection|resample|topk_topp|topp|sampl|argmax|logits|softmax_kernel|_compute_local"),
    ("norm_rope_elementwise", r"rms|norm|rotary|rope|triton_(poi|red|per)_fused|elementwise|vectorized|silu|act_and_mul|copy|fill|cat"),
]


def cat_of(name):
    n = name.lower()
    for c, rx in CATS:
        if re.search(rx, n):
            return c
    return "other"


def main():
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        tr = json.load(f)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    ks = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not ks:
        print("no GPU kernel events"); return
    t0 = min(e["ts"] for e in ks); t1 = max(e["ts"] + e["dur"] for e in ks)
    by_cat = defaultdict(float); by_name = defaultdict(lambda: [0.0, 0])
    for e in ks:
        c = cat_of(e["name"]); by_cat[c] += e["dur"]
        k = by_name[e["name"][:110]]; k[0] += e["dur"]; k[1] += 1
    ivs = sorted((e["ts"], e["ts"] + e["dur"]) for e in ks)
    busy = 0.0; cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s > ce:
            busy += ce - cs; cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    win = t1 - t0
    total = sum(by_cat.values())
    # step count: the LM-head / sampler kernels run once per engine step; use the rejection sampler if present
    steps = max((v[1] for n, v in by_name.items() if re.search(r"rejection", n.lower())), default=0)
    print(json.dumps({"window_ms": round(win / 1e3, 1), "gpu_busy_ms": round(busy / 1e3, 1),
                      "idle_share": round(1 - busy / win, 3), "kernel_sum_ms": round(total / 1e3, 1),
                      "steps_est": steps, "ms_per_step_est": round(win / 1e3 / steps, 2) if steps else None}))
    for c, v in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"{c:24s} {v / 1e3:9.1f} ms  {100 * v / total:5.1f} %" + (f"  {v / 1e3 / steps:7.2f} ms/step" if steps else ""))
    print("--- top kernels")
    for n, (d, cnt) in sorted(by_name.items(), key=lambda x: -x[1][0])[:top]:
        print(f"{d / 1e3:9.1f} ms {cnt:6d}x  {cat_of(n):12s} {n}")


if __name__ == "__main__":
    main()
