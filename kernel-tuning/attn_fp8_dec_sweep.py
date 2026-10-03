#!/usr/bin/env python3
"""FP8 KV spec-verify (3D split-KV) launch sweep at long context, GB10, MiMo TP=4 full-attention shapes (2026-10-03).

The 10-03 traces at ~500K showed the FP8 verify kernel at 762 us vs 1,000 us for bf16 (1.31x on half the bytes,
~177 GB/s vs ~270 GB/s): it still runs the bf16-tuned 3D launch (DEC_FULL 64:64:4:2, 32 segments). This sweeps the
FP8 path's TILE / warps / stages / segment count against the bf16 production launch on the same cache.

Loads /w/fp8kv_ops.py (copy of vllm-patches-fp8kv/v1/attention/ops/triton_unified_attention_diffkv.py) and
attn_bench.make(); times with stable_timer.ab() (warm-up, interleaved, >= MIN_S per measurement).
Prints one JSON row per (shape, config) and a SUMMARY per shape.
"""
import importlib.util, itertools, json
import torch
from attn_bench import make
import stable_timer
from stable_timer import ab

stable_timer.MIN_S, stable_timer.ROUNDS = 0.1, 3
spec = importlib.util.spec_from_file_location("fp8kv_ops", "/w/fp8kv_ops.py")
ops = importlib.util.module_from_spec(spec); spec.loader.exec_module(ops)
FP8, QK, V, MAXSEG = torch.float8_e4m3fn, 192, 128, 128
PROD = ((64, 64, 4, 2), 32)


def bufs(nq, segs, rows):
    return (torch.empty(rows, nq, segs, 128, device="cuda", dtype=torch.float32),
            torch.empty(rows, nq, segs, device="cuda", dtype=torch.float32),
            torch.empty(rows, nq, segs, device="cuda", dtype=torch.float32))


def runner(s, k, v, b, maxq, cfg, segs, kd=None, vd=None):
    so, sm, se = b

    def fn():
        ops._MIMO_DEC_FULL, ops._MIMO_SEGS = cfg, (segs, segs, 4)
        ops.unified_attention_diffkv(
            q=s["q"], k=k, v=v, out=s["out"], cu_seqlens_q=s["cu"], seqused_k=s["seqused"],
            softmax_scale=QK ** -0.5, causal=True, window_size=(-1, -1), block_table=s["bt"], softcap=0.0,
            max_seqlen_q=maxq, alibi_slopes=None, sinks=None, seq_threshold_3D=128, num_par_softmax_segments=segs,
            softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se, k_descale=kd, v_descale=vd)
    return fn


CONFIGS = [((64, t, w, st), sg) for t, w, st, sg in itertools.product((32, 64, 128), (4, 8), (2, 3, 4), (16, 32, 48, 64, 96, 128))]
for qlen, slen, nseq in [(4, 250000, 1), (4, 500000, 1), (4, 1000000, 1), (4, 500000, 2)]:
    s = make("ga", qlen, slen, nseq)
    phys16 = torch.cat([s["k"], s["v"]], dim=-1).transpose(1, 2).contiguous()
    phys8 = phys16.to(FP8)                                    # serving scale 1.0
    k16, v16 = phys16.transpose(1, 2)[..., :QK], phys16.transpose(1, 2)[..., QK:]
    k8, v8 = phys8.transpose(1, 2)[..., :QK], phys8.transpose(1, 2)[..., QK:]
    one = torch.ones(1, device="cuda", dtype=torch.float32)
    b = bufs(s["nq"], MAXSEG, qlen * nseq + 8)
    f16 = runner(s, k16, v16, b, qlen, *PROD)
    f8 = runner(s, k8, v8, b, qlen, *PROD, kd=one, vd=one)
    f8(); torch.cuda.synchronize(); ref8 = s["out"].clone()
    base = ab({"bf16_prod": f16, "fp8_prod": f8})
    t16, t8 = base["bf16_prod"][0], base["fp8_prod"][0]
    shape = f"ga:{qlen}:{slen}:{nseq}"
    rows = []
    for cfg, segs in CONFIGS:
        fn = runner(s, k8, v8, b, qlen, cfg, segs, kd=one, vd=one)
        try:
            s["out"].zero_(); fn(); torch.cuda.synchronize()
            err = (s["out"].float() - ref8.float()).abs().max().item()
            t = ab({"x": fn})["x"][0]
        except Exception as e:  # noqa: BLE001 - shared-memory overflows etc.
            print(json.dumps({"shape": shape, "cfg": cfg, "segs": segs, "error": repr(e)[:120]}), flush=True)
            continue
        row = {"shape": shape, "cfg": cfg, "segs": segs, "us": round(t, 1), "vs_bf16": round(t16 / t, 3),
               "vs_fp8_prod": round(t8 / t, 3), "maxdiff_vs_fp8_prod": round(err, 5)}
        rows.append(row); print(json.dumps(row), flush=True)
    best = sorted((r for r in rows if r["maxdiff_vs_fp8_prod"] < 0.02), key=lambda r: r["us"])[:5]
    gb = slen * nseq * (QK + V) * 1 / 1e9   # FP8 bytes per call
    print("SUMMARY " + json.dumps({"shape": shape, "bf16_prod_us": round(t16, 1), "fp8_prod_us": round(t8, 1),
                                   "fp8_prod_vs_bf16": round(t16 / t8, 3), "best": best,
                                   "best_GBps": round(gb / (best[0]["us"] * 1e-6), 1) if best else None}), flush=True)
    del s, phys16, phys8, b; torch.cuda.empty_cache()

# Phase 2: the 2D (prefill) path for a hot agent turn, ~1,500 new tokens on a long cached context. FP8 prefill ran
# 0.83-0.86x of bf16 on 10-01, so with this lane's 1M contexts the hot-turn TTFT costs +15-20 % (v1-lc1m 6.1-6.4 s vs
# 5.1-5.5 s). Sweep PF_FULL (BLOCK_M:TILE:warps:stages) for FP8 against the bf16 production launch.
PROD_PF = (128, 64, 8, 2)


def runner2d(s, k, v, b, maxq, cfg, kd=None, vd=None):
    so, sm, se = b

    def fn():
        ops._MIMO_PF_FULL = cfg
        ops.unified_attention_diffkv(
            q=s["q"], k=k, v=v, out=s["out"], cu_seqlens_q=s["cu"], seqused_k=s["seqused"],
            softmax_scale=QK ** -0.5, causal=True, window_size=(-1, -1), block_table=s["bt"], softcap=0.0,
            max_seqlen_q=maxq, alibi_slopes=None, sinks=None, seq_threshold_3D=128, num_par_softmax_segments=32,
            softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se, k_descale=kd, v_descale=vd)
    return fn


PF_CONFIGS = [(bm, t, w, st) for bm, t, w, st in itertools.product((64, 128), (32, 64, 128), (4, 8), (2, 3))]
for qlen, slen in [(1500, 250000), (1500, 500000), (1500, 1000000)]:
    s = make("ga", qlen, slen, 1)
    phys16 = torch.cat([s["k"], s["v"]], dim=-1).transpose(1, 2).contiguous()
    phys8 = phys16.to(FP8)
    k16, v16 = phys16.transpose(1, 2)[..., :QK], phys16.transpose(1, 2)[..., QK:]
    k8, v8 = phys8.transpose(1, 2)[..., :QK], phys8.transpose(1, 2)[..., QK:]
    one = torch.ones(1, device="cuda", dtype=torch.float32)
    b = bufs(s["nq"], 32, 8)
    f16 = runner2d(s, k16, v16, b, qlen, PROD_PF)
    f8 = runner2d(s, k8, v8, b, qlen, PROD_PF, kd=one, vd=one)
    f8(); torch.cuda.synchronize(); ref8 = s["out"].clone()
    base = ab({"bf16_prod": f16, "fp8_prod": f8})
    t16, t8 = base["bf16_prod"][0], base["fp8_prod"][0]
    shape = f"ga2d:{qlen}:{slen}:1"
    rows = []
    for cfg in PF_CONFIGS:
        fn = runner2d(s, k8, v8, b, qlen, cfg, kd=one, vd=one)
        try:
            s["out"].zero_(); fn(); torch.cuda.synchronize()
            err = (s["out"].float() - ref8.float()).abs().max().item()
            t = ab({"x": fn})["x"][0]
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"shape": shape, "cfg": cfg, "error": repr(e)[:120]}), flush=True)
            continue
        row = {"shape": shape, "cfg": cfg, "us": round(t, 1), "vs_bf16": round(t16 / t, 3), "vs_fp8_prod": round(t8 / t, 3),
               "maxdiff_vs_fp8_prod": round(err, 5)}
        rows.append(row); print(json.dumps(row), flush=True)
    best = sorted((r for r in rows if r["maxdiff_vs_fp8_prod"] < 0.02), key=lambda r: r["us"])[:5]
    print("SUMMARY " + json.dumps({"shape": shape, "bf16_prod_us": round(t16, 1), "fp8_prod_us": round(t8, 1),
                                   "fp8_prod_vs_bf16": round(t16 / t8, 3), "best": best}), flush=True)
    del s, phys16, phys8, b; torch.cuda.empty_cache()
