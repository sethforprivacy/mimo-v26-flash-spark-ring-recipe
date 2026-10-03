#!/usr/bin/env python3
"""FP8 KV numerics and speed for the tuned DiffKV kernel (vllm-patches-fp8kv, 2026-10-01), GB10, MiMo TP=4 shapes.

The patched module is loaded from /w/fp8kv_ops.py (copy of vllm-patches-fp8kv/v1/attention/ops/
triton_unified_attention_diffkv.py). Per shape, with the same bf16 queries:
  ref      bf16 cache                                     (what production computes)
  deq      bf16 cache = fp8(cache / s) * s, bf16 path     (exact target of the FP8 kernel)
  fp8      fp8 cache + per-tensor K / V descales           (the new path)
plumbing = max|fp8 - deq| must be bf16 rounding; quant = max|fp8 - ref| is the FP8 KV error itself. Scales 1.0 (the
serving default: the checkpoint has no KV scales) and 0.25 / 0.5 (exercises the descale plumbing)."""
import importlib.util, json, sys
import torch
from attn_bench import make, timeit

spec = importlib.util.spec_from_file_location("fp8kv_ops", "/w/fp8kv_ops.py")
ops = importlib.util.module_from_spec(spec); spec.loader.exec_module(ops)
FP8 = torch.float8_e4m3fn
QK, V = 192, 128


def bufs(nkv, nq, segs):
    rows = (128 // nkv) * 4
    return (torch.empty(rows, nq, segs, 128, device="cuda", dtype=torch.float32),
            torch.empty(rows, nq, segs, device="cuda", dtype=torch.float32),
            torch.empty(rows, nq, segs, device="cuda", dtype=torch.float32))


def call(s, k, v, b, maxq, kd=None, vd=None):
    so, sm, se = b
    ops.unified_attention_diffkv(
        q=s["q"], k=k, v=v, out=s["out"], cu_seqlens_q=s["cu"], seqused_k=s["seqused"],
        softmax_scale=QK ** -0.5, causal=True, window_size=(s["window"] - 1, 0) if s["window"] else (-1, -1),
        block_table=s["bt"], softcap=0.0, max_seqlen_q=maxq, alibi_slopes=None, sinks=s["sinks"],
        seq_threshold_3D=128 // s["nkv"], num_par_softmax_segments=32,
        softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se, k_descale=kd, v_descale=vd)


worst_plumb, worst_rel = 0.0, 0.0
for kind, qlen, slen, nseq in [("ga", 16384, 65536, 1), ("ga", 3072, 100000, 1), ("swa", 16384, 65536, 1),
                              ("ga", 4, 100000, 1), ("ga", 4, 100000, 4), ("ga", 4, 30000, 32),
                              ("swa", 4, 100000, 1), ("swa", 4, 30000, 32), ("ga", 1, 100000, 1), ("ga", 8, 50000, 2)]:
    for sk, sv in ((1.0, 1.0), (0.25, 0.5)):
        s = make(kind, qlen, slen, nseq)
        # the production physical layout [blocks, kv_heads, block, QK+V]; the backend passes .transpose(1, 2) views
        phys16 = torch.cat([s["k"], s["v"]], dim=-1).transpose(1, 2).contiguous()
        scale_vec = torch.cat([torch.full((QK,), sk), torch.full((V,), sv)]).to("cuda", torch.float32)
        phys8 = (phys16.float() / scale_vec).to(FP8)
        physdq = (phys8.float() * scale_vec).to(torch.bfloat16)

        def views(p):
            t = p.transpose(1, 2)
            return t[..., :QK], t[..., QK:]

        k16, v16 = views(phys16)
        k8, v8 = views(phys8)
        kdq, vdq = views(physdq)
        kd = torch.tensor([sk], device="cuda", dtype=torch.float32)
        vd = torch.tensor([sv], device="cuda", dtype=torch.float32)
        b = bufs(s["nkv"], s["nq"], 32)
        call(s, k16, v16, b, qlen); torch.cuda.synchronize(); ref = s["out"].clone()
        call(s, kdq, vdq, b, qlen); torch.cuda.synchronize(); deq = s["out"].clone()
        s["out"].zero_(); call(s, k8, v8, b, qlen, kd, vd); torch.cuda.synchronize(); out8 = s["out"].clone()
        plumb = (out8.float() - deq.float()).abs().max().item()
        quant = (out8.float() - ref.float()).abs().max().item()
        rel = quant / max(ref.float().abs().max().item(), 1e-6)
        worst_plumb, worst_rel = max(worst_plumb, plumb), max(worst_rel, rel)
        row = {"shape": f"{kind}:{qlen}:{slen}:{nseq}", "scales": [sk, sv], "plumbing_maxerr": round(plumb, 5),
               "fp8_vs_bf16_maxerr": round(quant, 4), "rel": round(rel, 4)}
        if sk == 1.0:
            t16 = timeit(lambda: call(s, k16, v16, b, qlen), 10)
            t8 = timeit(lambda: call(s, k8, v8, b, qlen, kd, vd), 10)
            row.update(bf16_us=round(t16 * 1e3, 1), fp8_us=round(t8 * 1e3, 1), speedup=round(t16 / t8, 2))
        print(json.dumps(row), flush=True)
        del s, b, phys16, phys8, physdq; torch.cuda.empty_cache()
print("WORST plumbing", worst_plumb, "WORST fp8-vs-bf16 relative", worst_rel)
sys.exit(0 if worst_plumb < 0.02 else 1)
