#!/usr/bin/env python3
"""bf16 counterpart of attn_fp8_dec_sweep.py phase 1 (2026-10-03): does the production bf16 spec-verify launch
(DEC_FULL 64:64:4:2, 32 segments) also gain from the FP8 sweep's winners (TILE 32, 64-128 segments) at long context?
Loads /w/bf16_ops.py (copy of vllm-patches-attn2/v1/attention/ops/triton_unified_attention_diffkv.py)."""
import importlib.util, itertools, json
import torch
from attn_bench import make
import stable_timer
from stable_timer import ab

stable_timer.MIN_S, stable_timer.ROUNDS = 0.1, 3
spec = importlib.util.spec_from_file_location("bf16_ops", "/w/bf16_ops.py")
ops = importlib.util.module_from_spec(spec); spec.loader.exec_module(ops)
QK, V, MAXSEG = 192, 128, 128
PROD = ((64, 64, 4, 2), 32)


def bufs(nq, segs, rows):
    return (torch.empty(rows, nq, segs, 128, device="cuda", dtype=torch.float32),
            torch.empty(rows, nq, segs, device="cuda", dtype=torch.float32),
            torch.empty(rows, nq, segs, device="cuda", dtype=torch.float32))


def runner(s, b, maxq, cfg, segs):
    so, sm, se = b

    def fn():
        ops._MIMO_DEC_FULL, ops._MIMO_SEGS = cfg, (segs, segs, 4)
        ops.unified_attention_diffkv(
            q=s["q"], k=s["k"], v=s["v"], out=s["out"], cu_seqlens_q=s["cu"], seqused_k=s["seqused"],
            softmax_scale=QK ** -0.5, causal=True, window_size=(-1, -1), block_table=s["bt"], softcap=0.0,
            max_seqlen_q=maxq, alibi_slopes=None, sinks=None, seq_threshold_3D=128, num_par_softmax_segments=segs,
            softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se)
    return fn


CONFIGS = [((64, t, w, st), sg) for t, w, st, sg in itertools.product((32, 64), (4, 8), (2, 3), (16, 32, 64, 128))]
for qlen, slen, nseq in [(4, 250000, 1), (4, 500000, 1), (4, 1000000, 1), (4, 500000, 2), (4, 100000, 8)]:
    s = make("ga", qlen, slen, nseq)
    b = bufs(s["nq"], MAXSEG, qlen * nseq + 8)
    f0 = runner(s, b, qlen, *PROD)
    f0(); torch.cuda.synchronize(); ref = s["out"].clone()
    t0 = ab({"prod": f0})["prod"][0]
    shape = f"ga:{qlen}:{slen}:{nseq}"
    rows = []
    for cfg, segs in CONFIGS:
        fn = runner(s, b, qlen, cfg, segs)
        try:
            s["out"].zero_(); fn(); torch.cuda.synchronize()
            err = (s["out"].float() - ref.float()).abs().max().item()
            t = ab({"x": fn})["x"][0]
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"shape": shape, "cfg": cfg, "segs": segs, "error": repr(e)[:120]}), flush=True)
            continue
        row = {"shape": shape, "cfg": cfg, "segs": segs, "us": round(t, 1), "vs_prod": round(t0 / t, 3), "maxdiff": round(err, 5)}
        rows.append(row); print(json.dumps(row), flush=True)
    best = sorted((r for r in rows if r["maxdiff"] < 0.02), key=lambda r: r["us"])[:4]
    print("SUMMARY " + json.dumps({"shape": shape, "bf16_prod_us": round(t0, 1),
                                   "prod_GBps": round(slen * nseq * (QK + V) * 2 / 1e9 / (t0 * 1e-6), 1), "best": best}), flush=True)
    del s, b; torch.cuda.empty_cache()
