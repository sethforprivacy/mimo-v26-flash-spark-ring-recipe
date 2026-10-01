#!/usr/bin/env python3
"""Stock vs patched unified_attention_diffkv(): same inputs, compare outputs and time (GB10, MiMo TP=4 shapes).
Patched module is loaded from $PATCHED_OPS (default /w/patched_ops.py: a copy of
overlays/vllm-patches-attn2/v1/attention/ops/triton_unified_attention_diffkv.py); buffers mimic the backend (rows = 128/nkv * 4, segments 16 vs 32)."""
import importlib.util, json, os, sys
import torch
from vllm.triton_utils import triton
import vllm.v1.attention.ops.triton_unified_attention_diffkv as stock
from attn_bench import make, timeit

spec = importlib.util.spec_from_file_location("patched_ops", os.environ.get("PATCHED_OPS", "/w/patched_ops.py"))
patched = importlib.util.module_from_spec(spec); spec.loader.exec_module(patched)
print("tune:", patched._MIMO_TUNE, patched._MIMO_PF_FULL, patched._MIMO_DEC_FULL, patched._MIMO_DEC_SWA, patched._MIMO_SEGS)


def bufs(nkv, nq, segs):
    rows = (128 // nkv) * 4
    return (torch.empty(rows, nq, segs, 128, device="cuda", dtype=torch.float32),
            torch.empty(rows, nq, segs, device="cuda", dtype=torch.float32),
            torch.empty(rows, nq, segs, device="cuda", dtype=torch.float32))


def call(mod, s, segs, b, maxq):
    so, sm, se = b
    mod.unified_attention_diffkv(
        q=s["q"], k=s["k"], v=s["v"], out=s["out"], cu_seqlens_q=s["cu"], seqused_k=s["seqused"],
        softmax_scale=192 ** -0.5, causal=True, window_size=(s["window"] - 1, 0) if s["window"] else (-1, -1),
        block_table=s["bt"], softcap=0.0, max_seqlen_q=maxq, alibi_slopes=None, sinks=s["sinks"],
        seq_threshold_3D=128 // s["nkv"], num_par_softmax_segments=segs,
        softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se)


worst = 0.0
for kind, qlen, slen, nseq in [("ga", 16384, 65536, 1), ("ga", 3072, 100000, 1), ("ga", 2048, 50000, 3),
                              ("swa", 16384, 65536, 1), ("swa", 2048, 50000, 3),
                              ("ga", 4, 100000, 1), ("ga", 4, 100000, 4), ("ga", 4, 60000, 8), ("ga", 4, 30000, 32),
                              ("swa", 4, 100000, 1), ("swa", 4, 100000, 8), ("swa", 4, 30000, 32),
                              ("ga", 1, 100000, 1), ("swa", 1, 100000, 16), ("ga", 8, 50000, 2)]:
    s = make(kind, qlen, slen, nseq)
    b16, b32 = bufs(s["nkv"], s["nq"], 16), bufs(s["nkv"], s["nq"], 32)
    call(stock, s, 16, b16, qlen); torch.cuda.synchronize(); ref = s["out"].clone()
    s["out"].zero_(); call(patched, s, 32, b32, qlen); torch.cuda.synchronize()
    err = (s["out"].float() - ref.float()).abs().max().item()
    worst = max(worst, err)
    t0 = timeit(lambda: call(stock, s, 16, b16, qlen), 10)
    t1 = timeit(lambda: call(patched, s, 32, b32, qlen), 10)
    print(json.dumps({"shape": f"{kind}:{qlen}:{slen}:{nseq}", "stock_us": round(t0 * 1e3, 1),
                      "patched_us": round(t1 * 1e3, 1), "speedup": round(t0 / t1, 2), "maxerr": round(err, 4)}), flush=True)
    del s, b16, b32; torch.cuda.empty_cache()
print("WORST maxerr", worst)
sys.exit(0 if worst < 0.02 else 1)
