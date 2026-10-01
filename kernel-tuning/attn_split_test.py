#!/usr/bin/env python3
"""QK-split experiment (2026-09-30): stock launcher vs the tuned overlay vs tuned + MIMO_DIFFKV_QK_SPLIT=1 with wider
tiles (the split frees 25 % of the Q/K tile shared memory). Loads /w/split_ops.py (a copy of
triton_unified_attention_diffkv_qksplit.py); configs are swept by overriding the module's tuning globals."""
import importlib.util, json, os, sys
import torch
import vllm.v1.attention.ops.triton_unified_attention_diffkv as stock
from attn_bench import make, timeit

spec = importlib.util.spec_from_file_location("split_ops", "/w/split_ops.py")
sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)


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


# (label, split, PF_FULL, PF_SWA, DEC_FULL, DEC_SWA)
ARMS = [("tuned", "0", None, None, None, None), ("split", "1", None, None, None, None),
        ("split+pf128:64:8:2", "1", (128, 64, 8, 2), (128, 64, 8, 2), None, None),
        ("split+pf128:64:4:2", "1", (128, 64, 4, 2), (128, 64, 4, 2), None, None),
        ("split+pf256:32:8:2", "1", (256, 32, 8, 2), (256, 32, 8, 2), None, None),
        ("split+pf128:32:8:4", "1", (128, 32, 8, 4), (128, 32, 8, 4), None, None),
        ("split+dec64:128:4:2", "1", None, None, (64, 128, 4, 2), None),
        ("split+dec64:64:4:3", "1", None, None, (64, 64, 4, 3), None)]
defaults = (sp._MIMO_PF_FULL, sp._MIMO_PF_SWA, sp._MIMO_DEC_FULL, sp._MIMO_DEC_SWA)
shapes = [("ga", 16384, 65536, 1), ("ga", 3072, 100000, 1), ("swa", 16384, 65536, 1),
          ("ga", 4, 100000, 1), ("ga", 4, 100000, 4), ("ga", 4, 50000, 16), ("swa", 4, 100000, 8)]
for kind, qlen, slen, nseq in shapes:
    s = make(kind, qlen, slen, nseq)
    b16, b32 = bufs(s["nkv"], s["nq"], 16), bufs(s["nkv"], s["nq"], 32)
    call(stock, s, 16, b16, qlen); torch.cuda.synchronize(); ref = s["out"].clone()
    t_stock = timeit(lambda: call(stock, s, 16, b16, qlen), 5)
    row = {"shape": f"{kind}:{qlen}:{slen}:{nseq}", "stock_us": round(t_stock * 1e3, 1)}
    for label, split, pf, pfs, dec, decs in ARMS:
        if (qlen > 8 and label.startswith("split+dec")) or (qlen <= 8 and label.startswith("split+pf")):
            continue
        os.environ["MIMO_DIFFKV_QK_SPLIT"] = split
        sp._MIMO_PF_FULL, sp._MIMO_PF_SWA, sp._MIMO_DEC_FULL, sp._MIMO_DEC_SWA = (
            pf or defaults[0], pfs or defaults[1], dec or defaults[2], decs or defaults[3])
        try:
            s["out"].zero_(); call(sp, s, 32, b32, qlen); torch.cuda.synchronize()
            err = (s["out"].float() - ref.float()).abs().max().item()
            t = timeit(lambda: call(sp, s, 32, b32, qlen), 5)
            row[label] = f"{t * 1e3:.1f}us x{t_stock / t:.2f} err{err:.4f}"
        except Exception as e:  # noqa: BLE001
            row[label] = "ERR " + repr(e)[:90]
    print(json.dumps(row), flush=True)
    del s, b16, b32; torch.cuda.empty_cache()
