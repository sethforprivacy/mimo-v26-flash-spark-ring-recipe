#!/usr/bin/env python3
"""Decode/verify (3D split-KV) microbenchmark for vLLM TRITON_ATTN_DIFFKV on GB10, MiMo-V2.6 TP=4 shapes.

  SHAPES="ga:4:100000:1" CONFIGS="16:16:0:0:16,64:16:4:2:32" python3 attn_bench3d.py
  shape  = <ga|swa>:<query_len per seq>:<seq_len>:<num_seqs>
  config = BLOCK_M:TILE:num_warps:num_stages:NUM_SEGMENTS   (0 = Triton default); 2D:<BLOCK_M>:<TILE>:<w>:<s> = 2D launch
Reference = the stock launch (BLOCK_M 16, TILE 16, 16 segments, 3D) — what vLLM runs for decode-only batches.
"""
import json, os
import torch
from vllm.triton_utils import triton
import vllm.v1.attention.ops.triton_unified_attention_diffkv as m
from attn_bench import make, launch as launch2d, timeit, QK, V, BS

K = m.kernel_unified_attention_diffkv
R = m.kernel_reduce_segments_diffkv


def launch3d(s, bufs, block_m, tile, warps, stages, nseg):
    q, k, v, out = s["q"], s["k"], s["v"], s["out"]
    nq, nkv = s["nq"], s["nkv"]
    qpk = nq // nkv
    BLOCK_M = block_m
    BLOCK_Q = BLOCK_M // qpk
    total_num_q_blocks = q.shape[0] // BLOCK_Q + s["nseq"]
    so, sm, se = bufs[nseg]
    kw = {}
    if warps:
        kw["num_warps"] = warps
    if stages:
        kw["num_stages"] = stages
    K[(total_num_q_blocks, nkv, nseg)](
        output_ptr=out, segm_output_ptr=so, segm_max_ptr=sm, segm_expsum_ptr=se,
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=s["sinks"], block_tables_ptr=s["bt"],
        seq_lens_ptr=s["seqused"], alibi_slopes_ptr=None, scale=QK ** -0.5, softcap=0.0,
        num_query_heads=nq, num_queries_per_kv=qpk, block_table_stride=s["bt"].stride(0),
        query_stride_0=q.stride(0), query_stride_1=q.stride(1), output_stride_0=out.stride(0),
        output_stride_1=out.stride(1), BLOCK_SIZE=BS, TILE_SIZE=tile, HEAD_SIZE_QK=QK,
        HEAD_SIZE_QK_PADDED=triton.next_power_of_2(QK), HEAD_SIZE_V=V, HEAD_SIZE_V_PADDED=triton.next_power_of_2(V),
        USE_ALIBI_SLOPES=False, USE_ALIBI_SQRT=False, USE_SOFTCAP=False, USE_SINKS=s["sinks"] is not None,
        SLIDING_WINDOW=s["window"],
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3), stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3), query_start_len_ptr=s["cu"],
        BLOCK_Q=BLOCK_Q, num_seqs=s["nseq"], BLOCK_M=BLOCK_M, NUM_SEGMENTS_PER_SEQ=nseg, IS_3D=True, **kw)
    R[(q.shape[0], nq)](
        output_ptr=out, segm_output_ptr=so, segm_max_ptr=sm, segm_expsum_ptr=se, seq_lens_ptr=s["seqused"],
        num_seqs=s["nseq"], num_query_heads=nq, output_stride_0=out.stride(0), output_stride_1=out.stride(1),
        TILE_SIZE=tile, HEAD_SIZE_V=V, HEAD_SIZE_V_PADDED=triton.next_power_of_2(V), query_start_len_ptr=s["cu"],
        BLOCK_Q=BLOCK_Q, NUM_SEGMENTS_PER_SEQ=nseg)


def main():
    shapes = [x.split(":") for x in os.environ.get("SHAPES", "ga:4:100000:1").split(",")]
    cfgs = os.environ.get("CONFIGS", "16:16:0:0:16").split(",")
    iters = int(os.environ.get("ITERS", "20"))
    res = []
    for sh in shapes:
        kind, qlen, slen, nseq = sh[0], int(sh[1]), int(sh[2]), int(sh[3])
        s = make(kind, qlen, slen, nseq)
        ntok = qlen * nseq
        bufs = {n: (torch.empty(ntok, s["nq"], n, triton.next_power_of_2(V), device="cuda", dtype=torch.float32),
                    torch.empty(ntok, s["nq"], n, device="cuda", dtype=torch.float32),
                    torch.empty(ntok, s["nq"], n, device="cuda", dtype=torch.float32)) for n in (1, 2, 4, 8, 16, 32, 64)}
        launch3d(s, bufs, 16, 16, 0, 0, 16); torch.cuda.synchronize()
        ref = s["out"].clone()
        base = None
        kv_bytes = nseq * slen * s["nkv"] * (QK + V) * 2 if kind == "ga" else nseq * min(slen, 128 + qlen) * s["nkv"] * (QK + V) * 2
        for c in cfgs:
            p = c.split(":")
            try:
                s["out"].zero_()
                if p[0] == "2D":
                    bm, tl_, w, st = (int(x) for x in p[1:5])
                    ms = timeit(lambda: launch2d(s, bm, tl_, w, st), iters)
                else:
                    bm, tl_, w, st, ns = (int(x) for x in p)
                    ms = timeit(lambda: launch3d(s, bufs, bm, tl_, w, st, ns), iters)
                err = (s["out"].float() - ref.float()).abs().max().item()
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"shape": sh, "cfg": c, "error": repr(e)[:160]}), flush=True)
                continue
            if base is None:
                base = ms
            row = {"shape": sh, "cfg": c, "us": round(ms * 1e3, 1), "speedup": round(base / ms, 3),
                   "kv_GBps": round(kv_bytes / (ms * 1e-3) / 1e9, 1), "maxerr": round(err, 4)}
            res.append(row)
            print(json.dumps(row), flush=True)
        del s, bufs
        torch.cuda.empty_cache()
    json.dump(res, open(os.environ.get("OUT", "/tmp/attn_bench3d.json"), "w"))


if __name__ == "__main__":
    main()
