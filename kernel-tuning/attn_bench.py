#!/usr/bin/env python3
"""Microbenchmark: vLLM TRITON_ATTN_DIFFKV prefill launch configs on GB10 (MiMo-V2.6 shapes, TP=4 per rank).

Runs inside the serving image (myllmbox/mimo-v26-flash-cluster-vllm:v2). Calls the image's own @triton.jit kernel with a launcher copied from
unified_attention_diffkv(), plus overrides for BLOCK_M / TILE_SIZE / num_warps / num_stages, and checks every
config's output against the stock launch.

  SHAPES="ga:16384:65536" CONFIGS="16:32:0:0,64:64:4:2" python3 attn_bench.py
  shape = <ga|swa>:<query_len>:<seq_len>[:<num_seqs>]   (the query is the LAST query_len tokens of seq_len)
  config = BLOCK_M:TILE:num_warps:num_stages  (0 = Triton default)
"""
import itertools, json, os, sys, time
import torch
from vllm.triton_utils import triton
import vllm.v1.attention.ops.triton_unified_attention_diffkv as m

K = m.kernel_unified_attention_diffkv
DEV = "cuda"
BS = int(os.environ.get("BLOCK_SIZE", "32"))
QK, V = 192, 128


def make(kind, qlen, slen, nseq):
    torch.manual_seed(0)
    nq = 16
    nkv = 1 if kind == "ga" else 2
    blocks_per_seq = triton.cdiv(slen, BS)
    nblocks = blocks_per_seq * nseq + 8
    # physical [num_blocks, num_kv_heads, block_size, QK+V]; the backend passes .transpose(1, 2)
    kv = (torch.randn(nblocks, nkv, BS, QK + V, device=DEV, dtype=torch.bfloat16) * 0.5)
    kvt = kv.transpose(1, 2)
    k, v = kvt[..., :QK], kvt[..., QK:QK + V]
    perm = torch.randperm(nblocks - 8, device=DEV)[: blocks_per_seq * nseq]
    bt = perm.view(nseq, blocks_per_seq).to(torch.int32).contiguous()
    q = torch.randn(qlen * nseq, nq, QK, device=DEV, dtype=torch.bfloat16)
    cu = torch.arange(0, nseq + 1, device=DEV, dtype=torch.int32) * qlen
    seqused = torch.full((nseq,), slen, device=DEV, dtype=torch.int32)
    sinks = torch.randn(nq, device=DEV, dtype=torch.float32) if kind == "swa" else None
    window = 128 if kind == "swa" else 0     # sliding_window_val = 1 + window_size[0]
    out = torch.empty(qlen * nseq, nq, V, device=DEV, dtype=torch.bfloat16)
    return dict(q=q, k=k, v=v, out=out, cu=cu, seqused=seqused, bt=bt, sinks=sinks, window=window, nq=nq, nkv=nkv,
                qlen=qlen, nseq=nseq)


def launch(s, block_m=0, tile=0, warps=0, stages=0):
    q, k, v, out = s["q"], s["k"], s["v"], s["out"]
    nq, nkv = s["nq"], s["nkv"]
    qpk = nq // nkv
    BLOCK_M = block_m or (16 if qpk <= 16 else triton.next_power_of_2(qpk))
    BLOCK_Q = BLOCK_M // qpk
    assert BLOCK_Q >= 1, (BLOCK_M, qpk)
    total_num_q_blocks = q.shape[0] // BLOCK_Q + s["nseq"]
    TILE = tile or 32
    kw = {}
    if warps:
        kw["num_warps"] = warps
    if stages:
        kw["num_stages"] = stages
    grid = (total_num_q_blocks, nkv)
    K[grid](
        output_ptr=out, segm_output_ptr=out, segm_max_ptr=out, segm_expsum_ptr=out,
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=s["sinks"], block_tables_ptr=s["bt"],
        seq_lens_ptr=s["seqused"], alibi_slopes_ptr=None, scale=QK ** -0.5, softcap=0.0,
        num_query_heads=nq, num_queries_per_kv=qpk, block_table_stride=s["bt"].stride(0),
        query_stride_0=q.stride(0), query_stride_1=q.stride(1), output_stride_0=out.stride(0),
        output_stride_1=out.stride(1), BLOCK_SIZE=BS, TILE_SIZE=TILE, HEAD_SIZE_QK=QK,
        HEAD_SIZE_QK_PADDED=triton.next_power_of_2(QK), HEAD_SIZE_V=V, HEAD_SIZE_V_PADDED=triton.next_power_of_2(V),
        USE_ALIBI_SLOPES=False, USE_ALIBI_SQRT=False, USE_SOFTCAP=False, USE_SINKS=s["sinks"] is not None,
        SLIDING_WINDOW=s["window"],
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3), stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3), query_start_len_ptr=s["cu"],
        BLOCK_Q=BLOCK_Q, num_seqs=s["nseq"], BLOCK_M=BLOCK_M, NUM_SEGMENTS_PER_SEQ=1, IS_3D=False, **kw)


def timeit(fn, iters):
    fn(); torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        t.append(a.elapsed_time(b))
    t.sort()
    return t[len(t) // 2]


def main():
    shapes = [x.split(":") for x in os.environ.get("SHAPES", "ga:16384:65536").split(",")]
    cfg_env = os.environ.get("CONFIGS", "")
    if cfg_env:
        cfgs = [tuple(int(v) for v in c.split(":")) for c in cfg_env.split(",")]
    else:
        cfgs = [(16, 32, 0, 0)] + [c for c in itertools.product((16, 32, 64, 128), (16, 32, 64, 128), (4, 8), (1, 2, 3))]
    iters = int(os.environ.get("ITERS", "3"))
    res = []
    for sh in shapes:
        kind, qlen, slen = sh[0], int(sh[1]), int(sh[2])
        nseq = int(sh[3]) if len(sh) > 3 else 1
        s = make(kind, qlen, slen, nseq)
        launch(s); torch.cuda.synchronize()
        ref = s["out"].clone()
        base = None
        for c in cfgs:
            try:
                s["out"].zero_()
                ms = timeit(lambda: launch(s, *c), iters)
                err = (s["out"].float() - ref.float()).abs().max().item()
            except Exception as e:  # noqa: BLE001  (smem overflow etc.)
                print(json.dumps({"shape": sh, "cfg": c, "error": repr(e)[:160]}), flush=True)
                continue
            if base is None:
                base = ms
            # attention FLOPs (causal-ish: each of the qlen queries sees ~ its position; window caps swa)
            ctx = min(slen, 128) if kind == "swa" else slen - qlen / 2
            tflops = 2 * qlen * nseq * ctx * s["nq"] * (QK + V) / (ms * 1e-3) / 1e12
            row = {"shape": sh, "cfg": c, "ms": round(ms, 3), "speedup": round(base / ms, 3), "tflops": round(tflops, 1),
                   "maxerr": round(err, 4)}
            res.append(row)
            print(json.dumps(row), flush=True)
        del s
        torch.cuda.empty_cache()
    json.dump(res, open(os.environ.get("OUT", "/tmp/attn_bench.json"), "w"))


if __name__ == "__main__":
    main()
