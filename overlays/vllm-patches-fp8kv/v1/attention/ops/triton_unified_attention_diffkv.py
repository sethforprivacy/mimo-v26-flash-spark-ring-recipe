# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton unified attention with different K/V head dimensions (DiffKV).

This is a slimmed fork of ``triton_unified_attention.py`` for models like
MiMo-V2.5 where the V tensor's head dimension differs from K's.  The KV cache
is the same packed layout used by ``FlashAttentionDiffKVBackend``:

    kv_cache: [num_blocks, block_size, num_kv_heads, head_size_qk + head_size_v]

We slice ``key_cache = kv_cache[..., :head_size_qk]`` and
``value_cache = kv_cache[..., head_size_qk:]`` on the host, so the kernel
takes two cache pointers but with two distinct head sizes.

Both 2D and 3D launches are supported:
  - 2D: one program per (q-block, kv-head); tile-loop walks the full KV
    sequence; final output written directly.  Used for prefill and large
    decode batches.
  - 3D: one program per (q-block, kv-head, segm); each program covers a
    KV slice and writes per-segment partials (max/expsum/output).  A
    follow-up ``kernel_reduce_segments_diffkv`` combines them.  Selected
    for decode-only batches whose 2D grid would under-fill the GPU.
"""

from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    find_seq_idx,
    init_softmax_M,
    resolve_seq_and_query_len,
    softmax_step,
    store_segm_reduce_scalars,
)

logger = init_logger(__name__)

is_batch_invariant = envs.VLLM_BATCH_INVARIANT


@triton.jit
def kernel_unified_attention_diffkv(
    # Output destinations.  In 2D mode we write the final result into
    # ``output_ptr``; in 3D mode we write per-segment partials into
    # ``segm_*`` and ``output_ptr`` is unused (callers may pass any
    # non-null pointer).
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,  # view of packed cache: [..., :head_size_qk]
    value_cache_ptr,  # view of packed cache: [..., head_size_qk:hqk+hv]
    k_descale_ptr,  # FP8 KV (vllm#58128 port): per-tensor K / V scales, read when USE_KV_SCALES
    v_descale_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    scale,
    softcap,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,  # == HEAD_SIZE_QK
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,  # == HEAD_SIZE_V
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_QK: tl.constexpr,
    HEAD_SIZE_QK_PADDED: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    QK_SPLIT_A: tl.constexpr,  # >0: compute Q.K^T as a QK_SPLIT_A-wide + a QK_SPLIT_B-wide dot (no 192->256 pad)
    QK_SPLIT_B: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    USE_KV_SCALES: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    # Strides for both cache views (they share the same packed buffer, so
    # dims 0/1/2 strides match; only the per-head extent differs).
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    # ``IS_3D`` toggles between 2D layout (one program walks the full KV
    # sequence) and 3D layout (split-KV / FlashDecoding-style: per-segm
    # programs write partials, finalized by ``kernel_reduce_segments_diffkv``).
    IS_3D: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d_qk = tl.arange(0, HEAD_SIZE_QK_PADDED)
    offs_d_v = tl.arange(0, HEAD_SIZE_V_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d_qk[None, :]
    )

    dim_mask_qk = tl.where(offs_d_qk < HEAD_SIZE_QK, 1, 0).to(tl.int1)
    dim_mask_v = tl.where(offs_d_v < HEAD_SIZE_V, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    if QK_SPLIT_A > 0:
        offs_qa = tl.arange(0, QK_SPLIT_A)
        offs_qb = QK_SPLIT_A + tl.arange(0, QK_SPLIT_B)
        q_base = query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1
        Qa = tl.load(query_ptr + q_base + offs_qa[None, :],
                     mask=query_mask_0[:, None] & query_mask_1[:, None], other=0.0)
        Qb = tl.load(query_ptr + q_base + offs_qb[None, :],
                     mask=(offs_qb < HEAD_SIZE_QK)[None, :] & query_mask_0[:, None] & query_mask_1[:, None], other=0.0)
        Q = Qa
    else:
        # Q : (BLOCK_M, HEAD_SIZE_QK_PADDED)
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask_qk[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )
    if USE_KV_SCALES:
        # FP8 KV: run both dots in fp16 - the fp8 -> fp16 upcast of every K / V tile is one cvt per pair on
        # Blackwell, fp8 -> bf16 is not (it made prefill 1.8x slower) - with the query cast once here.
        if QK_SPLIT_A > 0:
            Qa = Qa.to(tl.float16)
            Qb = Qb.to(tl.float16)
            Q = Qa
        else:
            Q = Q.to(tl.float16)

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_V_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_V_PADDED], dtype=tl.float32)
    # FP8 KV: K and V are upcast to the (fp16) query dtype on load; the K scale folds into the softmax scale, the V
    # scale multiplies the accumulator once in the epilogue (2D output and 3D partials alike).
    if USE_KV_SCALES:
        scale = scale * tl.load(k_descale_ptr)
        v_descale = tl.load(v_descale_ptr)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        False,  # USE_MM_PREFIX
        IS_3D,
    )

    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d_v[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        if QK_SPLIT_A > 0:
            k_base = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            Ka = tl.load(key_cache_ptr + k_base + offs_qa[:, None] * stride_k_cache_3,
                         mask=tile_mask[None, :], other=0.0).to(Q.dtype)
            Kb = tl.load(key_cache_ptr + k_base + offs_qb[:, None] * stride_k_cache_3,
                         mask=(offs_qb < HEAD_SIZE_QK)[:, None] & tile_mask[None, :], other=0.0).to(Q.dtype)
        else:
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d_qk[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            # K : (HEAD_SIZE_QK_PADDED, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask_qk[:, None] & tile_mask[None, :],
                other=0.0,
            )
            K = K_load.to(Q.dtype)
        # V : (TILE_SIZE, HEAD_SIZE_V_PADDED)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask_v[None, :] & tile_mask[:, None],
            other=0.0,
        )
        V = V_load.to(Q.dtype)

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            None,  # mm_prefix_range_ptr
            SLIDING_WINDOW,
            False,  # USE_MM_PREFIX
            0,  # MAX_MM_RANGES
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if QK_SPLIT_A > 0:
            qk = tl.dot(Qa, Ka)
            qk = tl.dot(Qb, Kb, qk)
            S += scale * qk
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )
        acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue --------------------------------------------------------
    if USE_KV_SCALES:
        acc = acc * v_descale
    if IS_3D:
        # Store per-segment partials; finalized by reduce_segments_diffkv.
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
            + segm_idx * HEAD_SIZE_V_PADDED
            + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
        )
        tl.store(
            segm_output_ptr + segm_output_offset,
            acc,
            mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        output_offset = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d_v[None, :]
        )
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


@triton.jit
def kernel_reduce_segments_diffkv(
    output_ptr,  # [num_tokens, num_query_heads, head_size_v]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size_v]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,  # == HEAD_SIZE_V
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Combine per-segment partials into the final softmax output.

    Mirrors ``reduce_segments`` from triton_unified_attention.py but
    indexes V's head size (``HEAD_SIZE_V``) instead of the shared one.
    """
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_V_PADDED) < HEAD_SIZE_V, 1, 0).to(
        tl.int1
    )

    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_V_PADDED
        + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_V_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


import os as _mbx_os
_MBX_DIFFKV_3D_MAX_Q = int(_mbx_os.environ.get("MBX_DIFFKV_3D_MAX_Q", "8"))  # MBX 3D spec verify


# Fleet GB10 (sm_121) launch tuning, 2026-09-30 (from our 09-30 tuning record). The stock launcher
# uses BLOCK_M=16 everywhere: at 16 query heads per KV head (MiMo-V2.6 full-attention layers at TP=4) that is
# BLOCK_Q=1, one query token per program, so prefill re-reads the whole KV per token and a K=3 verify step reads
# it 4x. Measured on GB10 against the stock launch (same outputs, max |diff| <= 1e-3):
#   full-attn prefill 2.7-3.2x (128:32:8:3), swa prefill 1.8x, full-attn verify 1.2-2.3x (3D 64:64:4:2 with 32
#   segments up to 4 seqs, 16 above), swa verify 1.25-3.5x (2D 32:32:4:2: 3D splits the whole sequence into
#   segments although only the 128-token window is live). Values are BLOCK_M:TILE:num_warps:num_stages; each
#   can be overridden, and MIMO_DIFFKV_TUNE=0 restores the stock launch.
def _mimo_cfg(name, default):
    v = _mbx_os.environ.get(name, default).split(":")
    return tuple(int(x) for x in v)


_MIMO_TUNE = _mbx_os.environ.get("MIMO_DIFFKV_TUNE", "1") == "1"
_MIMO_PF_FULL = _mimo_cfg("MIMO_DIFFKV_PF_FULL", "128:64:8:2")  # with the QK split (was 128:32:8:3)
_MIMO_PF_SWA = _mimo_cfg("MIMO_DIFFKV_PF_SWA", "128:32:8:2")
_MIMO_DEC_FULL = _mimo_cfg("MIMO_DIFFKV_DEC_FULL", "64:64:4:2")
_MIMO_DEC_SWA = _mimo_cfg("MIMO_DIFFKV_DEC_SWA", "32:32:4:2")
# segments for the full-attention 3D verify: (few seqs, more seqs, "few" cutoff)
_MIMO_SEGS = _mimo_cfg("MIMO_DIFFKV_SEGS", "32:16:4")


def unified_attention_diffkv(
    q,  # [num_tokens, num_query_heads, head_size_qk]
    k,  # view: [num_blocks, block_size, num_kv_heads, head_size_qk]
    v,  # view: [num_blocks, block_size, num_kv_heads, head_size_v]
    out,  # [num_tokens, num_query_heads, head_size_v]
    cu_seqlens_q,
    seqused_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    max_seqlen_q: int = 1,
    alibi_slopes=None,
    sinks=None,
    use_alibi_sqrt=False,
    # 3D / split-KV softmax buffers.  When all four are provided and the
    # batch is decode-only with few sequences, the 3D path is taken.
    seq_threshold_3D: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,  # FP8 KV cache (vllm#58128 port): per-tensor scales, bf16 queries
    v_descale: torch.Tensor | None = None,
):
    assert causal, "Only causal attention is supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size_qk = q.shape[2]
    head_size_v = v.shape[3]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Decide between 2D and 3D launch.  Mirrors the standard launcher:
    # 3D requires preallocated softmax buffers, decode-only batches, and
    # a small number of sequences (otherwise 2D already saturates the SM).
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > _MBX_DIFFKV_3D_MAX_Q  # MBX 3D spec verify (stock: > 1)
        or q.shape[0] > softmax_segm_output.shape[0]
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    # Tile size: 32 for prefill-class kernels.  Decode (small Q) prefers
    # smaller tiles to expose more parallelism along the KV dim.
    tile_size = 32 if not use_3d else (16 if q.element_size() >= 2 else 32)
    launch_kwargs = {}
    num_segments_3d = num_par_softmax_segments

    if _MIMO_TUNE and not is_batch_invariant:
        decode_shaped = max_seqlen_q <= _MBX_DIFFKV_3D_MAX_Q
        if sliding_window_val > 0 and decode_shaped:
            # Only the window is live: one 2D program per (seq, kv head) beats any segment split.
            cfg = _MIMO_DEC_SWA
            use_3d = False
        elif use_3d:
            cfg = _MIMO_DEC_FULL
            few, many, cutoff = _MIMO_SEGS
            num_segments_3d = min(num_par_softmax_segments, few if num_seqs <= cutoff else many)
        elif sliding_window_val > 0:
            cfg = _MIMO_PF_SWA
        elif decode_shaped:
            cfg = None  # full attention, decode batch too large for 3D: keep the stock launch
        else:
            cfg = _MIMO_PF_FULL
        if cfg is not None:
            block_m = max(cfg[0], triton.next_power_of_2(num_queries_per_kv))
            BLOCK_M = block_m
            BLOCK_Q = BLOCK_M // num_queries_per_kv
            total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
            tile_size = cfg[1]
            if cfg[2]:
                launch_kwargs["num_warps"] = cfg[2]
            if cfg[3]:
                launch_kwargs["num_stages"] = cfg[3]

    # QK split (fleet 2026-09-30, vllm-patches-attn2): 192 = 128 + 64 instead of padding Q/K to 256. GB10: full-attn
    # prefill +11-15 % over vllm-patches-attn (with 128:64:8:2 tiles, which only fit once the pad is gone), swa
    # prefill +8 %, decode unchanged; outputs equal to the stock launch within 1e-4. MIMO_DIFFKV_QK_SPLIT=0 disables.
    qk_split_a = qk_split_b = 0
    if _mbx_os.environ.get("MIMO_DIFFKV_QK_SPLIT", "1") == "1" and head_size_qk & (head_size_qk - 1):
        qk_split_a = 1 << (head_size_qk.bit_length() - 1)
        qk_split_b = triton.next_power_of_2(head_size_qk - qk_split_a)

    grid: tuple[Any, ...]
    if use_3d:
        num_par_softmax_segments = num_segments_3d
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        segm_output_ptr = softmax_segm_output
        segm_max_ptr = softmax_segm_max
        segm_expsum_ptr = softmax_segm_expsum
        num_segments = num_par_softmax_segments
    else:
        grid = (total_num_q_blocks, num_kv_heads)
        # 2D never touches the segm tensors but Triton wants a non-null
        # pointer; reuse ``out``.
        segm_output_ptr = out
        segm_max_ptr = out
        segm_expsum_ptr = out
        num_segments = 1

    kernel_unified_attention_diffkv[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        k_descale_ptr=k_descale,
        v_descale_ptr=v_descale,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        scale=softmax_scale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE_QK=head_size_qk,
        HEAD_SIZE_QK_PADDED=triton.next_power_of_2(head_size_qk),
        HEAD_SIZE_V=head_size_v,
        HEAD_SIZE_V_PADDED=triton.next_power_of_2(head_size_v),
        QK_SPLIT_A=qk_split_a,
        QK_SPLIT_B=qk_split_b,
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        USE_KV_SCALES=(k_descale is not None),
        SLIDING_WINDOW=sliding_window_val,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        IS_3D=use_3d,
        **launch_kwargs,
    )

    if use_3d:
        kernel_reduce_segments_diffkv[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            TILE_SIZE=tile_size,
            HEAD_SIZE_V=head_size_v,
            HEAD_SIZE_V_PADDED=triton.next_power_of_2(head_size_v),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
        )
