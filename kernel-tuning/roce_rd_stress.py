#!/usr/bin/env python3
"""Stale-data stress test for the ring-only RoCEnante recursive doubling (2026-10-01).

Question: can a recursive-doubling all-reduce inside a replayed CUDA graph ever return stale or partial data (the
serving loop-guard hits at 64K context were 3 of 10 RoCE runs vs 0 of 8 NCCL runs)? The graph holds N_AR all-reduces
of fixed random inputs at decode sizes (8 KB-1 MB) with compute gaps of varying length between them, like a decode
step; it is replayed REPLAYS times (with occasional host pauses, so the proxies back off to their sleep path). The
inputs never change, so every replay must reproduce the first replay's outputs bit for bit; any difference is a
protocol failure. The first replay is also checked against NCCL (bf16 tolerance) and across ranks (bit-identical).
Env as roce_rd_bench.py plus N_AR, REPLAYS, PAUSE_EVERY, PAUSE_S. Prints RESULT JSON on rank 0."""
import json, os, time
import torch
import torch.distributed as dist

RANK = int(os.environ["RANK"])
N_AR = int(os.environ.get("N_AR", "106"))
REPLAYS = int(os.environ.get("REPLAYS", "3000"))
PAUSE_EVERY = int(os.environ.get("PAUSE_EVERY", "250"))
PAUSE_S = float(os.environ.get("PAUSE_S", "0.2"))
SIZES = [8192, 32768, 131072, 524288, 1048576]
P0 = ("rocep1s0f0", "roceP2p1s0f0")
P1 = ("rocep1s0f1", "roceP2p1s0f1")
HCAS = {0: (P0, P1), 1: (P1, P0), 2: (P0, P1), 3: (P1, P0)}

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dist.init_process_group("nccl", rank=RANK, world_size=4, device_id=dev)
groups = {pair: dist.new_group(list(pair), backend="gloo") for pair in ((0, 1), (2, 3), (0, 3), (1, 2))}
gA = groups[(0, 1)] if RANK in (0, 1) else groups[(2, 3)]
gB = groups[(0, 3)] if RANK in (0, 3) else groups[(1, 2)]

from b12x.comm import roce
from b12x.comm.roce import _preparation
from b12x.preparation import PreparationSession, PreparedCall


def runtime_for(group, hcas, tag):
    rt = roce.AllReduce(exchange_group=group, device=dev, max_size=max(SIZES), max_gather_bytes=1 << 20,
                        hca_names=list(hcas))
    q = roce.query_from_runtime(rt, surface="AllReduce.all_reduce", call={"dtypes": ("bfloat16",)},
                                topology="roce_rdma", peer_hosts=(f"s-{tag}-a", f"s-{tag}-b"))
    plan = roce.plan(q, runtime=rt)
    seed = torch.zeros(8, dtype=torch.bfloat16, device=dev)

    def prepare(state):
        call = _preparation.prepared_call(state, inp=seed)
        return PreparedCall(run=lambda: call.run())

    PreparationSession(device=dev, autotune=False, compile_workers=2).prepare(
        (plan.request(name=f"s-{tag}", prepare_call=prepare),))
    return rt, plan


rtA, planA = runtime_for(gA, HCAS[RANK][0], "A")
rtB, planB = runtime_for(gB, HCAS[RANK][1], "B")
torch.manual_seed(4242 + RANK)
ins = [torch.randn(SIZES[i % len(SIZES)] // 2, dtype=torch.bfloat16, device=dev) for i in range(N_AR)]
mids = [torch.empty_like(x) for x in ins]
outs = [torch.empty_like(x) for x in ins]
# compute gaps: matmuls of a few sizes (~10-400 us), rank-dependent order so ranks arrive skewed like real decode
mats = [torch.randn(n, n, dtype=torch.bfloat16, device=dev) for n in (256, 512, 1024, 1536)]
gap = [((i * 7 + RANK * 3) % len(mats)) for i in range(N_AR)]
sink = torch.zeros(1, device=dev)


def step():
    for i in range(N_AR):
        m = mats[gap[i]]
        sink.add_((m @ m).float().mean() * 0)          # a compute gap of varying length, kept alive
        rtA.all_reduce(ins[i], out=mids[i], plan=planA)
        rtB.all_reduce(mids[i], out=outs[i], plan=planB)


s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        step()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s):
    step()
torch.cuda.synchronize(); dist.barrier()
g.replay(); torch.cuda.synchronize()
ref = [o.clone() for o in outs]
# first replay vs NCCL and across ranks
nccl_err = 0.0
for x, r in zip(ins, ref):
    y = x.clone(); dist.all_reduce(y)
    nccl_err = max(nccl_err, (y.float() - r.float()).abs().max().item())
ident = True
for r in ref[:8]:
    gl = [torch.empty_like(r) for _ in range(4)]; dist.all_gather(gl, r)
    ident = ident and all(torch.equal(gl[0], t) for t in gl[1:])
dist.barrier()
mismatch_replays, first_bad, t0 = 0, None, time.time()
for k in range(1, REPLAYS + 1):
    g.replay()
    if k % 25 == 0 or k == REPLAYS:
        torch.cuda.synchronize()
        bad = [i for i, (o, r) in enumerate(zip(outs, ref)) if not torch.equal(o, r)]
        if bad:
            mismatch_replays += 1
            if first_bad is None:
                i = bad[0]
                first_bad = {"replay": k, "ar": i, "bytes": ins[i].numel() * 2, "n_bad_ars": len(bad),
                             "maxdiff": (outs[i].float() - ref[i].float()).abs().max().item()}
        rtA.check_health(); rtB.check_health()
    if PAUSE_EVERY and k % PAUSE_EVERY == 0:
        time.sleep(PAUSE_S)
torch.cuda.synchronize()
secs = time.time() - t0
tot = torch.tensor([mismatch_replays], device=dev); dist.all_reduce(tot)
res = {"n_ar_per_replay": N_AR, "replays": REPLAYS, "all_reduces": N_AR * REPLAYS * 4, "seconds": round(secs, 1),
       "us_per_replay": round(secs * 1e6 / REPLAYS, 1), "checked_every": 25, "mismatching_checks_all_ranks": int(tot.item()),
       "first_bad_rank0": first_bad, "first_replay_vs_nccl_maxabs": nccl_err, "bit_identical_ranks": ident}
if RANK == 0:
    print("RESULT " + json.dumps(res), flush=True)
dist.barrier()
rtA.close(); rtB.close()
dist.destroy_process_group()
