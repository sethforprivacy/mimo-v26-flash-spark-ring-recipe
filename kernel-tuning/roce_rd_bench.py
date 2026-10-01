#!/usr/bin/env python3
"""Ring-only RoCEnante all-reduce on a four-Spark switchless ring vs NCCL (2026-10-01).

b12x's RoCEnante (b12x.comm.roce, local-inference-lab/b12x @ e4084d2e) is a one-shot all-reduce: every rank
RDMA-writes its input to every peer. On our ring 0-1-2-3-0 the diagonal peers (0-2, 1-3) have no direct link, and
SparkRing's answer is ConnectX hardware forwarding (host changes). Recursive doubling needs only direct links:

  round A: pairs 0-1 and 2-3 (each rank's port to its A neighbour), 2-rank one-shot exchange + sum
  round B: pairs 0-3 and 1-2 (the other port), 2-rank one-shot exchange of the round-A sums + sum

Two 2-rank RoCEnante runtimes per rank (explicit HCA lists = both PCIe functions of the port facing that
neighbour). Every rank computes (x0 + x1) + (x2 + x3) in a fixed order, so outputs are bit-identical across ranks.

Measures per-call latency for eager calls and CUDA-graph replay (graphs of back-to-back calls, the decode path),
NCCL ring with the lane's environment as the reference, and checks correctness against NCCL before and after timing.
Env: RANK, WORLD_SIZE=4, MASTER_ADDR, MASTER_PORT, SIZES, GRAPH_OPS, SAMPLES. Prints one RESULT JSON on rank 0."""
import json, os, statistics, time
import torch
import torch.distributed as dist

RANK = int(os.environ["RANK"])
SIZES = [int(x) for x in os.environ.get("SIZES", "8192,32768,131072,524288,1048576,2097152").split(",")]
GRAPH_OPS = int(os.environ.get("GRAPH_OPS", "100"))
SAMPLES = int(os.environ.get("SAMPLES", "15"))
MAX_SIZE = max(SIZES)
# rank -> (round-A HCAs, round-B HCAs); order [PCI domain 0 function, domain 2 function] on both ends of a link.
P0 = ("rocep1s0f0", "roceP2p1s0f0")   # enp1s0f0np0 / enP2p1s0f0np0
P1 = ("rocep1s0f1", "roceP2p1s0f1")   # enp1s0f1np1 / enP2p1s0f1np1
HCAS = {0: (P0, P1), 1: (P1, P0), 2: (P0, P1), 3: (P1, P0)}
#   A links: r0 port 0 <-> r1 port 1;  r2 port 0 <-> r3 port 1   (two point-to-point /24s per cable, one per function)
#   B links: r3 port 0 <-> r0 port 1;  r1 port 0 <-> r2 port 1

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dist.init_process_group("nccl", rank=RANK, world_size=4, device_id=dev)
# gloo exchange groups (a torch NCCL group per pair would cost GBs of unified memory); every rank creates all four
groups = {pair: dist.new_group(list(pair), backend="gloo") for pair in ((0, 1), (2, 3), (0, 3), (1, 2))}
gA = groups[(0, 1)] if RANK in (0, 1) else groups[(2, 3)]
gB = groups[(0, 3)] if RANK in (0, 3) else groups[(1, 2)]

from b12x.comm import roce
from b12x.comm.roce import _preparation
from b12x.preparation import PreparationSession, PreparedCall


def runtime_for(group, hcas, peer_tag):
    rt = roce.AllReduce(exchange_group=group, device=dev, max_size=MAX_SIZE, max_gather_bytes=1 << 20,
                        hca_names=list(hcas))
    q = roce.query_from_runtime(rt, surface="AllReduce.all_reduce", call={"dtypes": ("bfloat16",)},
                                topology="roce_rdma", peer_hosts=(f"pair-{peer_tag}-a", f"pair-{peer_tag}-b"))
    plan = roce.plan(q, runtime=rt)
    seed = torch.zeros(8, dtype=torch.bfloat16, device=dev)

    def prepare(state):
        call = _preparation.prepared_call(state, inp=seed)
        return PreparedCall(run=lambda: call.run())

    PreparationSession(device=dev, autotune=False, compile_workers=2).prepare(
        (plan.request(name=f"roce-{peer_tag}", prepare_call=prepare),))
    return rt, plan


t0 = time.time()
rtA, planA = runtime_for(gA, HCAS[RANK][0], "A")
rtB, planB = runtime_for(gB, HCAS[RANK][1], "B")
setup_s = time.time() - t0
dist.barrier()


def rd(inp, mid, out):
    rtA.all_reduce(inp, out=mid, plan=planA)
    rtB.all_reduce(mid, out=out, plan=planB)
    return out


def timed(fn, ops):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    xs = []
    for _ in range(SAMPLES):
        torch.cuda.synchronize(); dist.barrier()
        a.record(); fn(); b.record(); b.synchronize()
        xs.append(a.elapsed_time(b) * 1e3 / ops)
    return statistics.median(xs)


def graph_of(fn, n):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(n):
            fn()
    torch.cuda.synchronize(); dist.barrier()
    return g


def eager_loop(fn, n):
    def run():
        for _ in range(n):
            fn()
    return run


res = {"b12x": "e4084d2e", "nccl": ".".join(map(str, torch.cuda.nccl.version())), "setup_s": round(setup_s, 1),
       "hcas": {"A": list(HCAS[RANK][0]), "B": list(HCAS[RANK][1])}, "sizes": {}}
for nbytes in SIZES:
    n = nbytes // 2
    torch.manual_seed(1000 + RANK)
    x = torch.randn(n, dtype=torch.bfloat16, device=dev)
    mid, out = torch.empty_like(x), torch.empty_like(x)
    ref = x.clone(); dist.all_reduce(ref)
    rd(x, mid, out); torch.cuda.synchronize()
    err = (out.float() - ref.float()).abs().max().item()
    gathered = [torch.empty_like(out) for _ in range(4)]
    dist.all_gather(gathered, out)
    bit_identical = all(torch.equal(gathered[0], g) for g in gathered[1:])
    # zero inputs for the timed loops: repeated in-place NCCL reductions of random data would overflow
    z = torch.zeros_like(x); zm, zo = torch.empty_like(x), torch.empty_like(x)
    nccl_eager = timed(eager_loop(lambda: dist.all_reduce(z), 200), 200)
    rd_eager = timed(eager_loop(lambda: rd(z, zm, zo), 200), 200)
    gn = graph_of(lambda: dist.all_reduce(z), GRAPH_OPS)
    gr = graph_of(lambda: rd(z, zm, zo), GRAPH_OPS)
    nccl_graph = timed(gn.replay, GRAPH_OPS)
    rd_graph = timed(gr.replay, GRAPH_OPS)
    rtA.check_health(); rtB.check_health()
    rd(x, mid, out); torch.cuda.synchronize()
    err_after = (out.float() - ref.float()).abs().max().item()
    res["sizes"][nbytes] = {"nccl_eager_us": round(nccl_eager, 1), "nccl_graph_us": round(nccl_graph, 1),
                            "rd_eager_us": round(rd_eager, 1), "rd_graph_us": round(rd_graph, 1),
                            "max_abs_err_vs_nccl": err, "err_after": err_after,
                            "ref_absmax": ref.float().abs().max().item(), "bit_identical_ranks": bit_identical}
    del gn, gr
    if RANK == 0:
        print(f"size {nbytes}: {res['sizes'][nbytes]}", flush=True)
res["stats_A"] = {k: v for k, v in rtA.stats().items() if k in ("hcas", "epoch", "error_seq", "ctrl_seq")}
res["stats_B"] = {k: v for k, v in rtB.stats().items() if k in ("hcas", "epoch", "error_seq", "ctrl_seq")}
if RANK == 0:
    print("RESULT " + json.dumps(res), flush=True)
dist.barrier()
rtA.close(); rtB.close()
dist.destroy_process_group()
