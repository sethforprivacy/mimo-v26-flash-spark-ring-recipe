#!/usr/bin/env python3
"""4-rank all-reduce latency on the four-Spark ring with the lane's NCCL (2026-09-30): the decode step spends ~7 ms in
~106 bf16 all-reduces of 32 KB (K=3 verify: 4 tokens x 4096 x 2 B). Eager back-to-back calls and a CUDA graph of
back-to-back calls (the lane replays full graphs), per message size. Env: RANK, WORLD_SIZE=4, MASTER_ADDR/PORT."""
import json, os, time
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
torch.cuda.set_device(0)
dist.init_process_group("nccl", rank=rank, world_size=int(os.environ.get("WORLD_SIZE", "4")),
                        device_id=torch.device("cuda:0"))
SIZES = [int(x) for x in os.environ.get("SIZES", "8192,32768,131072,524288,1048576").split(",")]
out = {"config": os.environ.get("CONFIG", "?"), "nccl": ".".join(map(str, torch.cuda.nccl.version())), "sizes": {}}
for nbytes in SIZES:
    x = torch.randn(nbytes // 2, device="cuda", dtype=torch.bfloat16)
    for _ in range(50):
        dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    n = 400
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        dist.all_reduce(x)
    b.record(); b.synchronize()
    eager_us = a.elapsed_time(b) * 1e3 / n
    # CUDA graph of 100 back-to-back all-reduces (NCCL supports graph capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            dist.all_reduce(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize(); dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(100):
            dist.all_reduce(x)
    g.replay(); torch.cuda.synchronize(); dist.barrier()
    a.record()
    for _ in range(5):
        g.replay()
    b.record(); b.synchronize()
    graph_us = a.elapsed_time(b) * 1e3 / 500
    out["sizes"][nbytes] = {"eager_us": round(eager_us, 1), "graph_us": round(graph_us, 1)}
    del g
if rank == 0:
    print("RESULT " + json.dumps(out), flush=True)
dist.barrier()
dist.destroy_process_group()
