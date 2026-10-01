# SPDX-License-Identifier: Apache-2.0
"""Ring-only RoCEnante all-reduce for TP=4 on the C+D switchless ring (2026-10-01, MIMO_ROCE_RD=1).

b12x's RoCEnante (``b12x.comm.roce``, local-inference-lab/b12x @ e4084d2e, Apache-2.0; mounted read-only and put on
PYTHONPATH by vllm-rank.sh, nothing installed) is a one-shot all-reduce: every rank RDMA-writes its input into
pinned host slots of every peer (GB10 unified memory, no GPUDirect needed), a C proxy thread posts the writes, and
one kernel per collective sums in a fixed rank order. Its 4-rank form needs every peer reachable; on our ring
0-1-2-3-0 the diagonals are not cabled. Recursive doubling needs only the direct links:

  round A: 2-rank runtime with the neighbour over one port   (pairs 0-1, 2-3)
  round B: 2-rank runtime with the neighbour over the other  (pairs 0-3, 1-2), on the round-A sums

so every rank computes (x0 + x1) + (x2 + x3) in the same order: bit-identical outputs across ranks. Eligible
all-reduces (bf16/fp16/fp32, contiguous, 16-byte multiple, <= MIMO_ROCE_RD_MAX bytes, decided from shape/dtype only,
so every rank routes alike) use it, in eager mode and inside CUDA graphs; larger ones stay on NCCL. RoCEnante is
fail-stop: a wait that times out freezes the runtime and ``check_health`` raises (called from the eager logits
all-gather of every step).
"""

import os

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)

# (round-A HCAs, round-B HCAs) per TP rank: both PCIe functions of the port facing that round's neighbour, in the
# same [domain 0, domain 2] order on both ends of a link (C+D cabling: r0 f0 <-> r1 f1, r2 f0 <-> r3 f1,
# r0 f1 <-> r3 f0, r1 f0 <-> r2 f1).
_P0 = ("rocep1s0f0", "roceP2p1s0f0")
_P1 = ("rocep1s0f1", "roceP2p1s0f1")
_HCAS = {0: (_P0, _P1), 1: (_P1, _P0), 2: (_P0, _P1), 3: (_P1, _P0)}
_PAIRS_A = ((0, 1), (2, 3))
_PAIRS_B = ((0, 3), (1, 2))


class MimoRoceRD:
    """Two 2-rank RoCEnante runtimes doing a recursive-doubling all-reduce over a 4-rank TP group."""

    def __init__(self, cpu_group, device: torch.device, max_size: int, shadow_ref=None) -> None:
        self.disabled = True
        # MIMO_ROCE_RD_SHADOW=1 (diagnostic, 2026-10-01): also all-reduce every input through NCCL (shadow_ref) and
        # keep device-side counters of the worst relative difference and of calls differing by > 5 %; logged from
        # check_health every 512 calls. The RoCE result is still the one returned.
        self.shadow_ref = shadow_ref if os.environ.get("MIMO_ROCE_RD_SHADOW", "0") == "1" else None
        self.rank = dist.get_rank(cpu_group)
        world = dist.get_world_size(cpu_group)
        if world != 4:
            logger.warning("MIMO_ROCE_RD needs a 4-rank TP group (got %d); disabled", world)
            return
        from b12x.comm import roce
        from b12x.comm.roce import _preparation
        from b12x.preparation import PreparationSession, PreparedCall

        granks = dist.get_process_group_ranks(cpu_group)
        # every rank of the default group creates every pair group, in the same order
        groups = {
            pair: dist.new_group([granks[pair[0]], granks[pair[1]]], backend="gloo")
            for pair in _PAIRS_A + _PAIRS_B
        }
        g_a = next(groups[p] for p in _PAIRS_A if self.rank in p)
        g_b = next(groups[p] for p in _PAIRS_B if self.rank in p)
        self.device = device
        self.max_size = int(max_size)

        def runtime(group, hcas, tag):
            rt = roce.AllReduce(exchange_group=group, device=device, max_size=self.max_size,
                                max_gather_bytes=1 << 20, hca_names=list(hcas))
            query = roce.query_from_runtime(
                rt, surface="AllReduce.all_reduce", call={"dtypes": ("float16", "bfloat16", "float32")},
                topology="roce_rdma", peer_hosts=(f"mimo-rd-{tag}-0", f"mimo-rd-{tag}-1"))
            plan = roce.plan(query, runtime=rt)
            seeds = [torch.zeros(16 // dt.itemsize, dtype=dt, device=device)
                     for dt in (torch.float16, torch.bfloat16, torch.float32)]

            def prepare(state):
                calls = [_preparation.prepared_call(state, inp=s) for s in seeds]
                return PreparedCall(run=lambda: [c.run() for c in calls])

            PreparationSession(device=device, autotune=False, compile_workers=2).prepare(
                (plan.request(name=f"mimo-rd-{tag}", prepare_call=prepare),))
            return rt, plan

        if self.shadow_ref is not None:
            self.shadow_max = torch.zeros((), dtype=torch.float32, device=device)
            self.shadow_bad = torch.zeros((), dtype=torch.int64, device=device)
            self.shadow_calls = torch.zeros((), dtype=torch.int64, device=device)
            self.shadow_nonfinite = torch.zeros((), dtype=torch.int64, device=device)  # elements finite in one only
            self.shadow_nan_calls = torch.zeros((), dtype=torch.int64, device=device)  # calls with any non-finite
            self._health_calls = 0
        hcas = _HCAS[self.rank]
        self.rt_a, self.plan_a = runtime(g_a, hcas[0], "A")
        self.rt_b, self.plan_b = runtime(g_b, hcas[1], "B")
        self.disabled = False
        logger.info("MIMO_ROCE_RD: ring-only RoCEnante recursive doubling active (rank %d, A %s, B %s, max %d bytes)",
                    self.rank, ",".join(hcas[0]), ",".join(hcas[1]), self.max_size)

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        return not self.disabled and self.rt_a.should_allreduce(inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        mid = torch.empty_like(inp)
        out = torch.empty_like(inp)
        self.rt_a.all_reduce(inp, out=mid, plan=self.plan_a)
        self.rt_b.all_reduce(mid, out=out, plan=self.plan_b)
        if self.shadow_ref is not None:
            ref = self.shadow_ref(inp)
            o, r = out.float(), ref.float()
            fo, fr = torch.isfinite(o), torch.isfinite(r)
            both = fo & fr
            # padded graph rows can hold non-finite garbage in both results; count only RoCE-vs-NCCL disagreements
            self.shadow_nonfinite.add_((fo != fr).sum())
            self.shadow_nan_calls.add_((~both).any().to(torch.int64))
            diff = torch.where(both, (o - r).abs(), torch.zeros_like(o)).amax()
            scale = torch.where(both, r.abs(), torch.zeros_like(r)).amax().clamp_min(1e-3)
            rel = diff / scale
            torch.maximum(self.shadow_max, rel, out=self.shadow_max)
            self.shadow_bad.add_((rel > 0.05).to(torch.int64))
            self.shadow_calls.add_(1)
        return out

    def check_health(self) -> None:
        if not self.disabled:
            self.rt_a.check_health()
            self.rt_b.check_health()
            if self.shadow_ref is not None:
                self._health_calls += 1
                if self._health_calls % 512 == 0:
                    logger.info("MIMO_ROCE_RD_SHADOW rank %d: calls %d bad(>5%%) %d max_rel %.5f nonfinite_mismatch "
                                "%d calls_with_nonfinite %d", self.rank, int(self.shadow_calls.item()),
                                int(self.shadow_bad.item()), float(self.shadow_max.item()),
                                int(self.shadow_nonfinite.item()), int(self.shadow_nan_calls.item()))

    def close(self) -> None:
        if not self.disabled:
            self.disabled = True
            self.rt_a.close()
            self.rt_b.close()


def maybe_create(unique_name: str, cpu_group, device: torch.device, world_size: int, shadow_ref=None):
    if os.environ.get("MIMO_ROCE_RD", "0") != "1" or unique_name.split(":")[0] != "tp" or world_size != 4:
        return None
    return MimoRoceRD(cpu_group, device, int(os.environ.get("MIMO_ROCE_RD_MAX", str(2 << 20))), shadow_ref)
