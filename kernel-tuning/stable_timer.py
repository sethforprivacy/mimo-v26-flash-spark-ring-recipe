"""Clock-stable A/B timing for short GPU kernels: warm the GPU, then interleave the arms; every measurement runs
>= MIN_S of back-to-back calls (CUDA events). Returns per-arm medians and min/max over ROUNDS."""
import statistics, time
import torch

MIN_S, ROUNDS = 0.2, 5


def _burn(seconds=1.0):
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            x = (x @ x).clamp_(-1, 1)
        torch.cuda.synchronize()


def _measure(fn, iters):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); e1.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters  # us per call


def ab(arms):
    """arms: {name: fn}. Returns {name: (median_us, min_us, max_us)}."""
    for fn in arms.values():
        fn(); fn()
    torch.cuda.synchronize()
    probe = {n: _measure(f, 3) for n, f in arms.items()}
    iters = {n: max(3, int(MIN_S * 1e6 / max(t, 1.0))) for n, t in probe.items()}
    _burn()
    res = {n: [] for n in arms}
    for _ in range(ROUNDS):
        for n, f in arms.items():
            res[n].append(_measure(f, iters[n]))
    return {n: (statistics.median(v), min(v), max(v)) for n, v in res.items()}
