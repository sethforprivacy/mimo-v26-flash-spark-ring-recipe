import json, glob, sys
import torch
sys.path.insert(0, "/w")
import benchmark_w8a8_block_fp8 as b
DEFAULT = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 2}
NCOPY = 12
for f in sorted(glob.glob("/w/out/*.json")):
    n = int(f.split("N=")[1].split(",")[0]); k = int(f.split("K=")[1].split(",")[0])
    cfgs = json.load(open(f))
    Bs_ = [torch.randn(n, k, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn) for _ in range(NCOPY)]
    Ss_ = [torch.rand((n + 127) // 128, k // 128, device="cuda") * 1e-2 for _ in range(NCOPY)]
    row = []
    for m in (4, 8, 16, 32, 64, 128):
        A = torch.randn(m, k, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
        As = torch.rand(m, k // 128, device="cuda") * 1e-2
        def run(cfg, iters=60):
            for i in range(NCOPY):
                b.w8a8_block_matmul(A, Bs_[i], As, Ss_[i], [128, 128], cfg, torch.bfloat16)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for it in range(iters):
                i = it % NCOPY
                b.w8a8_block_matmul(A, Bs_[i], As, Ss_[i], [128, 128], cfg, torch.bfloat16)
            e1.record(); e1.synchronize()
            return e0.elapsed_time(e1) * 1e3 / iters
        td, tt = run(DEFAULT), run(cfgs[str(m)])
        row.append(f"M={m}: {td:.1f}->{tt:.1f}us x{td/tt:.2f} ({n*k/(tt*1e-6)/1e9:.0f} GB/s)")
    print(f"N={n} K={k} | " + " | ".join(row), flush=True)
    del Bs_, Ss_; torch.cuda.empty_cache()
