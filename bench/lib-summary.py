#!/usr/bin/env python3
"""One row per arm from llm-inference-bench outputs (bench/lib-bench.sh -> <arm>/lib-bench.json), 2026-10-01.

  python3 lib-summary.py <arm-dir> [<arm-dir> ...]

Aggregate decode tok/s per concurrency at each context (the tool's summary_table) and scout prefill tok/s."""
import json, os, sys


def row(path):
    d = json.load(open(os.path.join(path, "lib-bench.json")))
    st, pf = d.get("summary_table", {}), d.get("prefill", {})
    cells = []
    for ctx in sorted(st, key=int):
        conc = st[ctx]
        cells.append(f"ctx {int(ctx) // 1024}K: " + " / ".join(f"C{c} {conc[c]:.0f}" if conc[c] is not None else f"C{c} -" for c in sorted(conc, key=int)))
    pre = " / ".join(f"{int(k) // 1024}K {v.get('tok_per_sec', 0):.0f}" for k, v in sorted(pf.items(), key=lambda x: int(x[0])))
    return f"{os.path.basename(path.rstrip('/'))}: " + " | ".join(cells) + f" | prefill {pre}"


for p in sys.argv[1:]:
    try:
        print(row(p))
    except Exception as e:  # noqa: BLE001
        print(f"{p}: {e}")
