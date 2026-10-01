#!/usr/bin/env python3
"""One markdown table of the tuning arms (2026-09-30): the same metrics from every arm dir that has them.

  python3 compare-arms.py <results-root> [arm ...]      (default: every subdir with a suite.log, sorted)
Columns: RigMark single-stream code/prose/structured (median), short-code aggregate C1/C4/C16, sampled-prose
ladder C1/C8/C32 (mean of reps), longctx C1 hot TTFT + decode, C4 hot TTFT + per-stream decode (rounds 1-2),
cold prefill 8K/64K/128K/250K (mean of reps), garbage flagged/responses, agent-loop finished/dup."""
import glob, json, os, sys


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def f(x, nd=0):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def arm_row(d):
    r = {}
    c = load(os.path.join(d, "code.json"))
    if c:
        for w in ("code", "prose", "structured"):
            r[w] = c["decode"][w]["decode_tokens_per_second"]["median"]
        for k in ("1", "4", "16"):
            v = c["concurrency"].get(k)
            r[f"sc{k}"] = v["aggregate_end_to_end_tokens_per_second"]["median"] if v else None
    lad = [load(p) for p in sorted(glob.glob(os.path.join(d, "ladder-*.json")))]
    lad = [x for x in lad if x]
    for cc in (1, 8, 32):
        vals = [row["agg_tps"] for rep in lad for row in rep if row["c"] == cc]
        r[f"lad{cc}"] = sum(vals) / len(vals) if vals else None
    for cc in (1, 4):
        lc = load(os.path.join(d, f"longctx-c{cc}.json"))
        if lc:
            hot = [x for x in lc["rounds_detail"] if x["round"] > 0]
            r[f"lc{cc}_ttft"] = sum(x["ttft_mean_s"] for x in hot) / len(hot)
            r[f"lc{cc}_dec"] = sum(x["decode_tps_mean"] for x in hot) / len(hot)
            r[f"lc{cc}_cold"] = lc["rounds_detail"][0]["ttft_mean_s"]
    pf = load(os.path.join(d, "prefill.json"))
    if pf:
        for tgt, name in ((9800, "p8k"), (78000, "p64k"), (157000, "p128k"), (300000, "p250k")):
            vals = [x["prefill_tps"] for x in pf if x["target"] == tgt]
            r[name] = sum(vals) / len(vals) if vals else None
    g = load(os.path.join(d, "garbage.json"))
    if g:
        r["garbage"] = f"{g['summary']['flagged']}/{g['summary']['responses']}"
    try:
        a = json.loads(open(os.path.join(d, "agent-loop.log")).read().strip().splitlines()[-1])
        r["agent"] = f"{a['finished']}/{a['episodes']} d{a['duplicates']}"
    except Exception:
        pass
    return r


def main():
    root = sys.argv[1]
    arms = sys.argv[2:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(root, "*/suite.log")))
    cols = [("code", 1), ("prose", 1), ("structured", 1), ("sc1", 0), ("sc4", 0), ("sc16", 0), ("lad1", 1), ("lad8", 0),
            ("lad32", 0), ("lc1_ttft", 2), ("lc1_dec", 1), ("lc4_ttft", 2), ("lc4_dec", 1), ("lc1_cold", 1), ("lc4_cold", 1),
            ("p8k", 0), ("p64k", 0), ("p128k", 0), ("p250k", 0), ("garbage", 0), ("agent", 0)]
    print("| arm | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for a in arms:
        r = arm_row(os.path.join(root, a))
        print(f"| {a} | " + " | ".join(f(r.get(c), nd) for c, nd in cols) + " |")


if __name__ == "__main__":
    main()
