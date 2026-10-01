#!/usr/bin/env python3
"""Teacher-forced prompt-logprob probe: score fixed text windows on one endpoint and save per-position data, so two
arms (e.g. W4A16 vs W4A8 MoE activations, stock vs tuned attention) can be compared without a teacher. 2026-09-30.

  score:   API=... MODEL=... KEYFILE=... FILES="a.md b.py ..." WINDOW=3000 OUT=arm.json python3 ppl-probe.py
  compare: python3 ppl-probe.py compare base.json arm.json

Uses /v1/completions with prompt_logprobs=1 and max_tokens=1 (temperature 0); a window is the next WINDOW
characters of the concatenated FILES (deterministic). Reports mean NLL/token per arm; compare reports the NLL
delta, top-1 agreement and the share of positions whose token logprob moved by > 0.1 nats."""
import json, math, os, sys, urllib.request


def score():
    api, model = os.environ.get("API", "http://127.0.0.1:8025"), os.environ.get("MODEL", "mimo-v2.6-flash")
    kf = os.environ.get("KEYFILE")
    key = open(kf).readline().strip() if kf else "trial"
    files = os.environ["FILES"].split()
    win, nwin = int(os.environ.get("WINDOW", "3000")), int(os.environ.get("NWIN", "24"))
    corpus = "\n\n".join(open(f, errors="replace").read() for f in files)
    out = []
    for w in range(nwin):
        text = corpus[w * win:(w + 1) * win]
        if len(text) < win // 2:
            break
        body = {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
        req = urllib.request.Request(api + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        j = json.loads(urllib.request.urlopen(req, timeout=600).read())
        pl = j["choices"][0].get("prompt_logprobs") or []
        rows = []
        for pos in pl[1:]:            # position 0 has no logprob
            if not pos:
                rows.append(None); continue
            # each entry: {token_id: {"logprob": x, "rank": r, "decoded_token": s}}; the actual token has rank >= 1,
            # the top-1 alternative rank 1
            items = list(pos.values())
            actual = min(items, key=lambda d: 0 if d.get("rank") is None else 1)  # vLLM lists the actual token first
            actual = items[0]
            top1 = [d for d in items if d.get("rank") == 1]
            rows.append({"lp": actual["logprob"], "rank": actual.get("rank"), "top1": (top1[0].get("decoded_token") if top1 else None)})
        out.append(rows)
        nll = [-r["lp"] for r in rows if r]
        print(json.dumps({"window": w, "tokens": len(nll), "mean_nll": round(sum(nll) / max(1, len(nll)), 4)}), flush=True)
    allnll = [-r["lp"] for rows in out for r in rows if r]
    summary = {"windows": len(out), "positions": len(allnll), "mean_nll": round(sum(allnll) / max(1, len(allnll)), 5),
               "top1_rate": round(sum(1 for rows in out for r in rows if r and r["rank"] == 1) / max(1, len(allnll)), 4)}
    json.dump({"summary": summary, "windows": out}, open(os.environ.get("OUT", "/tmp/ppl.json"), "w"))
    print(json.dumps(summary))


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    n = agree = moved = 0
    d = 0.0
    for wa, wb in zip(A["windows"], B["windows"]):
        for ra, rb in zip(wa, wb):
            if not ra or not rb:
                continue
            n += 1
            d += (-rb["lp"]) - (-ra["lp"])
            agree += (ra["rank"] == 1) == (rb["rank"] == 1)
            moved += abs(ra["lp"] - rb["lp"]) > 0.1
    print(json.dumps({"positions": n, "base_mean_nll": A["summary"]["mean_nll"], "arm_mean_nll": B["summary"]["mean_nll"],
                      "mean_nll_delta": round(d / max(1, n), 5), "argmax_hit_agreement": round(agree / max(1, n), 4),
                      "moved_gt_0.1_nats": round(moved / max(1, n), 4)}))


if __name__ == "__main__":
    compare(sys.argv[2], sys.argv[3]) if len(sys.argv) > 1 and sys.argv[1] == "compare" else score()
