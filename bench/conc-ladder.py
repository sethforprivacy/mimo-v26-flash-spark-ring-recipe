#!/usr/bin/env python3
"""Concurrency ladder against one OpenAI-compatible endpoint (stdlib only).

Usage (on a host that can reach the backend; key file = one key per line):
  API=http://127.0.0.1:8015 MODEL=mimo-v2.6-flash KEYFILE=/path/to/api-keys CTK='{"enable_thinking":false}' \
  RUNGS=1,2,4,8,16 MAXTOK=400 PROMPT_WORDS=120 OUT=/tmp/ladder.json python3 conc-ladder.py
CTK (a JSON object) replaces chat_template_kwargs entirely; ab-suite.sh sets CTK={"enable_thinking":false} for MiMo.
Without CTK, EFFORT is passed as chat_template_kwargs.reasoning_effort (for models that take one).
Per rung c: fire c distinct-prefix chat requests simultaneously (streaming),
record TTFT, completion tokens (from usage), wall; snapshot spec-decode and
scheduler counters from /metrics before/after. Prints a table + JSON."""
import json, os, sys, time, threading, urllib.request, random, string, re

API = os.environ.get("API", "http://127.0.0.1:8015")
MODEL = os.environ.get("MODEL", "mimo-v2.6-flash")
KEY = open(os.environ["KEYFILE"]).readline().strip()
RUNGS = [int(x) for x in os.environ.get("RUNGS", "1,2,4,8,16").split(",")]
MAXTOK = int(os.environ.get("MAXTOK", "300"))
EFFORT = os.environ.get("EFFORT", "low")
PROMPT_WORDS = int(os.environ.get("PROMPT_WORDS", "120"))
OUT = os.environ.get("OUT", "/tmp/conc-ladder.json")

def metrics():
    try:
        t = urllib.request.urlopen(API + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    want = ["spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
            "spec_decode_num_accepted_tokens_total", "num_preemptions_total",
            "generation_tokens_total", "prompt_tokens_total"]
    out = {}
    for line in t.splitlines():
        for w in want:
            if line.startswith("vllm:" + w + "{"):
                out[w] = float(line.rsplit(" ", 1)[1])
    return out

FILLER = ("Consider a distributed systems design where nodes exchange heartbeats over a lossy link. "
          "Describe tradeoffs between timeout length and false-positive failure detection. ").split()

def make_prompt(i):
    nonce = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    words = [random.choice(FILLER) for _ in range(PROMPT_WORDS)]
    return (f"Session nonce {nonce}. Topic {i}: " + " ".join(words) +
            " Write a concise but complete answer in plain prose, roughly 250 words.")

def one(i, results):
    body = {"model": MODEL, "messages": [{"role": "user", "content": make_prompt(i)}],
            "max_tokens": MAXTOK, "temperature": 0.7, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": json.loads(os.environ["CTK"]) if os.environ.get("CTK") else {"reasoning_effort": EFFORT}}
    req = urllib.request.Request(API + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None; err = None; chunks = 0; finish = None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                try: j = json.loads(data)
                except Exception: continue
                if j.get("choices"):
                    d = j["choices"][0].get("delta", {})
                    if (d.get("content") or d.get("reasoning") or d.get("reasoning_content")) and ttft is None:
                        ttft = time.time() - t0
                    chunks += 1
                    if j["choices"][0].get("finish_reason"): finish = j["choices"][0]["finish_reason"]
                if j.get("usage"): usage = j["usage"]
    except Exception as e:
        err = repr(e)
    t1 = time.time()
    results[i] = {"ttft": ttft, "wall": t1 - t0, "usage": usage, "err": err, "chunks": chunks, "finish": finish}

report = []
for c in RUNGS:
    m0 = metrics(); results = {}; ths = []
    T0 = time.time()
    for i in range(c):
        th = threading.Thread(target=one, args=(i, results)); th.start(); ths.append(th)
    for th in ths: th.join()
    wall = time.time() - T0
    m1 = metrics()
    ok = [r for r in results.values() if r["usage"] and not r["err"]]
    comp = sum(r["usage"].get("completion_tokens", 0) for r in ok)
    prompt = sum(r["usage"].get("prompt_tokens", 0) for r in ok)
    per_stream = [r["usage"]["completion_tokens"] / max(1e-6, r["wall"] - (r["ttft"] or 0)) for r in ok]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    drafts = d.get("spec_decode_num_drafts_total", 0); acc = d.get("spec_decode_num_accepted_tokens_total", 0)
    row = {"c": c, "ok": len(ok), "errs": [r["err"] for r in results.values() if r["err"]],
           "wall_s": round(wall, 2), "completion_tokens": comp, "prompt_tokens": prompt,
           "agg_tps": round(comp / wall, 1),
           "per_stream_tps_mean": round(sum(per_stream) / len(per_stream), 1) if per_stream else None,
           "per_stream_tps_min": round(min(per_stream), 1) if per_stream else None,
           "ttft_mean_s": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
           "ttft_max_s": round(max(ttfts), 2) if ttfts else None,
           "tok_per_step": round(1 + acc / drafts, 2) if drafts else None,
           "accept_rate": round(acc / d.get("spec_decode_num_draft_tokens_total", 1), 3) if drafts else None,
           "preemptions": d.get("num_preemptions_total", 0),
           "finish": sorted(set(r["finish"] for r in results.values()))}
    report.append(row)
    print(json.dumps(row), flush=True)
    time.sleep(3)
json.dump(report, open(OUT, "w"), indent=1)
print("\n c | ok | wall | agg t/s | per-stream mean/min | TTFT mean/max | tok/step | accept | preempt")
for r in report:
    print(f"{r['c']:>2} | {r['ok']:>2} | {r['wall_s']:>6} | {r['agg_tps']:>7} | {r['per_stream_tps_mean']}/{r['per_stream_tps_min']} | {r['ttft_mean_s']}/{r['ttft_max_s']} | {r['tok_per_step']} | {r['accept_rate']} | {r['preemptions']}")
