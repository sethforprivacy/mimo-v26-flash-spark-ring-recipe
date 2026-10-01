#!/usr/bin/env python3
"""Cold prefill probe: unique prompts of ~N tokens, max_tokens 1, one at a time.

  API=http://127.0.0.1:8025 MODEL=mimo-v2.6-flash KEYFILE=... OUT=prefill.json SIZES=8192:2,32768:2,65536:1 \
      python3 prefill.py

Each prompt is fresh random prose (no shared prefix), so the prefix cache cannot help;
prefill tok/s = server-reported prompt_tokens / wall. Stdlib only. Written 2026-09-23.
"""
import json, os, random, time, urllib.request

API = os.environ.get("API", "http://127.0.0.1:8025")
MODEL = os.environ.get("MODEL", "mimo-v2.6-flash")
OUT = os.environ.get("OUT", "/tmp/prefill.json")
KEYFILE = os.environ.get("KEYFILE")   # first line = bearer key (keyed lanes); trial ports take any
KEY = open(KEYFILE).readline().strip() if KEYFILE else "trial"
SIZES = [tuple(int(x) for x in s.split(":")) for s in os.environ.get("SIZES", "8192:2,32768:2,65536:1").split(",")]
WORDS = ("latency throughput scheduler kernel buffer replica shard cache eviction prefix decode "
         "prefill token vector matrix tensor gradient router expert gate queue lease window page "
         "block stream graph capture replay commit checkpoint ledger quorum heartbeat timeout").split()


def prose(tokens):
    rng = random.Random()
    out, n = [], 0
    while n < tokens:
        s = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 16)))
        out.append(s.capitalize() + ".")
        n += len(s.split()) * 4 // 3 + 1
    return " ".join(out)


rows = []
for target, reps in SIZES:
    for rep in range(reps):
        body = {"model": MODEL, "max_tokens": 1, "temperature": 0,
                "messages": [{"role": "user", "content": prose(target) + " Reply with OK."}],
                "chat_template_kwargs": json.loads(os.environ["CTK"]) if os.environ.get("CTK") else {"reasoning_effort": "low"}}
        req = urllib.request.Request(API + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                d = json.load(r)
            wall = time.time() - t0
            pt = d["usage"]["prompt_tokens"]
            cached = (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
            row = {"target": target, "rep": rep, "prompt_tokens": pt, "cached_tokens": cached,
                   "wall_s": round(wall, 2), "prefill_tps": round(pt / wall, 1)}
        except Exception as e:  # noqa: BLE001
            row = {"target": target, "rep": rep, "err": repr(e), "wall_s": round(time.time() - t0, 2)}
        rows.append(row)
        print(json.dumps(row), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
