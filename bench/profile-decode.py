#!/usr/bin/env python3
"""Torch-profile decode steps only (2026-09-30): stream one request per context size, call /start_profile after the
first answer token (prefill done) and /stop_profile after STEPS more chunks. Needs the engine started with
--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=<dir>; every rank writes its own trace there.

  API=... MODEL=... KEYFILE=... CTXS=2000,60000 STEPS=120 python3 profile-decode.py
"""
import json, os, random, time, urllib.request

API = os.environ.get("API", "http://127.0.0.1:8025")
MODEL = os.environ.get("MODEL", "mimo-v2.6-flash")
KEY = open(os.environ["KEYFILE"]).readline().strip() if os.environ.get("KEYFILE") else "trial"
CTXS = [int(x) for x in os.environ.get("CTXS", "2000,60000").split(",")]
STEPS = int(os.environ.get("STEPS", "120"))
H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
WORDS = ("latency throughput scheduler kernel buffer replica shard cache eviction prefix decode prefill token vector "
         "matrix tensor gradient router expert gate queue lease window page block stream graph capture").split()


def post(path, body=None, timeout=120):
    req = urllib.request.Request(API + path, data=json.dumps(body or {}).encode(), headers=H, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


for ctx in CTXS:
    rng = random.Random(ctx)
    text = " ".join(rng.choice(WORDS) for _ in range(int(ctx * 0.75)))
    body = {"model": MODEL, "stream": True, "max_tokens": STEPS * 4 + 200, "temperature": 0.7, "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text + "\n\nWrite a long, detailed essay about operating a "
                                                           "distributed inference cluster."}]}
    req = urllib.request.Request(API + "/v1/chat/completions", data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); started = None; n = 0; stopped = False
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            j = json.loads(line[5:])
            if not (j.get("choices") and j["choices"][0].get("delta", {}).get("content")):
                continue
            if started is None:
                ttft = time.time() - t0
                print(json.dumps({"ctx": ctx, "ttft_s": round(ttft, 2), "start_profile": post("/start_profile")}), flush=True)
                started = time.time()
                continue
            n += 1
            if n == STEPS and not stopped:
                dt = time.time() - started
                print(json.dumps({"ctx": ctx, "chunks": n, "seconds": round(dt, 2),
                                  "stop_profile": post("/stop_profile", timeout=600)}), flush=True)
                stopped = True
    time.sleep(5)
