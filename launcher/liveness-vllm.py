#!/usr/bin/env python3
"""Optional scheduler-liveness shim for a load balancer or health checker (GET :8016/liveness, also /health).

vLLM's own /health only says the API process answers. This shim answers 200 iff rank 0's /health answers 200 AND
the engine is not stalled: while requests are running, the sum of vllm:prompt_tokens_total +
vllm:generation_tokens_total must have advanced within STALL_S seconds (a 512K cold prefill takes about two minutes
on this stack, so the default 600 s cannot trip on a long prefill). Otherwise 503. A poisoned RoCEnante runtime or an
NCCL hang therefore turns into a 503 here. JSON body fields: healthy, reason, running_requests, waiting_requests,
kv_cache_usage, progress_stalled_seconds, sample_age_seconds.

  python3 liveness-vllm.py            (env: API=http://127.0.0.1:8015 PORT=8016 STALL_S=600 SAMPLE_S=10)

Stdlib only; ring-up.sh runs it in a small container of the serving image on rank 0 (LIVENESS=1).
"""
import json
import os
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API = os.environ.get("API", "http://127.0.0.1:8015")
PORT = int(os.environ.get("PORT", "8016"))
STALL_S = float(os.environ.get("STALL_S", "600"))
SAMPLE_S = float(os.environ.get("SAMPLE_S", "10"))

state = {"healthy": False, "reason": "starting", "running_requests": 0.0, "waiting_requests": 0.0,
         "kv_cache_usage": 0.0, "progress_stalled_seconds": 0.0, "sampled_at": 0.0}
lock = threading.Lock()


def metric(text, name):
    total, seen = 0.0, False
    for m in re.finditer(rf"^{re.escape(name)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", text, re.M):
        total += float(m.group(1)); seen = True
    return total if seen else None


def sampler():
    last_progress, last_change = None, time.time()
    while True:
        now = time.time()
        try:
            with urllib.request.urlopen(API + "/health", timeout=5) as r:
                health_ok = r.status == 200
        except Exception:  # noqa: BLE001
            health_ok = False
        text = ""
        try:
            with urllib.request.urlopen(API + "/metrics", timeout=10) as r:
                text = r.read().decode()
        except Exception:  # noqa: BLE001
            pass
        running = metric(text, "vllm:num_requests_running") or 0.0
        waiting = metric(text, "vllm:num_requests_waiting") or 0.0
        kv = metric(text, "vllm:kv_cache_usage_perc") or 0.0
        progress = (metric(text, "vllm:prompt_tokens_total") or 0.0) + (metric(text, "vllm:generation_tokens_total") or 0.0)
        if last_progress is None or progress != last_progress or running == 0:
            last_progress, last_change = progress, now
        stalled = now - last_change
        if not health_ok:
            healthy, reason = False, "api /health not 200"
        elif not text:
            healthy, reason = False, "metrics unavailable"
        elif running > 0 and stalled > STALL_S:
            healthy, reason = False, f"no token progress for {stalled:.0f}s with {running:.0f} running"
        else:
            healthy, reason = True, "ok"
        with lock:
            state.update(healthy=healthy, reason=reason, running_requests=running, waiting_requests=waiting,
                         kv_cache_usage=kv, progress_stalled_seconds=round(stalled if running > 0 else 0.0, 1),
                         sampled_at=now)
        time.sleep(SAMPLE_S)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] not in ("/liveness", "/health"):
            self.send_response(404); self.end_headers(); return
        with lock:
            body = dict(state, schema="ringside-scheduler-liveness/v1",
                        sample_age_seconds=round(time.time() - state["sampled_at"], 1))
        fresh = body["sample_age_seconds"] < 3 * SAMPLE_S + 15
        ok = body["healthy"] and fresh
        if not fresh:
            body.update(healthy=False, reason="sampler stale")
        data = json.dumps(body).encode()
        self.send_response(200 if ok else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # quiet: health checkers poll often
        pass


if __name__ == "__main__":
    threading.Thread(target=sampler, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
