#!/usr/bin/env python3
"""Traffic-shaped load: C concurrent chat requests on long prompts (agent / chat shapes).

  API=http://127.0.0.1:8025 MODEL=mimo-v2.6-flash KEYFILE=... OUT=shaped.json MODE=shared|cold|multiturn \
      C=4 ROUNDS=3 PREFIX_TOKENS=12000 SUFFIX_TOKENS=2500 MAXTOK=300 python3 shaped.py

The shared/cold defaults model an agent lane we measured earlier (prompt p50 14.5K tokens, 89 % of prompt tokens
from the prefix cache, mostly 4 running requests, output p50 154 / mean 365 tokens); ab-suite.sh's longctx cell
uses multiturn at ~90K-token prefixes instead (the MiMo lane's own shape).

  cold    every request has a distinct ~PREFIX+SUFFIX-token prompt (no reuse): TTFT is
          cold prefill under contention, decode runs at ~15K context.
  shared  every request shares one PREFIX-token system prompt plus a unique SUFFIX; round 1
          publishes the prefix, later rounds should hit it (usage.prompt_tokens_details).
  multiturn  C independent sessions (own PREFIX-token system prompt each, no cross-session reuse);
          every round is one more turn: the previous prompt + the assistant's answer + a new
          ~TURN_TOKENS user message, i.e. the agent/chat extension pattern. Ideal reuse per turn is
          the whole previous conversation; `reuse` reports cached / (previous prompt + answer).

Per request: TTFT, decode tok/s (completion tokens over the post-first-token time), prompt
and cached tokens. Per round: aggregate output tok/s. Stdlib only. Written 2026-09-23.
"""
import json, os, random, string, threading, time, urllib.request

API = os.environ.get("API", "http://127.0.0.1:8025")
MODEL = os.environ.get("MODEL", "mimo-v2.6-flash")
OUT = os.environ.get("OUT", "/tmp/shaped.json")
MODE = os.environ.get("MODE", "shared")
C = int(os.environ.get("C", "4"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
PREFIX_TOKENS = int(os.environ.get("PREFIX_TOKENS", "12000"))
SUFFIX_TOKENS = int(os.environ.get("SUFFIX_TOKENS", "2500"))
MAXTOK = int(os.environ.get("MAXTOK", "300"))
EFFORT = os.environ.get("EFFORT", "low")
TURN_TOKENS = int(os.environ.get("TURN_TOKENS", "600"))
# 2026-09-30 (MiMo tuning): ROTATE=1 varies the instruction per round (with one fixed instruction the model can
# repeat its previous answer, which MTP drafts at ~100 % acceptance and reads as a fake decode speedup);
# METRICS=1 adds engine-side deltas per round from /metrics: verify steps, tokens/step and ms per step.
ROTATE = os.environ.get("ROTATE", "0") == "1"
METRICS = os.environ.get("METRICS", "0") == "1"
TASKS = ["Summarise the operational risks described above in about 200 words.",
         "List the five components mentioned most often above and explain the role of each in two sentences.",
         "Write a short incident report (impact, timeline, root cause, follow-ups) based on the text above.",
         "Explain to a new on-call engineer what to monitor first, based on the text above, in about 200 words."]


def scrape():
    try:
        txt = urllib.request.urlopen(API + "/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for line in txt.splitlines():
        for k in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total",
                  "vllm:spec_decode_num_draft_tokens_total", "vllm:generation_tokens_total"):
            if line.startswith(k + "{"):
                out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    return out
KEYFILE = os.environ.get("KEYFILE")   # first line = bearer key (keyed lanes); trial ports take any
KEY = open(KEYFILE).readline().strip() if KEYFILE else "trial"

WORDS = ("latency throughput scheduler kernel buffer replica shard cache eviction prefix decode "
         "prefill token vector matrix tensor gradient router expert gate queue lease window page "
         "block stream graph capture replay commit checkpoint ledger quorum heartbeat timeout "
         "backoff jitter packet frame socket channel ring fabric rail port switch cable node "
         "rank worker head master barrier fence epoch batch chunk span slot tail head").split()


def text(tokens, seed):
    """~tokens tokens of plausible but unrepeated prose (≈0.75 words per token)."""
    rng = random.Random(seed)
    out, n = [], 0
    while n < tokens:
        sent = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 16)))
        out.append(sent.capitalize() + ".")
        n += len(sent.split()) * 4 // 3 + 1
    return " ".join(out)


def one(idx, rnd, system, results, history=None):
    nonce = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
    user = (f"Request {rnd}-{idx}-{nonce}. " + text(TURN_TOKENS if history is not None else SUFFIX_TOKENS, f"{nonce}") +
            "\n\n" + (TASKS[rnd % len(TASKS)] if ROTATE else TASKS[0]))
    if history is not None:
        msgs = history + [{"role": "user", "content": user}]
    else:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    content = []
    body = {"model": MODEL, "messages": msgs, "max_tokens": MAXTOK, "temperature": 0.7, "top_p": 0.95,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": json.loads(os.environ["CTK"]) if os.environ.get("CTK") else {"reasoning_effort": EFFORT}}
    req = urllib.request.Request(API + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); ttft = None; usage = None; err = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except Exception:
                    continue
                if j.get("choices"):
                    d = j["choices"][0].get("delta", {})
                    if d.get("content"):
                        content.append(d["content"])
                    if ttft is None and (d.get("content") or d.get("reasoning") or d.get("reasoning_content")):
                        ttft = time.time() - t0
                if j.get("usage"):
                    usage = j["usage"]
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    wall = time.time() - t0
    comp = (usage or {}).get("completion_tokens", 0)
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    results[idx] = {"round": rnd, "idx": idx, "ttft_s": round(ttft, 2) if ttft else None, "wall_s": round(wall, 2),
                    "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": cached,
                    "completion_tokens": comp,
                    "decode_tps": round(comp / max(1e-6, wall - (ttft or 0)), 1) if comp else None, "err": err,
                    "_user": user, "_answer": "".join(content)}


report = {"mode": MODE, "c": C, "rounds": ROUNDS, "prefix_tokens": PREFIX_TOKENS,
          "suffix_tokens": SUFFIX_TOKENS, "maxtok": MAXTOK, "rounds_detail": []}
system = text(PREFIX_TOKENS, "shared-system-prompt-20260923") if MODE == "shared" else None
histories = ([[{"role": "system", "content": text(PREFIX_TOKENS, f"mt-{i}-{random.random()}")}] for i in range(C)]
             if MODE == "multiturn" else None)
prev_total = [None] * C          # multiturn: previous turn's prompt + completion tokens per session
for rnd in range(ROUNDS):
    results = {}
    # cold mode: every request gets its own distinct long prefix
    prompts = [system if MODE == "shared" else text(PREFIX_TOKENS, f"cold-{rnd}-{i}-{random.random()}")
               for i in range(C)]
    ths = []
    m0 = scrape() if METRICS else {}
    T0 = time.time()
    for i in range(C):
        s = prompts[i]
        th = threading.Thread(target=one, args=(i, rnd, s, results),
                              kwargs={"history": histories[i]} if histories is not None else {})
        th.start(); ths.append(th)
    for th in ths:
        th.join()
    wall = time.time() - T0
    m1 = scrape() if METRICS else {}
    rows = [results[i] for i in sorted(results)]
    ok = [r for r in rows if not r["err"]]
    agg = sum(r["completion_tokens"] for r in ok) / wall if wall else 0
    summary = {"round": rnd, "wall_s": round(wall, 2), "agg_output_tps": round(agg, 1), "ok": len(ok),
               "ttft_mean_s": round(sum(r["ttft_s"] or 0 for r in ok) / max(1, len(ok)), 2),
               "ttft_max_s": max((r["ttft_s"] or 0) for r in ok) if ok else None,
               "decode_tps_mean": round(sum(r["decode_tps"] or 0 for r in ok) / max(1, len(ok)), 1),
               "cached_tokens": [r["cached_tokens"] for r in rows], "prompt_tokens": [r["prompt_tokens"] for r in rows],
               "errs": [r["err"] for r in rows if r["err"]], "requests": rows}
    if METRICS and m0 and m1:
        # engine-side, content-independent: verify steps (one per request per engine step), tokens/step, and the
        # mean engine step time over the decode phase (decode wall summed over requests / verify steps)
        d = {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in m1}
        steps = d.get("vllm:spec_decode_num_drafts_total", 0.0)
        gen = d.get("vllm:generation_tokens_total", 0.0)
        dec_wall = sum(max(0.0, r["wall_s"] - (r["ttft_s"] or 0)) for r in ok)
        summary.update({"verify_steps": int(steps), "tok_per_step": round(gen / steps, 3) if steps else None,
                        "accept_rate": round(d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
                                             / d["vllm:spec_decode_num_draft_tokens_total"], 3)
                        if d.get("vllm:spec_decode_num_draft_tokens_total") else None,
                        "step_ms": round(1000 * dec_wall / steps, 2) if steps else None})
    if histories is not None:
        summary["reuse"] = [round(r["cached_tokens"] / prev_total[r["idx"]], 3)
                            if prev_total[r["idx"]] and r["cached_tokens"] is not None else None for r in rows]
        for r in rows:
            if not r["err"]:
                histories[r["idx"]] += [{"role": "user", "content": r["_user"]},
                                        {"role": "assistant", "content": r["_answer"]}]
                prev_total[r["idx"]] = (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
    for r in rows:
        r.pop("_user", None); r.pop("_answer", None)
    report["rounds_detail"].append(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "requests"}), flush=True)
    time.sleep(3)
json.dump(report, open(OUT, "w"), indent=1)
