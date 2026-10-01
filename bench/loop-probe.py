#!/usr/bin/env python3
"""Equal-length repetition-loop probe (2026-10-01): is the loop rate per generated token the same on two engines?

llm-inference-bench's decode cells are duration-bound, so a faster engine generates more tokens per stream and its
"write maximally" generations get further into the range where they loop. Here every request has the same prompt
(~CTX_TOKENS of deterministic filler + a long-form instruction, thinking on, T=1.0, unseeded) and the same
max_tokens, so the loop rate per request compares engines at equal length. A request "loops" when its content ends
in an exact periodic tail: period <= 4096 chars, >= 3 full cycles, >= 1024 chars (llm-inference-bench's rule).

  API=http://127.0.0.1:8025 KEYFILE=... N=48 C=8 MAXTOK=3000 CTX_TOKENS=64000 OUT=probe.json python3 loop-probe.py
"""
import concurrent.futures as cf
import json, os, random, time, urllib.request

API = os.environ.get("API", "http://127.0.0.1:8025")
KEY = open(os.environ["KEYFILE"]).readline().strip()
N, C = int(os.environ.get("N", "48")), int(os.environ.get("C", "8"))
MAXTOK, CTX = int(os.environ.get("MAXTOK", "3000")), int(os.environ.get("CTX_TOKENS", "64000"))
OUT = os.environ.get("OUT", "loop-probe.json")
WORDS = ("architecture structure material design engineering history culture city building space light form "
         "function climate stone timber steel glass concrete facade plan section detail scale order rhythm").split()


def filler(tokens):
    rnd = random.Random(1234)  # deterministic: identical prompt on every engine (prefix cache shared per engine)
    out, n = [], 0
    while n < tokens:
        s = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(9, 18))).capitalize() + "."
        out.append(s); n += len(s) // 4
    return " ".join(out)


PROMPT = (filler(CTX) + "\n\nWrite an extremely detailed, comprehensive encyclopedia article about the complete history "
          "of mathematics from ancient Mesopotamia to 2025. Cover every civilization, every major mathematician, every "
          "theorem, proof, and breakthrough. Do not summarize - provide maximum detail on every topic.")


def periodic_tail(text):
    n = len(text)
    for p in range(1, min(4096, n // 3) + 1):
        if text[n - p:] != text[n - 2 * p:n - p]:
            continue
        k = 2
        while (k + 1) * p <= n and text[n - (k + 1) * p:n - k * p] == text[n - p:]:
            k += 1
        if k >= 3 and k * p >= 1024 and text[n - p:].strip():
            return {"period_chars": p, "cycles": k}
    return None


def one(i):
    body = {"model": "mimo-v2.6-flash", "temperature": 1.0, "max_tokens": MAXTOK, "stream": False,
            "messages": [{"role": "user", "content": PROMPT}]}
    req = urllib.request.Request(API + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return {"i": i, "s": round(time.time() - t0, 1), "completion_tokens": d["usage"]["completion_tokens"],
            "finish": d["choices"][0]["finish_reason"], "content_chars": len(content),
            "loop_content": periodic_tail(content), "loop_reasoning": periodic_tail(reasoning)}


rows = []
with cf.ThreadPoolExecutor(C) as ex:
    for r in ex.map(one, range(N)):
        rows.append(r)
loops = [r for r in rows if r["loop_content"] or r["loop_reasoning"]]
summary = {"n": len(rows), "c": C, "max_tokens": MAXTOK, "ctx_tokens_target": CTX,
           "loops": len(loops), "loops_content": sum(1 for r in rows if r["loop_content"]),
           "loops_reasoning": sum(1 for r in rows if r["loop_reasoning"]),
           "hit_max_tokens": sum(1 for r in rows if r["finish"] == "length"),
           "mean_completion_tokens": round(sum(r["completion_tokens"] for r in rows) / max(1, len(rows)), 1)}
json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
print(json.dumps(summary))
