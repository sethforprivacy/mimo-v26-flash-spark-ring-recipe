#!/usr/bin/env python3
"""Needle-in-a-haystack check for a long-context OpenAI-compatible endpoint.

An advertised context length says nothing about whether the model can actually
retrieve from deep inside one. Adapted in spirit from r0b0tlab's
run-niah-max-context.py, but self-contained: no tokenizers dependency and no
/v1/chat/completions/render, so it runs anywhere with stdlib only.

Filler is varied prose (a repeated identical sentence compresses into a
trivially cacheable pattern and would flatter the result). The needle is a
random code the model cannot guess. Both are drawn from a fresh rng seeded per
run, so no two runs share a prompt: a re-run cannot be served out of the prefix
cache, and the needle is not guessable from a previous run's output. The filler
is rebuilt per depth for the same reason -- one shared haystack lets each depth
prefix-match the previous one, leaving only the first depth cold. The seed
actually used is printed and reported in the JSON, so a run can be replayed
with --seed when you want the same prompt back (e.g. to re-test a FAIL).

usage:  NIAH_API_KEY=<key> niah.py <base_url> <approx_ctx_tokens> [depth_fractions]
        niah.py <base_url> <api_key> <approx_ctx_tokens> [depth_fractions]   # legacy

  --model NAME  served model id (default $NIAH_MODEL, else the id below)
  --seed N      replay one exact prompt; omitted, every run differs

<base_url> is the ORIGIN only. We append /v1/chat/completions ourselves, so a
trailing /v1 yields /v1/v1/... and 404s at every depth.

Prefer the env var: an argv key is visible to any local user via `ps`.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

DEFAULT_MODEL = "mimo-v2.6-flash"

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("base_url", help="ORIGIN only, no /v1 suffix")
ap.add_argument("rest", nargs="*", metavar="ARG",
                help="[api_key] <approx_ctx_tokens> [depth_fractions]")
ap.add_argument("--model", default=os.environ.get("NIAH_MODEL", DEFAULT_MODEL),
                help="served model id (default: $NIAH_MODEL or %(default)s)")
ap.add_argument("--seed", type=int, default=None,
                help="replay one exact prompt; default is a fresh seed per run")
args = ap.parse_args()

BASE = args.base_url.rstrip("/")
_rest = list(args.rest)
KEY = os.environ.get("NIAH_API_KEY", "")
if not KEY:
    KEY, _rest = _rest[0], _rest[1:]        # legacy argv form
TARGET_TOKENS = int(_rest[0])
FRACS = [float(x) for x in (_rest[1] if len(_rest) > 1 else "0.25,0.5,0.9").split(",")]
MODEL = args.model

# A fixed seed would make every run byte-identical -- filler AND needle -- so
# re-runs would be prefix-cache hits and a PASS need not have touched attention
# at depth. Fresh entropy by default; --seed only when a replay is the point.
SEED = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big")
rng = random.Random(SEED)
sys.stdout.reconfigure(line_buffering=True)  # progress visible when redirected

NOUNS = "router switch daemon ledger cipher packet kernel socket buffer index shard replica cursor token lease quorum".split()
VERBS = "reconciles validates rotates flushes replays throttles mirrors audits caches evicts signs verifies".split()
ADJS = "stale nightly regional encrypted ephemeral durable inbound signed pending archived".split()


def sentence(i: int) -> str:
    return (f"Record {i}: the {rng.choice(ADJS)} {rng.choice(NOUNS)} "
            f"{rng.choice(VERBS)} the {rng.choice(ADJS)} {rng.choice(NOUNS)} "
            f"during window {rng.randint(1000, 9999)}.")


def post(path: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ~4 chars/token is the usual English ratio; we report the real count from usage. MiMo's tokenizer makes ~15 %
# more tokens from this filler (a 430000 target is ~500K real tokens), so size targets below the context limit.
target_chars = TARGET_TOKENS * 4
def build_filler():
    """Fresh prose for ONE depth.

    Rebuilt per depth on purpose. A single haystack shared across depths lets
    each depth prefix-match the one before it, so only the first depth would
    measure cold prefill. Measured on 2026-08-27 at 256K, on two independent
    clusters that agreed exactly: 143,360 cached blocks per 3-depth run, and
    depth 90% ran 45% faster than depth 10% (199.2s -> 109.0s) purely from
    reuse. Retrieval was still genuinely tested -- the needle and everything
    after it is unique per depth -- but the prefill timings were not cold.
    """
    out, size, n = [], 0, 0
    while size < target_chars:
        s = sentence(n)
        out.append(s)
        size += len(s) + 1
        n += 1
    return out, size


print(f"model={MODEL}  seed={SEED}"
      f"{'' if args.seed is not None else '  (fresh; pass --seed to replay)'}")
results = []

for frac in FRACS:
    filler, size = build_filler()
    print(f"depth {frac:>5.0%}  built {len(filler)} fresh filler sentences "
          f"(~{size} chars, target ~{TARGET_TOKENS} tokens)")
    code = f"{rng.randint(10000000, 99999999)}"
    needle = (f"\n\nIMPORTANT SECRET VERIFICATION CODE: {code}\n"
              f"Remember this code, it will be requested later.\n\n")
    at = max(1, int(len(filler) * frac))
    body = "\n".join(filler[:at]) + needle + "\n".join(filler[at:])
    prompt = (
        "The following is a long log archive. Read it carefully.\n\n"
        + body
        + "\n\nQuestion: What is the IMPORTANT SECRET VERIFICATION CODE stated in "
          "the archive above? Reply with only the digits, nothing else."
    )
    t0 = time.time()
    try:
        d = post("/v1/chat/completions", {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 256,  # room for a short reasoning preamble; the code is 8 digits
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }, timeout=3600)
    except Exception as e:  # noqa: BLE001
        print(f"depth {frac:>5.0%}  ERROR  {type(e).__name__}: {str(e)[:120]}")
        results.append({"frac": frac, "passed": False, "error": str(e)[:200]})
        continue
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    out = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
    ptok = d["usage"]["prompt_tokens"]
    finish = d["choices"][0].get("finish_reason")
    ok = code in out
    in_reasoning = (not ok) and code in reasoning
    results.append({"frac": frac, "passed": ok, "code_in_reasoning_only": in_reasoning,
                    "prompt_tokens": ptok, "expected": code, "got": out[:40],
                    "finish_reason": finish, "completion_tokens": d["usage"].get("completion_tokens"),
                    "elapsed_s": round(dt, 1)})
    print(f"depth {frac:>5.0%}  {'PASS' if ok else 'FAIL'}  finish={finish}  "
          f"prompt_tokens={ptok:>8}  {dt:6.1f}s  expected={code} got={out[:24]!r}"
          + ("  (code present in reasoning)" if in_reasoning else ""))

passed = sum(1 for r in results if r["passed"])
print(json.dumps({"target_tokens": TARGET_TOKENS, "model": MODEL, "seed": SEED,
                  "passed": passed, "total": len(results),
                  "results": results}))
sys.exit(0 if passed == len(results) else 1)
