#!/usr/bin/env python3
"""Corrupted-output probe for vllm-project/vllm#46669 (MiMo + MTP/DFlash + async scheduling at concurrency > 1
emits U+FFFD / stray foreign-script tokens; reconfirmed on MiMo-V2.6-Flash 2026-09-23). Written 2026-09-30.

C concurrent streams x ROUNDS, sampled at the lane's defaults (T=1.0, top_p 0.95), thinking off, English-only
tasks with checkable output: (a) prose, (b) a 24-row table of CSS colour names -> #RRGGBB. A response is flagged
when it contains U+FFFD, a C0 control other than \\n/\\t, or characters from scripts the task never calls for
(CJK, kana, hangul, Cyrillic, Arabic, Hebrew, Thai, Devanagari); a colour row is flagged when its hex is not
6 hex digits. Prints one JSON summary line; exit 1 when any response is flagged.

  API=http://127.0.0.1:8025 MODEL=mimo-v2.6-flash KEYFILE=... C=32 ROUNDS=2 MAXTOK=1200 OUT=garbage.json \\
      python3 garbage-probe.py
"""
import json, os, re, threading, time, urllib.request

API = os.environ.get("API", "http://127.0.0.1:8025")
MODEL = os.environ.get("MODEL", "mimo-v2.6-flash")
C = int(os.environ.get("C", "32"))
ROUNDS = int(os.environ.get("ROUNDS", "2"))
MAXTOK = int(os.environ.get("MAXTOK", "1200"))
OUT = os.environ.get("OUT", "/tmp/garbage.json")
KEYFILE = os.environ.get("KEYFILE")
KEY = open(KEYFILE).readline().strip() if KEYFILE else "trial"
TEMP = float(os.environ.get("TEMP", "1.0"))

TOPICS = ["how a CPU cache hierarchy works", "the history of the printing press", "how vaccines train the immune system",
          "why the sky is blue", "how TCP congestion control works", "the water cycle", "how a bicycle stays upright",
          "the causes of the French Revolution", "how compilers optimise loops", "how bees communicate",
          "the life cycle of a star", "how a refrigerator works", "the rules of chess openings",
          "how GPS determines position", "the basics of double-entry bookkeeping", "how rainbows form"]
COLOURS = ("aliceblue antiquewhite aquamarine azure beige bisque blanchedalmond blueviolet burlywood cadetblue "
           "chartreuse chocolate coral cornflowerblue cornsilk crimson darkcyan darkgoldenrod darkkhaki darkorchid "
           "darksalmon deeppink dodgerblue firebrick").split()
BAD = re.compile(r"[�\u0000-\u0008\u000b\u000c\u000e-\u001fЀ-ӿ֐-׿؀-ۿ"
                 r"ऀ-ॿ฀-๿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")
HEXROW = re.compile(r"^\s*\|?\s*`?([a-z]+)`?\s*\|\s*`?(#[^\s|`]*)`?", re.M)


def task(i):
    if i % 2 == 0:
        return "prose", (f"In clear English, write a detailed explanation of {TOPICS[i // 2 % len(TOPICS)]} for a "
                         "curious adult. Use several paragraphs and no headings.")
    return "hex", ("Make a Markdown table with columns Name | Hex giving the standard CSS hex code (#RRGGBB) for each "
                   "of these colours, in this order: " + ", ".join(COLOURS) + ". Output only the table.")


def one(i, rnd, res):
    kind, prompt = task(i + rnd)
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK,
            "temperature": TEMP, "top_p": 0.95, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(API + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            j = json.loads(r.read())
        text = j["choices"][0]["message"].get("content") or ""
        comp = j.get("usage", {}).get("completion_tokens", 0)
        err = None
    except Exception as e:  # noqa: BLE001
        text, comp, err = "", 0, repr(e)[:200]
    bad = sorted({f"U+{ord(ch):04X}" for ch in BAD.findall(text)})
    badhex = []
    if kind == "hex":
        for name, hx in HEXROW.findall(text):
            if name.lower() in ("name",) or set(hx) <= {"#", "-"}:
                continue
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", hx):
                badhex.append(f"{name}:{hx}")
    ctx = []
    for m in BAD.finditer(text):
        ctx.append(text[max(0, m.start() - 30): m.end() + 30].replace("\n", " "))
    res[(rnd, i)] = {"round": rnd, "i": i, "kind": kind, "tokens": comp, "wall_s": round(time.time() - t0, 1),
                     "err": err, "bad_chars": bad, "bad_hex": badhex[:6], "context": ctx[:3]}


summary_rows = []
res = {}
T0 = time.time()
for rnd in range(ROUNDS):
    ths = [threading.Thread(target=one, args=(i, rnd, res)) for i in range(C)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
rows = [res[k] for k in sorted(res)]
flagged = [r for r in rows if r["bad_chars"] or r["bad_hex"] or r["err"]]
out = {"c": C, "rounds": ROUNDS, "maxtok": MAXTOK, "temperature": TEMP, "responses": len(rows),
       "flagged": len(flagged), "errors": sum(1 for r in rows if r["err"]),
       "bad_char_responses": sum(1 for r in rows if r["bad_chars"]),
       "bad_hex_responses": sum(1 for r in rows if r["bad_hex"]),
       "tokens": sum(r["tokens"] for r in rows), "wall_s": round(time.time() - T0, 1),
       "examples": [{k: r[k] for k in ("kind", "bad_chars", "bad_hex", "context", "err")} for r in flagged[:6]]}
json.dump({"summary": out, "rows": rows}, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False))
raise SystemExit(1 if flagged else 0)
