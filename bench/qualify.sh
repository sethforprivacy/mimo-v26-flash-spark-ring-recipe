#!/usr/bin/env bash
# Production qualification of a MiMo-V2.6-Flash ring (what each production profile passed before promotion).
#   qualify.sh <out-dir> <keyfile>        run on rank 0's host; keyfile = the API key file (one key per line)
#   env: URL (default http://127.0.0.1:8015), MODEL (mimo-v2.6-flash), SKIP_KEYS=1 (skip step 1, e.g. on a test port)
# 1 every key + 401 without / with a bad key, /v1/models   2 gates (canary, tool turns) + agent loop
# 3 needles at ~500K tokens (10 / 50 / 90 % depth) + ~287K (50 %)   4 concurrent long prompts 2 x ~247K and 4 x ~123K
# 5 an image request.   Exit 0 iff every step passed.
set -uo pipefail
OUT=${1:?out-dir}; KEYFILE=${2:?keyfile}; mkdir -p "$OUT"
URL=${URL:-http://127.0.0.1:8015}; M=${MODEL:-mimo-v2.6-flash}; HERE=$(cd "$(dirname "$0")" && pwd)
export NIAH_API_KEY="$(head -1 "$KEYFILE")"
fail=0; log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/qual.log"; }
# 1. keys (the key file never leaves this host; no key is printed)
if [ "${SKIP_KEYS:-0}" != 1 ]; then
python3 - "$URL" "$M" "$KEYFILE" <<'PY' | tee -a "$OUT/qual.log"
import json, sys, urllib.request, urllib.error
url, model, keyfile = sys.argv[1:4]
keys = [k.strip() for k in open(keyfile) if k.strip()]
def req(key, body=None, path="/v1/chat/completions"):
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = "Bearer " + key
    r = urllib.request.Request(url + path, data=json.dumps(body).encode() if body else None, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=120) as x: return x.status, json.loads(x.read())
    except urllib.error.HTTPError as e: return e.code, {}
body = {"model": model, "max_tokens": 16, "temperature": 0, "messages": [{"role": "user", "content": "What is 17*23? Answer with just the number."}], "chat_template_kwargs": {"enable_thinking": False}}
ok = 0
for k in keys:
    s, d = req(k, body); c = (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    ok += s == 200 and "391" in c
print(f"keys: {ok}/{len(keys)} accepted with 391")
print("no key:", req(None, body)[0], " bad key:", req("invalid-key-for-the-401-check", body)[0])
s, d = req(keys[0], None, "/v1/models"); print("models:", s, [(m["id"], m.get("max_model_len")) for m in d.get("data", [])])
PY
grep -qE "keys: ([0-9]+)/\1 accepted" "$OUT/qual.log" && grep -q "no key: 401  bad key: 401" "$OUT/qual.log" || { log "FAIL keys/401"; fail=1; }
fi
# 2. gates + agent-loop (ab-suite cells)
OUT_ROOT="$OUT" BIN="$HERE" RM=/nonexistent bash "$HERE/ab-suite.sh" gates-agent "$URL" "$M" "$KEYFILE" gates,agent > /dev/null 2>&1
grep -E "canary|tool-turn|agent-loop" "$OUT/gates-agent/suite.log" | cut -c1-200 | tee -a "$OUT/qual.log"
grep -q "canary: PASS" "$OUT/gates-agent/suite.log" && ! grep -q "\[FAIL\]" "$OUT/gates-agent/tool-turn.log" && grep -q "rc=0" <(grep agent-loop "$OUT/gates-agent/suite.log") || { log "FAIL gates/agent"; fail=1; }
# 3. needles. niah.py sizes by ~4 chars/token; MiMo makes ~15 % more tokens, so 430000 ~ 500K real and 250000 ~ 287K
python3 "$HERE/niah.py" "$URL" 430000 0.1,0.5,0.9 --model "$M" > "$OUT/niah-430k.log" 2>&1
python3 "$HERE/niah.py" "$URL" 250000 0.5 --model "$M" > "$OUT/niah-250k.log" 2>&1
grep -hE "^depth .*(PASS|FAIL|ERROR)" "$OUT"/niah-*.log | tee -a "$OUT/qual.log"
[ "$(grep -hcE '^depth .*PASS' "$OUT"/niah-*.log | paste -sd+ - | bc)" = 4 ] || { log "FAIL needles"; fail=1; }
# 4. concurrent long prompts
for n in 2 4; do
  ctx=$(( n == 2 ? 247000 : 123000 )); pids=()
  for i in $(seq "$n"); do python3 "$HERE/niah.py" "$URL" "$ctx" 0.5 --model "$M" > "$OUT/conc$n-$i.log" 2>&1 & pids+=($!); done
  t0=$(date +%s); for p in "${pids[@]}"; do wait "$p"; done
  pass=$(grep -hcE '^depth .*PASS' "$OUT"/conc$n-*.log | paste -sd+ - | bc)
  log "concurrent $n x ~$ctx: $pass/$n PASS in $(( $(date +%s) - t0 )) s"; [ "$pass" = "$n" ] || fail=1
done
# 5. image
python3 - "$URL" "$M" <<'PY' | tee -a "$OUT/qual.log"
import base64, json, os, struct, sys, urllib.request, zlib
url, model = sys.argv[1:3]
key = os.environ["NIAH_API_KEY"]  # from the environment, not argv (argv is visible in ps)
w, h = 64, 64  # a solid red square PNG
raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
body = {"model": model, "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                                                  {"type": "text", "text": "What single colour fills this image? One word."}]}]}
r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
try:
    c = json.loads(urllib.request.urlopen(r, timeout=300).read())["choices"][0]["message"]["content"]
    print("image:", "PASS" if "red" in c.lower() else "FAIL", repr(c))
except Exception as e:
    print("image: FAIL", repr(e)[:200])
PY
grep -q "image: PASS" "$OUT/qual.log" || { log "FAIL image"; fail=1; }
log "=== qualification $([ $fail = 0 ] && echo PASS || echo FAILED) ==="
exit $fail
