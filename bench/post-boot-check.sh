#!/usr/bin/env bash
# Post-boot verification of the production ring (what we run after every restore), on rank 0's host.
# Keys + 401s + /v1/models, a vision request, tool turns, the container's profile-derived settings and the RoCE
# runtime on every rank, the KV pool, the liveness shim and the memory picture. Prints PASS/FAIL per line; never
# prints a key.  env: URL (http://127.0.0.1:8015), PUBLIC_URL (optional: also send the canary through your public
# endpoint), RUN_DIR ($HOME/mimo26-run); site facts (API_KEY_FILE, ranks) from launcher/hosts.env.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../launcher/load-hosts.sh
. "$HERE/../launcher/load-hosts.sh"
K=${API_KEY_FILE:?API_KEY_FILE not set (hosts.env)}; URL=${URL:-http://127.0.0.1:8015}; NAME=${CONTAINER:-vllm_mimo26}
python3 - "$K" "$URL" "${PUBLIC_URL:-}" <<'PY'
import base64, json, struct, sys, urllib.request, urllib.error, zlib
keyfile, url, public = sys.argv[1:4]
keys = [k.strip() for k in open(keyfile) if k.strip()]
def req(key, body=None, path="/v1/chat/completions", base=url):
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = "Bearer " + key
    r = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=180) as x: return x.status, json.loads(x.read())
    except urllib.error.HTTPError as e: return e.code, {}
canary = {"model": "mimo-v2.6-flash", "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
          "messages": [{"role": "user", "content": "What is 17*23? Answer with just the number."}]}
ok = sum(1 for k in keys if (lambda s, d: s == 200 and "391" in ((d.get("choices") or [{}])[0].get("message", {}).get("content") or ""))(*req(k, canary)))
print(f"keys: {ok}/{len(keys)} accepted with 391", "PASS" if ok == len(keys) else "FAIL")
nk, bk = req(None, canary)[0], req("invalid-key-for-the-401-check", canary)[0]
print(f"no key {nk} / bad key {bk}", "PASS" if (nk, bk) == (401, 401) else "FAIL")
s, d = req(keys[0], None, "/v1/models")
print("models:", s, [(m["id"], m.get("max_model_len")) for m in d.get("data", [])])
w = h = 64
raw = b"".join(b"\x00" + b"\x00\x00\xff" * w for _ in range(h))  # solid blue
def chunk(t, x): return struct.pack(">I", len(x)) + t + x + struct.pack(">I", zlib.crc32(t + x) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
img = {"model": "mimo-v2.6-flash", "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
       "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                                                 {"type": "text", "text": "What single colour fills this image? One word."}]}]}
s, d = req(keys[0], img); c = (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
print("vision:", repr(c), "PASS" if "blue" in c.lower() else "FAIL")
if public:
    s, d = req(keys[0], canary, base=public); c = (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    print("public endpoint:", s, repr(c), "PASS" if s == 200 and "391" in c else "FAIL")
PY
python3 "$HERE/tool-turn-probe.py" --api "$URL" --model mimo-v2.6-flash --keys-file "$K" 2>&1 | tail -3
for i in 0 1 2 3; do
  if [ "$i" = 0 ]; then sh=(bash -c); else sh=(ssh -n -o BatchMode=yes "$(rank_ssh "$i")"); fi
  "${sh[@]}" 'echo "rank '"$i"': roce=$(docker logs '"$NAME"' 2>&1 | grep -c "MIMO_ROCE_RD: ring-only RoCEnante recursive doubling active") roce_errors=$(docker logs '"$NAME"' 2>&1 | grep -ciE "RoCE (proxy failed|collective on rank)|poisoned|timed out waiting") env=$(docker inspect -f "{{range .Config.Env}}{{println .}}{{end}}" '"$NAME"' | grep -cE "^(MIMO_ROCE_RD=1|MIMO_ROCE_RD_MAX=1048576|NCCL_MIN_NCHANNELS=4|PYTHONPATH=/opt/b12x)$")/4 mem=$(awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo)GiB"'
done
docker inspect -f '{{join .Args " "}}' "$NAME" | tr ' ' '\n' | grep -E -A1 '^--(port|max-model-len|max-num-seqs|kv-cache-memory-bytes)$' | paste - - | tr '\n' ' '; echo
docker logs "$NAME" 2>&1 | grep -E "GPU KV cache size" | tail -1 | cut -c1-160
echo "ready marker: $(cat "${RUN_DIR:-$HOME/mimo26-run}/ready" 2>/dev/null || echo none)"
curl -s -o /dev/null -w "liveness %{http_code}\n" --max-time 10 http://127.0.0.1:8016/liveness
