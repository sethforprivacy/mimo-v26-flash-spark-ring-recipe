#!/usr/bin/env bash
# llm-inference-bench (local-inference-lab/llm-inference-bench, pinned a50025a3, update check off) against the ring,
# run on rank 0's host as a second harness beside RigMark: sustained, duration-based decode over a concurrency x
# context matrix (exact stream-usage token counts, loop guard, effective-concurrency detection) plus scout prefill.
# The client runs in a throwaway container of the serving image (httpx / rich / psutil present), pinned to the
# A725 cores 0-4,10-14 so it does not compete with the engine; the key file is read inside the container.
#   lib-bench.sh <label> <keyfile>      results: $OUT_ROOT/<label>/lib-bench.json + lib-bench.log
#   env: LIB_SRC ($HOME/llm-inference-bench, a checkout at a50025a3), LIB_CONC=1,8,16,32 LIB_CTX=0,65536,131072
#        LIB_DURATION=30 PORT=8015 OUT_ROOT ($HOME/mimo26-bench) VLLM_IMAGE CLIENT_CPUS (0-4,10-14)
set -uo pipefail
LABEL=${1:?label}; KEYFILE=${2:?keyfile}; PORT=${PORT:-8015}
SRC=${LIB_SRC:-$HOME/llm-inference-bench}
OUT=${OUT_ROOT:-$HOME/mimo26-bench}/$LABEL; mkdir -p "$OUT"
IMAGE=${VLLM_IMAGE:-myllmbox/mimo-v26-flash-cluster-vllm:v2}
[ "$(git -C "$SRC" rev-parse --short=8 HEAD 2>/dev/null)" = a50025a3 ] || { echo "llm-inference-bench at $SRC is not at a50025a3"; exit 1; }
curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "no engine on :$PORT"; exit 1; }
docker run --rm --network host --cpuset-cpus "${CLIENT_CPUS:-0-4,10-14}" --memory 4g -e LLM_BENCH_NO_UPDATE_CHECK=1 \
  -v "$SRC:/b:ro" -v "$OUT:/out" -v "$(readlink -f "$KEYFILE"):/run/secrets/api-keys:ro" \
  --entrypoint bash "$IMAGE" -c 'exec python3 /b/llm_decode_bench.py --host 127.0.0.1 --port '"$PORT"' \
    --api-key "$(head -1 /run/secrets/api-keys)" --model mimo-v2.6-flash --concurrency '"${LIB_CONC:-1,8,16,32}"' \
    --contexts '"${LIB_CTX:-0,65536,131072}"' --duration '"${LIB_DURATION:-30}"' --display-mode plain \
    --prefill-contexts 8k,64k,128k --no-hw-monitor --output /out/lib-bench.json' > "$OUT/lib-bench.log" 2>&1
rc=$?
echo "rc=$rc"; tail -45 "$OUT/lib-bench.log"
exit "$rc"
