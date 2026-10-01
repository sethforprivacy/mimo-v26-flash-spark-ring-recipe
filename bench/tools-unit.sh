#!/usr/bin/env bash
# Offline (CPU-only, no network) check of the mimo tool parser: stock image vs the vllm-patches-tools overlay.
# Shows the base bug (leading / trailing newlines dropped from string arguments) and that the overlay fixes it.
#   tools-unit.sh            env: VLLM_IMAGE, MODEL_HOST (the checkpoint dir, for the tokenizer), OVERLAYS
# Exit 0 iff the overlay run passes.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${VLLM_IMAGE:-myllmbox/mimo-v26-flash-cluster-vllm:v2}
M=${MODEL_HOST:-/srv/models/MiMo-V2.6-Flash-MOPD/2479e2d0029eca9a34cc7e7f55a121925f81908e}
PD="${OVERLAYS:-$HERE/overlays}/vllm-patches-tools"; (cd "$PD" && sha256sum -c --quiet SHA256SUMS) || { echo "tools overlay checksum mismatch"; exit 1; }
mounts=(); while read -r _ rel; do mounts+=(-v "$PD/$rel:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro"); done < "$PD/SHA256SUMS"
run() { docker run --rm --memory 8g --network none --entrypoint python3 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
          -v "$M:/models/m:ro" -v "$HERE/bench/test_mimo_tool_parser.py:/t.py:ro" "$@" "$IMAGE" /t.py 2>&1 | grep -v -i warning; }
echo "== stock image"; run; echo "rc=$?"
echo "== with vllm-patches-tools"; run "${mounts[@]}"; rc=$?; echo "rc=$rc"
exit "$rc"
