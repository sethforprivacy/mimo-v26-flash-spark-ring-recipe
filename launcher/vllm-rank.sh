#!/usr/bin/env bash
# One vLLM rank of MiMo-V2.6-Flash-MOPD on a four-node DGX Spark (GB10) switchless ring, TP=4.
# Image: myllmbox/mimo-v26-flash-cluster-vllm:v2 (vLLM nightly ddd6fbca + their mbx loader, split-KV spec-verify
# attention, non-chained 3-layer MTP). Their recipe.yaml flags, TP=2 -> 4.
# Fabric: a patched NCCL 2.30.7 (nccl/) preloaded into the container (LD_PRELOAD + VLLM_NCCL_SO_PATH).
#   vllm-rank.sh <rank> {check|up}      site facts from hosts.env, profile knobs from the environment (below);
#                                       rank 0 serves the API. check prints the docker command and starts nothing.
set -euo pipefail
R=${1:?rank 0..3}; MODE=${2:-check}
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=load-hosts.sh
. "$HERE/load-hosts.sh"
case "$R" in 0|1|2|3) ipv=RANK${R}_IP; IP=${!ipv:?$ipv not set (hosts.env)};; *) exit 2;; esac
MASTER_IP=${RANK0_IP:?RANK0_IP not set (hosts.env)}
OVERLAYS=${OVERLAYS:-$(cd "$HERE/.." && pwd)/overlays}
NAME=${CONTAINER:-vllm_mimo26}
# Pinned: myllmbox/mimo-v26-flash-cluster-vllm:v2@sha256:5af49bd0c38923d1d2b3636aa7570b311ff55480f123207259b083cd9afde726,
# whose config digest (the image ID below) is checked before every start. A docker save | docker load relay drops the
# repo digest, so identity is checked by image ID; pull by digest, tag it, set VLLM_IMAGE to the tag if you rename it.
IMAGE=${VLLM_IMAGE:-myllmbox/mimo-v26-flash-cluster-vllm:v2}
VLLM_IMAGE_ID=${VLLM_IMAGE_ID:-sha256:a09549748e0b42d5d89608888b55a826ead04a2d7a7b5318c47dcf8ea0593ace}
MODEL_HOST=${MODEL_HOST:-/srv/models/MiMo-V2.6-Flash-MOPD/2479e2d0029eca9a34cc7e7f55a121925f81908e}
STATE=${STATE:-$HOME/vllm-mimo26-state}
API_KEY_FILE=${API_KEY_FILE:?API_KEY_FILE not set (hosts.env): one API key per line}
PORT=${PORT:-8025}
KV_BYTES=${KV_BYTES:-24000000000}          # per rank; TP=2 recipe: 16 GB = 1.22M tokens, KV/token halves at TP=4
MAX_LEN=${MAX_LEN:-524288}
MAX_SEQS=${MAX_SEQS:-32}
SPEC=${SPEC:-'{"method":"mtp","num_speculative_tokens":'"${SPEC_K:-3}"'}'}   # SPEC_K: MTP depth without passing JSON
# VLLM_CC='<json>': --compilation-config (2026-10-01 tuning: PIECEWISE graphs split at vllm::all_reduce, so the TP
# all-reduces run eagerly between graph pieces). Unset = vLLM's default compilation config, as in production.
LOAD_FORMAT=${LOAD_FORMAT:-mbx}
NCCL_SO=${NCCL_SO:-$HOME/nccl-dual-pci/libnccl.so.2}
HCAS=rocep1s0f0,rocep1s0f1; DUAL_ENV=()
if [ "${NCCL_DUAL:-0}" = 1 ]; then
  NCCL_SO=${NCCL_DUAL_SO:-$HOME/nccl-dual-pci/libnccl.so.2}
  HCAS=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
  DUAL_ENV=(-e NCCL_IB_EXTENDED_IPV4_GIDS=1 -e NCCL_IB_PRESERVE_PCI_DOMAIN=1 -e NCCL_IB_QPS_PER_CONNECTION=1)
fi
fail(){ echo "vllm-rank $R: $*" >&2; exit 1; }
# VLLM_PATCHES=1: mount every file under overlays/vllm-patches/ (paths relative to the vllm package) read-only over
# the image, after checking vllm-patches/SHA256SUMS. 2026-09-29: vllm-project/vllm#58235 (MiMo ViT sink as a null
# softmax logit; flat-colour images were read as "Black").
PATCH_MOUNTS=()
if [ "${VLLM_PATCHES:-0}" = 1 ]; then
  PD="$OVERLAYS/vllm-patches"; (cd "$PD" && sha256sum -c --quiet SHA256SUMS) || fail "vllm-patches checksum mismatch"
  while read -r _ rel; do PATCH_MOUNTS+=(-v "$PD/$rel:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro"); done < "$PD/SHA256SUMS"
fi
# VLLM_PATCHES_EXTRA=dir[,dir]: further overlay dirs beside vllm-patches/ (same layout: paths relative to the vllm
# package, plus SHA256SUMS), mounted the same way; their files must not overlap. 2026-09-30 tuning:
# vllm-patches-attn2 (GB10 TRITON_ATTN_DIFFKV launch tuning + QK split), vllm-patches-fp8cfg (GB10 block-FP8 GEMM
# configs), vllm-patches-tools (vllm#58019 port); 2026-10-01: vllm-patches-roce (needs B12X).
PE="${VLLM_PATCHES_EXTRA:-}"; for pd in ${PE//,/ }; do
  PD="$OVERLAYS/$pd"; (cd "$PD" && sha256sum -c --quiet SHA256SUMS) || fail "$pd checksum mismatch"
  while read -r _ rel; do PATCH_MOUNTS+=(-v "$PD/$rel:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro"); done < "$PD/SHA256SUMS"
done
# VLLM_ENV=KEY=VALUE[,KEY=VALUE]: extra container environment (e.g. MIMO_DIFFKV_TUNE=0, VLLM_MARLIN_INPUT_DTYPE=fp8).
VE="${VLLM_ENV:-}"; ENV_EXTRA=(); for kv in ${VE//,/ }; do ENV_EXTRA+=(-e "$kv"); done
# B12X=<short sha> (2026-10-01): mount the b12x package of $B12X_ROOT/<sha> (a local-inference-lab/b12x checkout,
# Apache-2.0) read-only at /opt/b12x/b12x on PYTHONPATH, for RoCEnante (vllm-patches-roce); nothing installed.
if [ -n "${B12X:-}" ]; then
  BD="${B12X_ROOT:-$HOME/b12x}/$B12X"; [ "$(git -C "$BD" rev-parse --short=8 HEAD 2>/dev/null)" = "$B12X" ] || fail "b12x checkout $BD is not at $B12X"
  ENV_EXTRA+=(-v "$BD/b12x:/opt/b12x/b12x:ro" -e PYTHONPATH=/opt/b12x -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce)
fi
[ "$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null)" = "$VLLM_IMAGE_ID" ] || fail "image identity mismatch"
[ -s "$MODEL_HOST/model.safetensors.index.json" ] || fail "checkpoint missing"
[ -s "$NCCL_SO" ] || fail "NCCL library missing"
mkdir -p "$STATE/cache"
# The container sees the checkpoint at the same path name the recipe uses; the key file is read at exec time so
# no key appears in `docker inspect`.
serve=(vllm serve /models/MiMo-V2.6-Flash-MOPD --host 0.0.0.0 --port "$PORT"
  --nnodes 4 --node-rank "$R" --master-addr "$MASTER_IP" --master-port "${MASTER_PORT:-25026}" --tensor-parallel-size 4
  --served-model-name mimo-v2.6-flash --trust-remote-code --distributed-executor-backend mp
  --gpu-memory-utilization "${GMU:-0.70}" --kv-cache-memory-bytes "$KV_BYTES" --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS" --load-format "$LOAD_FORMAT" --moe-backend "${MOE_BACKEND:-marlin}"
  --speculative-config "$SPEC" --reasoning-parser mimo --enable-auto-tool-choice --tool-call-parser mimo
  --generation-config vllm --override-generation-config '{"temperature":1.0,"top_p":0.95}' ${VLLM_EXTRA//,/ }   # VLLM_EXTRA: comma-separated flags
  ${VLLM_CC:+--compilation-config "$VLLM_CC"})
[ "$R" != 0 ] && serve+=(--headless)
IFN=${MGMT_IFNAME:-enP7s7}
cmd=(docker run -d --name "$NAME" --restart no --label family=mimo26 --label engine=vllm
  --network host --ipc host --gpus all --shm-size 32g --cap-add IPC_LOCK --cap-add SYS_PTRACE
  --ulimit memlock=-1:-1 --device /dev/infiniband:/dev/infiniband ${CPUSET:+--cpuset-cpus "$CPUSET"}
  -v "$MODEL_HOST:/models/MiMo-V2.6-Flash-MOPD:ro" -v "$STATE/cache:/cache"
  -v "$(readlink -f "$NCCL_SO"):/opt/fleet-nccl/libnccl.so.2:ro" -v "$API_KEY_FILE:/run/secrets/api-keys:ro"
  "${PATCH_MOUNTS[@]}"
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-workspace
  -e VLLM_CACHE_ROOT=/cache/vllm-cache -e PYTHONUNBUFFERED=1
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e MBX_MTP_NONCHAIN="${MBX_MTP_NONCHAIN:-1}" "${ENV_EXTRA[@]}"
  -e LD_PRELOAD=/opt/fleet-nccl/libnccl.so.2 -e VLLM_NCCL_SO_PATH=/opt/fleet-nccl/libnccl.so.2
  -e "VLLM_HOST_IP=$IP" -e "NCCL_SOCKET_IFNAME=$IFN" -e "GLOO_SOCKET_IFNAME=$IFN"
  -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 -e "NCCL_IB_HCA=$HCAS" "${DUAL_ENV[@]}" -e NCCL_IB_GID_INDEX=3
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=1
  -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_ALGO=Ring -e NCCL_SKIP_TREE_CONNECT=1 -e NCCL_SWITCHLESS_RING_ONLY=1
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN
  --entrypoint bash "$IMAGE" -c 'exec "$@" --api-key $(tr "\n" " " < /run/secrets/api-keys)' _ "${serve[@]}")
idle(){
  # IDLE_FILTERS: container-name filters that must match nothing before a start (default: any vLLM / SGLang rank)
  local f filters=(); for f in ${IDLE_FILTERS:-sgl_ vllm_}; do filters+=(--filter "name=$f"); done
  [ -z "$(docker ps -q "${filters[@]}")" ] || fail "a model container is already running"
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || fail "GPU has active compute processes"
  awk '/MemAvailable/{exit !($2 > 100*1048576)}' /proc/meminfo || fail "MemAvailable < 100 GiB"
}
case "$MODE" in
 check) printf '%q ' "${cmd[@]}"; printf '\n';;
 up)
  idle; docker rm "$NAME" >/dev/null 2>&1 || true
  nohup python3 "$HERE/memory-guard.py" --state "$STATE" --container "$NAME" >> "$STATE/memory-guard.log" 2>&1 < /dev/null &
  "${cmd[@]}";;
 *) fail "usage: vllm-rank.sh <rank> {check|up}";;
esac
