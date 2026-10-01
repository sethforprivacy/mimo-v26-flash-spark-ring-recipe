#!/usr/bin/env bash
# 4-rank all-reduce latency sweep on the ring with the lane's NCCL environment (vllm-rank.sh, NCCL_DUAL=1), one NCCL
# knob at a time (2026-09-30). Run from rank 0's host while the ring is idle (no model containers): short-lived
# containers of the serving image on every rank, removed after each config. Prints rank 0's RESULT line per config.
#   [SIZES=bytes,...] nccl-bench.sh [config ...]
#   configs: base ch1 ch2 ch4 ch8 ch16 ch4max16 gmix0 greg0 ch4gmix0 ch4greg0 ch4lm ll128 simple nt128 nt512 single
# Site facts from launcher/hosts.env; the recipe must sit at the same path (REMOTE_DIR) on all four nodes.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../launcher/load-hosts.sh
. "$HERE/../launcher/load-hosts.sh"
REMOTE_DIR=${REMOTE_DIR:-$(cd "$HERE/.." && pwd)}
IMAGE=${VLLM_IMAGE:-myllmbox/mimo-v26-flash-cluster-vllm:v2}
IFN=${MGMT_IFNAME:-enP7s7}
CONFIGS=("$@"); [ ${#CONFIGS[@]} -gt 0 ] || CONFIGS=(base ch1 ch2 ch4 ch8 ll128 simple nt128 nt512 single)
r() { if [ "$1" = 0 ]; then bash -c "$2"; else ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$(rank_ssh "$1")" "$2"; fi; }
for cfg in "${CONFIGS[@]}"; do
  nccl=${NCCL_DUAL_SO:?NCCL_DUAL_SO not set (hosts.env)}; hcas=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
  dual="-e NCCL_IB_EXTENDED_IPV4_GIDS=1 -e NCCL_IB_PRESERVE_PCI_DOMAIN=1 -e NCCL_IB_QPS_PER_CONNECTION=1"; extra=""
  case "$cfg" in
    base) ;;
    ch1) extra="-e NCCL_MIN_NCHANNELS=1 -e NCCL_MAX_NCHANNELS=1";;
    ch2) extra="-e NCCL_MIN_NCHANNELS=2 -e NCCL_MAX_NCHANNELS=2";;
    ch4) extra="-e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4";;
    ch8) extra="-e NCCL_MIN_NCHANNELS=8 -e NCCL_MAX_NCHANNELS=8";;
    ch16) extra="-e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16";;
    ch4max16) extra="-e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=16";;
    gmix0) extra="-e NCCL_GRAPH_MIXING_SUPPORT=0";;
    greg0) extra="-e NCCL_GRAPH_REGISTER=0";;
    ch4gmix0) extra="-e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_GRAPH_MIXING_SUPPORT=0";;
    ch4greg0) extra="-e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_GRAPH_REGISTER=0";;
    ch4lm) extra="-e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_LAUNCH_MODE=GROUP";;
    ll128) extra="-e NCCL_PROTO=LL128";;
    simple) extra="-e NCCL_PROTO=Simple";;
    nt128) extra="-e NCCL_NTHREADS=128";;
    nt512) extra="-e NCCL_NTHREADS=512";;
    single) nccl=${NCCL_SO:?NCCL_SO not set (hosts.env)}; hcas=rocep1s0f0,rocep1s0f1; dual="";;
    *) echo "unknown config $cfg"; continue;;
  esac
  for i in 3 2 1 0; do
    run="docker rm -f nccl-bench >/dev/null 2>&1; timeout 240 docker run --rm --name nccl-bench --network host --ipc host --gpus all \
      --shm-size 4g --cap-add IPC_LOCK --ulimit memlock=-1:-1 --device /dev/infiniband:/dev/infiniband \
      -v \$(readlink -f $nccl):/opt/fleet-nccl/libnccl.so.2:ro -v $REMOTE_DIR/kernel-tuning/nccl_ar_bench.py:/b.py:ro \
      -e LD_PRELOAD=/opt/fleet-nccl/libnccl.so.2 -e RANK=$i -e WORLD_SIZE=4 -e MASTER_ADDR=$RANK0_IP -e MASTER_PORT=29755 \
      -e CONFIG=$cfg -e SIZES=${SIZES:-8192,32768,131072,524288,1048576} -e NCCL_SOCKET_IFNAME=$IFN -e GLOO_SOCKET_IFNAME=$IFN -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none \
      -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$hcas $dual -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 \
      -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_IB_MERGE_NICS=0 \
      -e NCCL_CROSS_NIC=1 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_ALGO=Ring -e NCCL_SKIP_TREE_CONNECT=1 \
      -e NCCL_SWITCHLESS_RING_ONLY=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
      -e NCCL_DEBUG=WARN $extra --entrypoint python3 $IMAGE /b.py"
    if [ "$i" = 0 ]; then r 0 "$run" 2>&1 | grep -E "^RESULT|Error|error" | head -5
    else r "$i" "nohup bash -c '$run' > /tmp/nccl-bench-$cfg.log 2>&1 &"; fi
  done
  for i in 1 2 3; do r "$i" "docker rm -f nccl-bench >/dev/null 2>&1; true"; done
done
