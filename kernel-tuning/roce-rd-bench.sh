#!/usr/bin/env bash
# Ring-only RoCEnante recursive-doubling all-reduce vs NCCL on the four-Spark ring (2026-10-01). Run from rank 0's
# host while the ring is idle (no model containers): one throwaway container of the serving image per rank, b12x
# from the read-only checkout on PYTHONPATH (nothing installed), NCCL = the lane's patched library and environment.
#   [SIZES=bytes,...] roce-rd-bench.sh        prints rank 0's RESULT line (roce_rd_bench.py)
#   RD_SCRIPT=roce_rd_stress.py roce-rd-bench.sh   the stale-data stress test (N_AR, REPLAYS, PAUSE_EVERY, PAUSE_S)
# Site facts from launcher/hosts.env; the recipe must sit at the same path (REMOTE_DIR) on all four nodes, and the
# b12x checkout at $B12X_ROOT/$B12X_SHA on every node.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../launcher/load-hosts.sh
. "$HERE/../launcher/load-hosts.sh"
REMOTE_DIR=${REMOTE_DIR:-$(cd "$HERE/.." && pwd)}
IMAGE=${VLLM_IMAGE:-myllmbox/mimo-v26-flash-cluster-vllm:v2}
IFN=${MGMT_IFNAME:-enP7s7}
RD_SCRIPT=${RD_SCRIPT:-roce_rd_bench.py}
B12X_SHA=${B12X_SHA:-e4084d2e}
B12X=${B12X_ROOT:-$HOME/b12x}/$B12X_SHA
CACHE=${B12X_BENCH_CACHE:-$HOME/b12x-cache}
r() { if [ "$1" = 0 ]; then bash -c "$2"; else ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$(rank_ssh "$1")" "$2"; fi; }
for i in 0 1 2 3; do r "$i" "mkdir -p $CACHE; test \"\$(git -C $B12X rev-parse --short=8 HEAD)\" = $B12X_SHA" || { echo "rank $i: b12x tree missing or not at $B12X_SHA"; exit 1; }; done
nccl=${NCCL_DUAL_SO:?NCCL_DUAL_SO not set (hosts.env)}; hcas=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
for i in 3 2 1 0; do
  run="docker rm -f roce-rd-bench >/dev/null 2>&1; timeout ${BENCH_TIMEOUT:-900} docker run --rm --name roce-rd-bench --network host --ipc host --gpus all \
    --shm-size 4g --cap-add IPC_LOCK --ulimit memlock=-1:-1 --device /dev/infiniband:/dev/infiniband \
    -v \$(readlink -f $nccl):/opt/fleet-nccl/libnccl.so.2:ro -v $REMOTE_DIR/kernel-tuning/$RD_SCRIPT:/b.py:ro \
    -v $B12X:/b12x:ro -v $CACHE:/cache/b12x -e PYTHONPATH=/b12x -e B12X_ROCE_CACHE_DIR=/cache/b12x/roce \
    -e LD_PRELOAD=/opt/fleet-nccl/libnccl.so.2 -e RANK=$i -e WORLD_SIZE=4 -e MASTER_ADDR=$RANK0_IP -e MASTER_PORT=29757 \
    -e SIZES=${SIZES:-8192,32768,131072,524288,1048576,2097152} -e GRAPH_OPS=${GRAPH_OPS:-100} -e SAMPLES=${SAMPLES:-15} -e N_AR=${N_AR:-106} -e REPLAYS=${REPLAYS:-3000} -e PAUSE_EVERY=${PAUSE_EVERY:-250} -e PAUSE_S=${PAUSE_S:-0.2} \
    -e NCCL_SOCKET_IFNAME=$IFN -e GLOO_SOCKET_IFNAME=$IFN -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none \
    -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$hcas -e NCCL_IB_EXTENDED_IPV4_GIDS=1 -e NCCL_IB_PRESERVE_PCI_DOMAIN=1 \
    -e NCCL_IB_QPS_PER_CONNECTION=1 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 \
    -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_IB_MERGE_NICS=0 \
    -e NCCL_CROSS_NIC=1 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_ALGO=Ring -e NCCL_SKIP_TREE_CONNECT=1 \
    -e NCCL_SWITCHLESS_RING_ONLY=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_DEBUG=WARN --entrypoint python3 $IMAGE /b.py"
  if [ "$i" = 0 ]; then r 0 "$run" 2>&1 | grep -v -iE 'warn|^\s*$' | tail -40
  else r "$i" "nohup bash -c '$run' > /tmp/roce-rd-bench.log 2>&1 &"; fi
done
for i in 1 2 3; do echo "--- rank $i tail"; r "$i" "tail -4 /tmp/roce-rd-bench.log; docker rm -f roce-rd-bench >/dev/null 2>&1; true"; done
