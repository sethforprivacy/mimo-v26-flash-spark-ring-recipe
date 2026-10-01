#!/usr/bin/env bash
# Build the NCCL library the launcher preloads: NVIDIA NCCL v2.30.7-1 (commit 73cf1122) + SparkRing's cumulative
# switchless-cycle / dual-PCI-domain patch, for GB10 (sm_121) with the CUDA 13.0 toolkit. CPU only; no GPU, no root.
# This is the procedure behind the library we serve with (built on a DGX OS host, CUDA 13.0); see nccl/PROVENANCE.md.
#
#   build-nccl.sh [out-dir]          default $HOME/nccl-dual-pci; writes libnccl.so.2, its .sha256, LICENSE.txt and
#                                    ThirdPartyNotices.txt (keep NCCL's licence files with the binary)
#   env: CUDA_HOME (/usr/local/cuda), JOBS (4), PATCH (default: the vendored copy beside this script; or fetch
#        SparkRing's file, see PROVENANCE.md), IMAGE (optional: also check that the library loads in that image)
#
# Build on the host OS (or inside the serving image): the binary needs a glibc no newer than the image's. A build on
# DGX OS (glibc 2.39) loads in the pinned vLLM image; an older image needs a build inside that image.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HOME/nccl-dual-pci}
REV=73cf112295c33aee2b895f329f592f2a9b4b0f97
PATCH=${PATCH:-$HERE/nccl-2.30.7-dual-pci-domain.patch}
PATCH_SHA=8e2b8715d62d2b07a74caca3778da0eff2e7b54caf8a184bb728f179a5d1eba4
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
[ "$(sha256sum "$PATCH" | cut -d' ' -f1)" = "$PATCH_SHA" ] || { echo "patch sha256 mismatch: $PATCH" >&2; exit 1; }
WORK=$(mktemp -d "${TMPDIR:-/tmp}/nccl-build.XXXXXX")
echo "building in $WORK"
# LF checkout of the exact revision (the patch is written against LF sources; it adds one file with CRLF endings)
git -c core.autocrlf=false clone --quiet https://github.com/NVIDIA/nccl.git "$WORK/nccl"
git -C "$WORK/nccl" -c advice.detachedHead=false checkout --quiet "$REV"
[ "$(git -C "$WORK/nccl" rev-parse HEAD)" = "$REV" ] || { echo "checkout is not $REV" >&2; exit 1; }
cd "$WORK/nccl"
patch --batch --fuzz=0 -p1 < "$PATCH"
getconf GNU_LIBC_VERSION; g++ --version | head -1; "$CUDA_HOME/bin/nvcc" --version | tail -2
# the patch's CPU-only listener-handle / PCI-root compatibility test
g++ -std=c++11 -O2 -Wall -Wextra -Werror tests/routing_handle/compat.cc -o "$WORK/routing-handle-test"
"$WORK/routing-handle-test"
nice -n 10 make -j"${JOBS:-4}" src.build CUDA_HOME="$CUDA_HOME" CUDA_LIB="$CUDA_HOME/lib64" \
  NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'
mkdir -p "$OUT"
cp build/lib/libnccl.so.2.30.7 "$OUT/libnccl.so.2"
cp LICENSE.txt ThirdPartyNotices.txt "$OUT/"
ldd "$OUT/libnccl.so.2"
LD_PRELOAD="$OUT/libnccl.so.2" /bin/true
check='import ctypes, sys
lib = ctypes.CDLL(sys.argv[1]); v = ctypes.c_int()
assert lib.ncclGetVersion(ctypes.byref(v)) == 0 and v.value == 23007, v.value
print("ncclGetVersion", v.value)'
python3 -c "$check" "$OUT/libnccl.so.2"
if [ -n "${IMAGE:-}" ]; then
  docker run --rm --network none --memory 1g -v "$OUT/libnccl.so.2:/opt/candidate-nccl.so:ro" --entrypoint python3 \
    "$IMAGE" -c "$check" /opt/candidate-nccl.so
fi
(cd "$OUT" && sha256sum libnccl.so.2 | tee libnccl.so.2.sha256)
echo "done: copy $OUT to the same path on all four nodes and compare the sha256 on each"
