# NCCL library: provenance and how to obtain it

The launcher preloads one NCCL library into every rank container (`LD_PRELOAD` and `VLLM_NCCL_SO_PATH`). It is not
stock NCCL, and no binary is shipped here.

- **Base:** NVIDIA NCCL `v2.30.7-1`, commit `73cf112295c33aee2b895f329f592f2a9b4b0f97`
  (https://github.com/NVIDIA/nccl). The engines report it as `NCCL version 2.30.7+cuda13.0`. PyTorch's
  `torch.cuda.nccl.version()` reports the image's build-time pip NCCL (2.29.7) instead, not the preloaded library.
- **Patch:** `nccl-2.30.7-dual-pci-domain.patch`, sha256
  `8e2b8715d62d2b07a74caca3778da0eff2e7b54caf8a184bb728f179a5d1eba4`. It comes from SparkRing
  (https://github.com/FujitsuPolycom/sparkring, Apache-2.0) at commit `a48c862d9deba66c3cc1462095874a8ad4718811`,
  path `spark_transport/nccl/` (introduced by FujitsuPolycom/sparkring#248). The copy here is byte-identical to upstream, and so
  is `dual-pci-domain.json` (SparkRing's manifest, which labels the patch "research-only"). It is cumulative: it
  already contains SparkRing's switchless-cycle change. Apply it to the unmodified revision, and do not also apply
  `switchless-cycle.patch`. The lines it adds to `src/transport/generic.cc` have CRLF endings, as upstream; keep the
  file byte-exact (the sha256 above).
  Upstream URL of the same file:
  `https://raw.githubusercontent.com/FujitsuPolycom/sparkring/a48c862d9deba66c3cc1462095874a8ad4718811/spark_transport/nccl/nccl-2.30.7-dual-pci-domain.patch`
- **What the patch adds** (every flag defaults off):
  - `NCCL_SWITCHLESS_RING_ONLY=1` skips Tree and PAT transport setup. On a switchless ring those would try to
    connect ranks that share no cable.
  - `NCCL_IB_EXTENDED_IPV4_GIDS=1` lets a listener advertise up to four IPv4 GIDs instead of two, so the functions
    of both PCIe domains are reachable.
  - `NCCL_IB_PRESERVE_PCI_DOMAIN=1` makes the subnet fallback prefer a device on the same PCIe root.
  - `NCCL_IB_ROUTE_DIAGNOSTICS=1` logs the final route per QP (diagnostics only).
  - `NCCL_IB_SUBNET_AWARE_ROUTING` and `NCCL_IB_SUBNET_PREFIX_LEN` already exist in stock 2.30.7.
  - `NCCL_SKIP_TREE_CONNECT=1` is the name another patched NCCL (RiNGSiDE's) uses for the same skip. The launcher
    sets both, as our production does. This library acts on `NCCL_SWITCHLESS_RING_ONLY`.
- **Build:** `build-nccl.sh` runs the procedure we used: CUDA 13.0, `-gencode=arch=compute_121,code=sm_121`,
  `make src.build`, the patch's CPU compatibility test, and a load check (`ncclGetVersion` = 23007).
  - Our binary was built on a DGX OS host and has sha256
    `0553d4c74b9488224b3a9cb109c7fba5e4a4d663aed2dcdd4d2a42dbad8885d7`. That hash is provenance only. A rebuild with
    another toolchain is not byte-identical, so record your own hash and make sure all four nodes match.
  - SparkRing's measured reference build (CUDA 13.3) has a different hash again (`dual-pci-domain.json`).
  - A host build needs glibc >= 2.38 in the image that loads it. The pinned vLLM image is fine. For an older image,
    build inside that image.
- **Licences:** the patch is SparkRing's (Apache-2.0). It modifies NVIDIA NCCL sources, which carry their own licence
  (`LICENSE.txt`). Ship NCCL's `LICENSE.txt` and `ThirdPartyNotices.txt` with any binary you build;
  `build-nccl.sh` copies them next to it.
- **Evidence on this ring** (2026-09-13, before MiMo, with another model on SGLang):
  - with all four functions and both flags, 84/168 MiB all-reduces ran 2.03x/2.08x faster, and serving prefill
    gained 5-7 % with decode neutral;
  - listing four HCAs without the flags left the secondary domain idle.
