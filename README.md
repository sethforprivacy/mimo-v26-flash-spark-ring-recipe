# MiMo-V2.6-Flash on four DGX Sparks: a switchless-ring TP4 recipe for vLLM

By Seth For Privacy.

This recipe serves [XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) at
revision `2479e2d0029eca9a34cc7e7f55a121925f81908e` with tensor parallelism 4 across four NVIDIA DGX Spark (GB10)
nodes. The nodes are cabled as a switchless ring. The model is a 309B-parameter MoE (~15B active) with MXFP4 routed
experts, FP8 attention and 3 MTP layers; it accepts images.

The stack:
- **Engine:** vLLM from the public image `myllmbox/mimo-v26-flash-cluster-vllm:v2`, pinned by digest.
- **Overlays:** six sets of file overlays, sha-checked and mounted read-only:
  - two upstream vLLM fixes ported to this image;
  - GB10 attention and FP8 GEMM tuning;
  - an FP8 KV cache with its own GB10-tuned attention launch;
  - a ring-only RDMA all-reduce built on b12x's RoCEnante.
- **Transport:** a patched NCCL 2.30.7 that uses both PCIe functions of every link.

It is the production profile we have served since 2026-10-03 (FP8 KV cache), with the benchmark and microbenchmark
scripts behind every number below. A bf16-KV alternative is kept: `launcher/profile.bf16-20261001.env`, the previous
production profile. Every number was measured on our own cluster.

**Status:** a research-grade recipe from one cluster. **Licence:** MIT for our own files; third-party pieces keep
their own licences (see [Licence](#licence)).

## Contents

- [Results: launch day vs today](#results-launch-day-2026-09-29-vs-today-2026-10-01)
- [Long context: bf16 KV vs FP8 KV](#long-context-bf16-kv-vs-fp8-kv-2026-10-03)
- [Hardware and cabling](#hardware-and-cabling)
- [Software pins](#software-pins)
- [Bring-up](#bring-up)
- [The production profile](#the-production-profile)
- [What each overlay changes](#what-each-overlay-changes)
- [Tried and rejected](#tried-and-rejected)
- [Rollback](#rollback)
- [Limitations and caveats](#limitations-and-caveats)
- [Reproducing the numbers](#reproducing-the-numbers)
- [Layout](#layout) · [Credits](#credits) · [Licence](#licence)

## Results: launch day (2026-09-29) vs today (2026-10-01)

The hardware, cables, checkpoint and image are the same in both columns; only the profile and the overlays differ.
"Launch" is the 2026-09-29 profile (`launcher/profile.launch-20260929.env`). "Today" is the 2026-10-01 production
profile, with a bf16 KV cache (`launcher/profile.bf16-20261001.env`). The FP8-KV profile that replaced it on
2026-10-03 (`launcher/profile.env`) is compared at long context in the next section.

| | launch | today | change |
|---|---:|---:|---:|
| ~500K-token prompt, cold, needle at 10 / 50 / 90 % depth (median wall time) | 729 s | 301 s | **2.4x** |
| ~287K-token prompt, cold, needle at 50 % | 271 s | 129 s | 2.1x |
| Cold prefill at ~64K / ~128K / ~250K tokens (tok/s) | 2,549 / 1,802 / 1,181 | 3,555 / 3,025 / 2,379 | +39 / +68 / **+101 %** |
| Max context / seats / KV pool | 524,288 / 32 / 2.56M tokens | **1,048,576 / 64** / 3.96M tokens | 2x / 2x / 1.55x |
| ~1M-token prompt, needle at 50 % | not servable | 1,007,197 tokens, exact, 977 s ¹ | new |
| Single stream, RigMark prose / code / structured (tok/s) | 63.4 / 99.7 / 107.2 | 75.8 / 121.7 / 128.2 | **+20 / +22 / +20 %** |
| Short-code aggregate, RigMark C1 / C4 / C16 (tok/s) | 86 / 190 / 406 | 103 / 232 / 512 | +20 / +22 / +26 % |
| Sampled-prose aggregate C1 / C8 / C32 (tok/s) | 59.5 / 170 / 295 | 71.8 / 202 / 444 | +21 / +19 / **+50 %** |
| Sampled prose C64 / RigMark short-code C64 (tok/s) | (32 seats) | 608 / 1,083 | new |
| llm-inference-bench, 1 stream at 0 / 128K context (tok/s) | 67.4 / 53.5 | 77.6 / 74.9 | +15 / **+40 %** |
| llm-inference-bench, C32 at 0 / 64K context; C16 at 128K (tok/s) | 421 / 333; 236 | 641 / 532; 327 | +52 / +60; +39 % |
| llm-inference-bench scout prefill 8K / 64K / 128K (tok/s) | 3,800 / 2,526 / 1,814 | 4,303 / 3,522 / 3,016 | +13 / +39 / +66 % |
| ~77K-token multi-turn session, hot-turn TTFT at C1 / C4 | 1.29 / 3.59 s | 0.86 / 2.16 s | 1.5x / 1.7x |
| Coding-agent tool loop, 12 episodes: seconds per turn | 2.28 | 1.68 | **1.36x** |
| Agent loop finished / duplicate tool calls; garbage-token probe flagged | 12/12, 0; 0 of 64 | 12/12, 0; 0 of 128 | |

¹ Measured on 2026-09-30, on the same profile minus RoCEnante. RoCEnante only carries all-reduces of 1 MB or less;
cold prefill at 250K read 2,365 tok/s without it and 2,379 with it.

**Basis:**
- **Clean boots.** Every measurement except the launch needles ran on a fresh boot: all four nodes were rebooted,
  and the engine started only once every node reported > 100 GiB MemAvailable, ≥ 2,000 free 32 MiB blocks and no
  model container.
  - The launch column was measured on 2026-09-30 with the unchanged 2026-09-29 profile. llm-inference-bench ran on
    a second clean boot of the same profile.
  - The today column is one boot of the 2026-10-01 profile. The qualification (needles) and llm-inference-bench ran
    on the same boot.
- **Client.** The benchmark client ran on rank 0's node, pinned to the A725 cores 0-4,10-14.
- **The launch needles** come from the 2026-09-29 cutover qualification instead. It ran on the lane as booted for
  the cutover, not after a fresh reboot, and its client was another host on the LAN. For prompts this long the
  client side is negligible.
- **Settings.** Both columns use the same speculative decoding (MTP K=3) and the same sampling defaults.
- **Run counts:**
  - RigMark: median of 2 runs per cell; short-code aggregates over 2 concurrency runs. Thinking off, T=0.
  - Sampled-prose ladder: mean of 2 repetitions at T=0.7, 400-token answers, ~120-word prompts, thinking off.
  - Cold prefill: fresh random prompts with `max_tokens` 1; 2 repetitions at ~8K / 32K / 64K, 1 at ~128K / 250K.
    The rate is prompt tokens divided by wall time.
  - Needles: one request per depth.
  - llm-inference-bench: one 30 s duration cell per (concurrency, context), thinking on, T=1.0.
  - Multi-turn: 3 turns per session; "hot" means turns 2-3.
  - Agent loop: 12 episodes, 4 in parallel, thinking on, T=1.0.
  - Garbage probe: C32 × 1,200 tokens × 2 rounds at launch, 4 rounds today.
- **Noise.** Two boots with identical kernels read 107.2 and 104.4 tok/s on RigMark code, so single-stream
  differences under ~3-4 % are not resolvable.

The decode step itself (torch profiler, C1, rank 0) went from 34.1 to 30.7 ms at 2K context and from 35.1 to
31.3 ms at 60K. That change is RoCEnante alone; the attention tuning had already removed most of the slowdown with
context depth.

## Long context: bf16 KV vs FP8 KV (2026-10-03)

**Why.** At 250K-1M tokens of context, a decode step is dominated by reading the 9 full-attention layers' KV cache:
- at ~500K, those reads take 9.7 ms of a 39.6 ms step, against 0.7 ms at 2K (torch profiler, C1, rank 0);
- the bf16 reads already run at ~270 GB/s, GB10's memory-bandwidth roof.

So the remaining lever is fewer bytes: an FP8 KV cache.

**The FP8 kernel first had to be retuned.** With the bf16-tuned launch, the FP8 cache (`vllm-patches-fp8kv`) reached
only 1.17-1.19x on half the bytes (~177 GB/s). A launch sweep on one GB10 (`kernel-tuning/attn_fp8_dec_sweep.py`;
outputs within 2e-5) found:
- **spec-verify:** 64:32:4:2 with 128 split-KV segments, 1.59 / 1.64 / 1.68x vs bf16 at 250K / 500K / 1M, and 2.09x
  for two 500K sequences;
- **prefill tail:** 128:128:8:2, 0.91x of bf16 (0.83x before). The 128-token tile only fits in shared memory with
  FP8.

The same sweep on the bf16 launch (`kernel-tuning/attn_bf16_dec_sweep.py`) found no gain: 0.99-1.03x at one
sequence.

**Serving, one session.** Each run: a cold prompt, then two hot ~1.3K-token turns. Engine step from the engine's own
counters; each profile on a fresh boot.

| context | engine step, bf16 → FP8 | hot-turn TTFT, bf16 → FP8 | cold prefill, bf16 → FP8 |
|---|---:|---:|---:|
| ~248K | 35.2-36.1 → 34.1 ms (−4 %) | 1.65-1.8 → 1.87-1.89 s | 104 → 110 s |
| ~494K | 40.5-40.8 → 36.1-36.9 ms (−10 %) | 2.9-3.5 → 3.37 s | 298 → 321 s |
| ~933K | 49.6-50.8 → 42.0-42.1 ms (**−16 %**) | 5.1-5.5 → 5.4-6.0 s | 854 → 943 s |

- **Per turn:** for a turn generating ~770 tokens, about −2.5 / −7 / −10 %.
- **Capacity:** the KV pool doubles to 7,927,271 tokens.

**Quality.**

| | FP8 | bf16 |
|---|---|---|
| Corpus NLL | 1.8327 | 1.8281-1.8309 (several boots) |
| Needle at 1,007,259 tokens | exact, 1,125 s | exact, 977 s |

On FP8, also:
- needles exact at 287K and at ~500K (10 / 50 / 90 % depth);
- concurrent 2 × ~247K and 4 × ~123K needles all exact;
- agent loop 12/12 with no duplicate tool calls, and 0 of 64 garbage-probe responses flagged.

**Two concurrent sessions** (FP8 profile):

| context | engine step | aggregate vs one session | per-session decode | hot-turn TTFT |
|---|---:|---:|---:|---:|
| ~248K | 51 ms | ~1.4x | 47-54 tok/s | 2.5 s |
| ~494K | 56-61 ms | ~1.3x | 40-43 tok/s | 4.3-4.4 s (mean) |

The second session mostly adds MoE expert reads (more distinct experts per step), not attention.

**At short context FP8 is neutral to slightly negative.** On 2026-10-01, with the earlier launch:
- RigMark single stream read 119.6 / 73.3 / 127.5 tok/s (code / prose / structured), against 122.2 / 71.1 / 128.1
  on bf16;
- cold prefill was 3-8 % slower.

For short contexts, or where exact bf16 KV numerics matter, use `launcher/profile.bf16-20261001.env`.

## Hardware and cabling

- **Nodes.** Four DGX Spark (GB10, 128 GB unified memory, 10 Cortex-X925 + 10 Cortex-A725 cores). Each node has a
  ConnectX-7 with two QSFP ports. Each port appears as **two PCIe functions on two PCIe root domains**, which gives
  four RoCE devices per node:

  | QSFP port | PCIe domain 0 | PCIe domain 2 |
  |---|---|---|
  | port 0 | `rocep1s0f0` (netdev `enp1s0f0np0`) | `roceP2p1s0f0` (netdev `enP2p1s0f0np0`) |
  | port 1 | `rocep1s0f1` (netdev `enp1s0f1np1`) | `roceP2p1s0f1` (netdev `enP2p1s0f1np1`) |

- **Ring.** The four nodes form a **ring 0-1-2-3-0**: each node's port 0 is cabled to the next node's port 1, and
  there are no diagonal cables (0-2, 1-3). Rank numbers must follow the cables, because the RoCE overlay hardcodes
  which port faces which neighbour.
- **Addressing.** Every PCIe function gets its own **point-to-point /24** with the matching function on the other
  end of the cable: 4 cables × 2 functions = 8 subnets. The ring needs no IP forwarding, relay routes or switch.
  NCCL's ring and the recursive-doubling all-reduce only ever talk to a cabled neighbour.
  Example plan (all addresses are placeholders; `.1` is the port-0 end, `.2` the port-1 end):

  | cable | port 0 of | port 1 of | domain-0 subnet (`enp1s0f0np0` ↔ `enp1s0f1np1`) | domain-2 subnet (`enP2p1s0f0np0` ↔ `enP2p1s0f1np1`) |
  |---|---|---|---|---|
  | A | rank 0 | rank 1 | 10.200.1.0/24 | 10.200.2.0/24 |
  | B | rank 1 | rank 2 | 10.200.3.0/24 | 10.200.4.0/24 |
  | C | rank 2 | rank 3 | 10.200.5.0/24 | 10.200.6.0/24 |
  | D | rank 3 | rank 0 | 10.200.7.0/24 | 10.200.8.0/24 |

- **Fabric settings.** MTU 9000 on all 16 functions; RoCE active MTU 4096. **GID index 3 must be the RoCE v2
  IPv4-mapped GID** of each function (the launcher sets `NCCL_IB_GID_INDEX=3`).
  - Use static addressing. On our nodes, a default DHCP / IPv6 link-local profile on an unused function took the GID
    slots we expected for IPv4, and only a dedicated static profile with IPv6 disabled kept the order stable.
- **Management.** A separate management LAN (the onboard `enP7s7`) carries ssh, torch.distributed, gloo and the NCCL
  bootstrap. If a host firewall is active, allow the four nodes to reach each other on it: `MASTER_PORT`, plus
  ephemeral TCP ports for gloo and NCCL. Allow ICMP on the fabric interfaces for the launcher's link checks. RDMA
  itself is offloaded to the NIC.

Checks per node (read-only):

```bash
ip -br addr show enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1        # four addresses, UP, mtu 9000
for d in rocep1s0f0 rocep1s0f1 roceP2p1s0f0 roceP2p1s0f1; do
  echo "$d $(cat /sys/class/infiniband/$d/ports/1/gids/3) $(cat /sys/class/infiniband/$d/ports/1/gid_attrs/types/3)"
done                                                                          # 0000:...:ffff:<ipv4 in hex> and "RoCE v2"
ping -M do -s 8972 -c 3 <the neighbour's address on each of the four subnets>  # jumbo frames end to end
```

## Software pins

| piece | pin |
|---|---|
| Serving image | `myllmbox/mimo-v26-flash-cluster-vllm:v2@sha256:5af49bd0c38923d1d2b3636aa7570b311ff55480f123207259b083cd9afde726`, image ID (config digest) `sha256:a09549748e0b42d5d89608888b55a826ead04a2d7a7b5318c47dcf8ea0593ace`, arm64, ~10.1 GB compressed. See the note below the table. |
| Checkpoint | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` @ `2479e2d0029eca9a34cc7e7f55a121925f81908e`, identical on all four nodes (`launcher/fetch-checkpoint.py` verifies it against the Hub's per-file digests) |
| b12x (RoCEnante) | [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) @ `e4084d2eef4932e0fa06db3f7a94deb83ad132d7` (b12x 1.5.0, Apache-2.0). A checkout is mounted **read-only** into the container on `PYTHONPATH`; nothing is installed. The launcher refuses to start if the checkout is not at that commit. |
| NCCL | NVIDIA NCCL `v2.30.7-1` (`73cf112295c33aee2b895f329f592f2a9b4b0f97`) + SparkRing's cumulative switchless-cycle / dual-PCI-domain patch (`nccl/`, sha256 `8e2b8715…`), CUDA 13.0, `sm_121`. Preloaded with `LD_PRELOAD` and `VLLM_NCCL_SO_PATH`. Build and provenance: [`nccl/PROVENANCE.md`](nccl/PROVENANCE.md). |
| Overlays | `overlays/` = the five production overlay sets, identical to production apart from one comment, each with `SHA256SUMS` (checked at every start) |
| Host | DGX OS (Ubuntu 24.04 base, glibc 2.39; kernel 6.17.0-1031-nvidia when measured), Docker with the NVIDIA container runtime, python3 and git on the host |
| Benchmarks | RigMark: [othexmr/rigmark](https://github.com/othexmr/rigmark) @ `40fabcaf`, a fork of alexellis/rigmark. llm-inference-bench: [local-inference-lab/llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) @ `a50025a3`. |

What the image is:
- Stock `vllm/vllm-openai:nightly-ddd6fbca` (2026-09-26) plus about 10 changed files and myllmbox's closed-source
  loader `libmbx_loader.so`.
- `--load-format mbx` converts the BF16 `o_proj` layers and the LM head to NVFP4 at load and caches the result
  under the state directory.
- The image also provides split-KV attention for spec-verify steps (`MBX_DIFFKV_3D_MAX_Q=8`) and non-chained MTP
  (`MBX_MTP_NONCHAIN=1`).
- It ships FlashInfer 0.7.0, Triton ≤ 3.7.1 and xgrammar 0.2.7.

## Bring-up

All four nodes need the same user, the same recipe path, bash ≥ 4.4 (DGX OS ships 5.x), and key-based
non-interactive ssh from rank 0 to ranks 1-3.

1. **Fabric.** Cable and address the ring as above. Run the checks on every node.
2. **Image, on every node** (or pull once and relay with `docker save | ssh <node> docker load`; the launcher checks
   the image ID, which survives a relay):
   ```bash
   docker pull myllmbox/mimo-v26-flash-cluster-vllm:v2@sha256:5af49bd0c38923d1d2b3636aa7570b311ff55480f123207259b083cd9afde726
   docker tag myllmbox/mimo-v26-flash-cluster-vllm@sha256:5af49bd0c38923d1d2b3636aa7570b311ff55480f123207259b083cd9afde726 \
     myllmbox/mimo-v26-flash-cluster-vllm:v2
   docker image inspect --format '{{.Id}}' myllmbox/mimo-v26-flash-cluster-vllm:v2   # sha256:a09549748e0b...
   ```
   With Docker's containerd image store, `.Id` can be the manifest digest (here `sha256:5af49bd0…`) instead of the
   config digest. In that case set `VLLM_IMAGE_ID` to what your daemon prints for the image you pulled by digest.
3. **Checkpoint, staged once and copied:**
   ```bash
   python3 launcher/fetch-checkpoint.py --dest /srv/models      # needs huggingface_hub + hf_xet
   # copy /srv/models/MiMo-V2.6-Flash-MOPD/ to ranks 1-3 (rsync over the LAN or the fabric), then on each node:
   cd /srv/models/MiMo-V2.6-Flash-MOPD/2479e2d0029eca9a34cc7e7f55a121925f81908e && sha256sum -c --quiet ../2479e2d0029eca9a34cc7e7f55a121925f81908e.sha256
   ```
4. **b12x, on every node:**
   ```bash
   git clone https://github.com/local-inference-lab/b12x ~/b12x/e4084d2e
   git -C ~/b12x/e4084d2e checkout --detach e4084d2eef4932e0fa06db3f7a94deb83ad132d7
   ```
5. **NCCL, built once and copied:**
   ```bash
   IMAGE=myllmbox/mimo-v26-flash-cluster-vllm:v2 nccl/build-nccl.sh ~/nccl-dual-pci   # CPU-only, CUDA 13.0 toolkit
   # copy ~/nccl-dual-pci/ to the same path on ranks 1-3 and compare libnccl.so.2's sha256 on all four
   ```
6. **Site file and keys.**
   - `cp launcher/hosts.env.example launcher/hosts.env` and edit the addresses, `LINK_CHECKS` and paths. The file
     must be identical on all nodes.
   - Put the API key file (one key per line) at `API_KEY_FILE` on **all four** nodes. Every rank parses the same
     command line, though only rank 0 serves HTTP. The file is bind-mounted read-only and read at exec time, so no
     key appears in `docker inspect`.
7. **Dry run.** `launcher/ring-up.sh check` prints every rank's `docker run` command. On each rank it verifies the
   image ID, the checkpoint, the NCCL library, the b12x commit and every overlay's checksums. It starts nothing.
8. **Boot.** Reboot all four nodes first. GB10 unified memory fragments over time; an un-rebooted node can lose a
   large share of its throughput while passing every memory check. Then run, on rank 0:
   ```bash
   launcher/ring-up.sh up        # or from rank 0's crontab: @reboot sleep 60 && bash .../launcher/ring-up.sh up >> ~/mimo26-run/owner.log 2>&1
   ```
   - What `ring-up.sh up` does:
     - waits for ssh to ranks 1-3, one fabric ping per cable, and > 100 GiB MemAvailable on every node;
     - starts the ranks in the order **3, 2, 1, 0**: headless workers first, then rank 0, which serves the API on
       `PORT`;
     - starts a host memory guard per rank and the optional `:8016` liveness shim;
     - waits up to 60 min for `/health`.
   - Timing: a restore with warm caches went from reboot to serving in about 5 minutes on our ring. The first boot
     is longer, because it fills the FlashInfer autotune, mbx and b12x proxy caches under `STATE`.
9. **Verify:**
   - `bench/post-boot-check.sh`: keys, 401s, vision, tool turns, RoCE active with 0 errors on every rank, the KV
     pool, liveness;
   - `bench/qualify.sh <out-dir> <keyfile>`: the full qualification with needles to ~500K tokens; about 25 min.

A request:

```bash
curl -s http://<rank0>:8015/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"mimo-v2.6-flash","messages":[{"role":"user","content":"Hello"}],"chat_template_kwargs":{"enable_thinking":false}}'
```

Thinking is on by default (`--reasoning-parser mimo` returns it separately). Tool calls use `--tool-call-parser
mimo`. Generation defaults are T=1.0, top_p=0.95.

## The production profile

`launcher/profile.env`. `ring-up.sh` passes it to `launcher/vllm-rank.sh` as environment.

| knob | value | why |
|---|---|---|
| `PORT` | 8015 | API port on rank 0 |
| `NCCL_DUAL` | 1 | NCCL over all four RoCE functions (both PCIe domains) with `NCCL_IB_EXTENDED_IPV4_GIDS=1 NCCL_IB_PRESERVE_PCI_DOMAIN=1` |
| `VLLM_EXTRA` | `--linear-backend=triton,--max-num-batched-tokens=16384,--kv-cache-dtype=fp8` | CUTLASS FP8 linear rejects the per-rank (4096, 3392) QKV shape at TP4; 16K batched tokens gave +38 % / +20 % prefill at 8K / 64K, with decode unchanged; FP8 KV cache (per-tensor scale 1.0) |
| `KV_BYTES` | 30000000000 | KV cache per rank; pool 7,927,271 tokens with FP8 KV (7.56 × 1M; bf16: 3,963,635) |
| `MAX_LEN` / `MAX_SEQS` | 1048576 / 64 | 1M context; 64 seats (aggregate keeps scaling past C32) |
| `MBX_MTP_NONCHAIN` | 1 | each of the 3 MTP layers reads the target model's hidden states |
| `VLLM_PATCHES` + `VLLM_PATCHES_EXTRA` | `vllm-patches` + `fp8kv,fp8cfg,tools,roce` | the overlays below, sha-checked and mounted read-only (`fp8kv` replaces `attn2`) |
| `B12X` | e4084d2e | the b12x checkout `vllm-patches-roce` needs |
| `VLLM_ENV` | `NCCL_MIN_NCHANNELS=4,NCCL_MAX_NCHANNELS=4,MIMO_ROCE_RD=1,MIMO_ROCE_RD_MAX=1048576,MIMO_DIFFKV_DEC_FULL=64:32:4:2,MIMO_DIFFKV_SEGS=128:16:4,MIMO_DIFFKV_PF_FULL=128:128:8:2` | NCCL 4 channels; TP all-reduces ≤ 1 MB on RoCEnante recursive doubling; the FP8-tuned attention launch (never use `PF_FULL=128:128:...` with a bf16 cache: the tile only fits shared memory with FP8) |

**Fixed in `vllm-rank.sh`:**
- Engine: TP4 over `--nnodes 4`, MTP K=3 (`--speculative-config {"method":"mtp","num_speculative_tokens":3}`),
  `--moe-backend marlin`, `--gpu-memory-utilization 0.70`, `--generation-config vllm` with T=1.0 / top_p=0.95,
  `--load-format mbx`.
- NCCL: Ring algorithm only, with P2P, SHM, NVLS and cuMem disabled; GID 3, RoCE v2, IPv4; subnet-aware routing on
  /24; one QP per connection; `NCCL_SWITCHLESS_RING_ONLY=1`.
- Container: host network and IPC, `/dev/infiniband`, unlimited memlock, `--restart no`.

The profile format is `KEY=VALUE`, one per line, with comments on their own lines. Any other `vllm-rank.sh` knob can
be added the same way (`SPEC_K`, `MOE_BACKEND`, `GMU`, `CPUSET`, `VLLM_CC`, ...).

## What each overlay changes

`vllm-rank.sh` mounts every file listed in an overlay's `SHA256SUMS` read-only over the image's vLLM package, after
`sha256sum -c`. Overlays must not overlap. The files are modified copies of this image's vLLM files (see
[`overlays/NOTICE`](overlays/NOTICE)), so they are valid for this image only.

**`vllm-patches`: vllm-project/vllm#58235, MiMo ViT sink as a null softmax logit (2026-09-29).**
- The bug: the vision tower's window attention applied its learned sink as a bias on key 0, and every flat-colour
  image was described as "Black".
- With the upstream fix ported: red, blue and green squares at 64 and 448 px, and a red | blue split image, are
  all read correctly.
- Correctness only; no speed effect.

**`vllm-patches-attn2`: GB10 launch tuning of the Triton DiffKV attention, plus a Q·Kᵀ split (2026-09-30).**
- The problem: at TP4, MiMo's 9 full-attention layers have 16 query heads per KV head on each rank. The stock
  launcher's `BLOCK_M=16` therefore gives every program a single query token, so prefill re-reads the whole KV
  cache once per token and a K=3 verify step reads it four times.
- The overlay adds a per-layer-type launch policy:
  - full-attention prefill: 128:64:8:2 tiles;
  - sliding-window prefill: 128:32:8:2;
  - full-attention verify: one split-KV (3D) program per verify, 64:64:4:2 with 32 segments up to 4 sequences;
  - sliding-window verify: a 2D 32:32:4:2 launch, because only the 128-token window is live.
- It also adds a QK split: Q·Kᵀ as 128 + 64 dims instead of padding 192 to 256. That frees the shared memory the
  wider prefill tiles need.
- Kernel effect: 2.7-3.3x on prefill and 1.2-2.3x on verify, with outputs equal to the stock launch within bf16
  rounding.
- In serving:
  - the first version (together with `vllm-patches-fp8cfg`) raised cold prefill at 64K / 128K / 250K by
    +38 / +63 / +91 %;
  - the QK split added +2.6 / +4.5 / +6.6 % on top (250K prompt: 211 s to 104 s);
  - ~77K hot-turn TTFT improved 1.5x;
  - single-stream decode no longer falls with context depth (1 stream at 128K: 53.5 to ~70 tok/s);
  - corpus NLL stayed within 0.0005.
- Switches: `MIMO_DIFFKV_TUNE=0` restores the stock launch, `MIMO_DIFFKV_QK_SPLIT=0` drops the split, and
  `MIMO_DIFFKV_PF_FULL`, `_PF_SWA`, `_DEC_FULL`, `_DEC_SWA` and `_SEGS` override single configs (pass them via
  `VLLM_ENV`).
- Upstream vllm-project/vllm#58177 and vllm-project/vllm#59085 / vllm-project/vllm#58141 cover parts of this.

**`vllm-patches-fp8cfg`: block-FP8 GEMM configs for NVIDIA_GB10 (2026-09-30).**
- The boot log warned that the four per-rank QKV / dense shapes (N = 3392, 3712, 4096, 8192; K = 4096) run on
  default Triton configs; vLLM ships none for GB10.
- The configs were tuned with vLLM's own tuner (`kernel-tuning/tune_fp8.py`).
- Effect: 1.2-1.4x on the decode-size GEMMs with L2-cold weights (default ~145-150 GB/s, tuned 190-215 GB/s), about
  1 ms per decode step; 1.08-1.11x at prefill sizes.

**`vllm-patches-tools`: port of vllm-project/vllm#58019, the `mimo` tool parser (2026-09-30).**
- The bug: `--tool-call-parser mimo` mapped to the Qwen3 engine parser, whose converter strips one leading and one
  trailing newline from **every** string argument. MiMo emits compact `<parameter=NAME>VALUE</parameter>`, so real
  newlines were lost: written files lost their final newline, and `str_replace` strings lost a line break at either
  end.
- With the port, values are verbatim, and typed parameters are still coerced by the tool schema (`bench/tools-unit.sh`
  shows the stock bug and the fix offline).
- Difference from upstream: no xgrammar structural tag, because the image has xgrammar 0.2.7 and upstream's tag needs
  0.2.8. Strict `tool_choice` (`required` or a named tool) therefore goes through vLLM's generic path.

**`vllm-patches-roce`: ring-only RoCEnante recursive doubling for TP all-reduces ≤ 1 MB (2026-10-01).**
- What RoCEnante is: b12x's one-shot all-reduce. Every rank RDMA-writes its input into pinned host slots of every
  peer, which works on GB10's unified memory without GPUDirect; GPUDirect RDMA is unsupported on DGX Spark. A C proxy
  posts the writes, and one kernel per collective sums in a fixed order.
- Its 4-rank form needs every peer reachable, and on this ring the diagonals are not cabled.
  `mimo_roce_rd.py` composes two **2-rank** runtimes over the direct links instead:
  - round A pairs ranks (0,1) and (2,3);
  - round B pairs (0,3) and (1,2), on the round-A sums.

  Every rank computes (x0 + x1) + (x2 + x3) in the same order, so the outputs are bit-identical across ranks.
- Eligibility, decided from shape and dtype only so every rank routes alike: bf16 / fp16 / fp32, contiguous, a
  multiple of 16 bytes, ≤ `MIMO_ROCE_RD_MAX` (1 MB). Eligible all-reduces take this path eagerly and inside CUDA
  graphs; larger ones and the all-gathers stay on NCCL.
- Measured:
  - per call in graph replay vs NCCL: 8 KB 19.0 vs 70.6 µs, 128 KB 42.4 vs 85.1 µs, 1 MB 184.6 vs 187.2 µs (NCCL
    wins at 2 MB, so the cutoff is 1 MB);
  - the ~106 all-reduces of a decode step: 3.3 ms instead of NCCL's 6.3 ms plus host-node gaps;
  - decode step 34.1 to 30.7 ms (C1, 2K context);
  - RigMark single stream +12 / +18 / +13 % (code / prose / structured);
  - ~77K multi-turn engine step −11 % at C1, −7 % at C4;
  - agent loop 1.94 to 1.75 s per turn;
  - prefill unchanged.
- Correctness:
  - 0 of 128 garbage-probe responses flagged, and the agent loop at 12/12;
  - NLL 1.8281 vs 1.8289 (bf16 summation order);
  - an in-serving shadow check ran every RoCE all-reduce through NCCL too: 1.92M calls per rank, none more than 5 %
    apart;
  - a graph-replay stress test: 1.27M all-reduces, 0 mismatches.
- The overlay also adds a health check before every eager logits all-gather, so a poisoned runtime raises like an
  NCCL timeout.

**`vllm-patches-fp8kv`: FP8 KV cache for the DiffKV attention (ported 2026-10-01, production since 2026-10-03).**
- A port of vllm-project/vllm#58128 onto the tuned `attn2` kernel, so it **replaces** `vllm-patches-attn2`:
  - MiMo's decoder and MTP layers forward `cache_config`, so `--kv-cache-dtype fp8` takes effect; on stock vLLM it
    is silently ignored;
  - the DiffKV backend and kernel accept FP8 K/V with per-tensor descales; the checkpoint carries no KV scales, so
    they are 1.0.
- Difference from upstream: both dots run in fp16 (Q cast once; fp8 → fp16 is one conversion per pair). Upcasting FP8
  tiles to bf16 made prefill attention 1.8x slower on GB10 (Triton 3.7.1); fp16 dots cut that to 0.83-0.86x.
- The FP8 path needs its own launch, set through `VLLM_ENV`:
  - `MIMO_DIFFKV_DEC_FULL=64:32:4:2` and `MIMO_DIFFKV_SEGS=128:16:4` take spec-verify from 1.17-1.19x to
    1.59-1.68x of bf16;
  - `MIMO_DIFFKV_PF_FULL=128:128:8:2` takes the prefill tail from 0.83x to 0.91x;
  - segment counts must be powers of two.

  See [Long context](#long-context-bf16-kv-vs-fp8-kv-2026-10-03) for the serving numbers.
- The kernel test `kernel-tuning/attn_fp8kv_test.py`:
  - the FP8 path matches the bf16 kernel run on the dequantized cache to ≤ 0.00098;
  - the FP8 quantization itself is ~3 % relative.

**Not overlays, also part of today's profile:**
- **NCCL 4 channels.** NCCL's own channel choice was 2.7-5.3x too slow at 512 KB-1 MB, exactly the verify
  all-reduce size at C16-C32. Pinning 4 channels gave C32 sampled prose 302 to 414 tok/s (+37 %) and short-code C16
  +19 %; single stream and prefill stayed flat.
- **Capacity:** 64 seats (C48 / C64 sampled prose 505 / 583 tok/s with no preemptions), KV 30 GB per rank, and a
  1,048,576-token context (needle exact at ~1M tokens).

## Tried and rejected

Each item was measured after a clean reboot, against the then-current profile.

- 2026-09-30:
  - **CPU pinning** to the ten X925 cores (`CPUSET=5-9,15-19`): noise-level (single stream −3 %, C16 +5 %).
  - **MTP K=2** (`SPEC_K=2`): prose and C32 +11 %, but code −10 % and structured −12 %. K=3 stays for code-heavy
    output.
  - **`--max-num-batched-tokens 32768`:** the memory guard tripped on all four ranks at boot, with 0.8-1.5 GiB free
    while FlashInfer autotuned the new token buckets.
  - **FlashInfer CUTLASS MXFP4×MXFP8 MoE** (`--moe-backend flashinfer_cutlass`): decode −3 to −9 %, no prefill gain.
  - **`NCCL_GRAPH_MIXING_SUPPORT=0`:** faster captured all-reduces, but NCCL documents it as unsafe when an eager
    collective runs while a graph launch is outstanding, which async scheduling does.
  - **Marlin W4A8** (`VLLM_MARLIN_INPUT_DTYPE=fp8`): +4-8 % prefill and NLL-neutral, but held: activation
    quantization untested on agent tasks.
  - **Other engines** on this model and topology at the time: SGLang + DFlash lost 43 % single-stream prose; the
    others were not viable here or lacked multi-node TP. SparkRing's MiMo profile was not run: its RoCEnante path
    needs ConnectX forwarding (host changes) on a switchless ring.
- 2026-10-01:
  - **Eager TP all-reduce between PIECEWISE graphs** (a minimal port of open vllm-project/vllm#48877): shorter NCCL kernels, but
    launch gaps raised GPU idle from 4.9 to 7.2 %. No gain.
  - **A capture-only NCCL communicator (`graphUsageMode=1`):** +1-2 %, superseded by RoCEnante.
  - **Dynamic MTP depth** (`num_speculative_tokens_per_batch_size`): only C32 prose gained (+4 %), short-code C16
    fell 9 %, and 1 of 64 garbage-probe responses switched script mid-sentence.
  - **DFlash 7 drafter:** code / structured +22 / +46 %, but prose −29 %, sampled-prose aggregates −20 to −30 %, and
    a slower agent loop.
  - **Column-parallel MTP `eh_proj`:** the GEMM saving was eaten by graph-captured all-gathers.
  - **Marlin FP8 W8A16 for the block-FP8 layers:** decode step −1.6-1.8 % but prefill 3-4 % slower. Net within noise.
  - **FP8 KV cache** (vllm-project/vllm#58128 ported onto our kernel): held on 2026-10-01 for short contexts; adopted
    on 2026-10-03 with its own launch tuning (see [Long context](#long-context-bf16-kv-vs-fp8-kv-2026-10-03)).
  - **128 seats:** still scales (C128 sampled prose ~815 tok/s, short code ~1,400) but not adopted, because the KV
    pool still caps long-context concurrency.

- 2026-10-03:
  - **bf16 attention launch re-tune at long context:** no gain. The production launch already reads KV at 240-255
    GB/s for one sequence; the best alternative was 0.99-1.03x, and only two concurrent 500K sequences gained
    (1.17x).
  - **Calibrated FP8 KV scales:** `--calculate-kv-scales` does not exist in this image's vLLM, so the scales stay 1.0.
  - **The 7-token DFlash drafter at long context:** not run. Its per-step cost on 2026-10-01 (43.5 vs 32 ms at ~77K)
    is drafter and verify compute, which does not shrink at long context, so MTP K=3 stays ahead.

## Rollback

Every rollback is the same sequence: `launcher/ring-up.sh down`, reboot all four nodes, then
`PROFILE=<profile> launcher/ring-up.sh up`.

- **bf16 KV cache:** `launcher/profile.bf16-20261001.env`, the production profile of 2026-10-01 to 2026-10-03.
- **Without RoCEnante:** `launcher/profile.rollback-20260930.env`, the production profile of 2026-09-30 to
  2026-10-01, with every all-reduce on NCCL. Changing `MIMO_ROCE_RD=1` to `MIMO_ROCE_RD=0` in `VLLM_ENV` does the
  same with the overlay still mounted.
- **Launch-day profile:** `launcher/profile.launch-20260929.env`.
- **Single features:**
  - `MIMO_DIFFKV_TUNE=0` (stock attention launch) or `MIMO_DIFFKV_QK_SPLIT=0` in `VLLM_ENV`;
  - drop an overlay from `VLLM_PATCHES_EXTRA`;
  - `NCCL_DUAL=0` (two HCAs, routing flags off).

Keep the previous profile file next to the live one; that is the whole rollback state.

## Limitations and caveats

- **RoCEnante is labelled research-only upstream.**
  - **Failure mode:** fail-stop. A collective that times out poisons the runtime; the next step's health check
    raises, and the engine dies as it would on an NCCL timeout.
  - **No automatic restart:** containers run with `--restart no`. The liveness shim returns 503, a load balancer
    should evict the lane, and recovery is a reboot plus `ring-up.sh up`. Our `@reboot` owner does exactly that.
  - **What to watch:** `docker logs vllm_mimo26 | grep -ciE "RoCE (proxy failed|collective on rank)|poisoned|timed
    out waiting"` on every rank (`bench/post-boot-check.sh` prints it).
- **HCA pairing is hardcoded for this cabling.**
  - `mimo_roce_rd.py` assumes ranks 0-1-2-3-0 with each node's port 0 cabled to the next node's port 1. For other
    cablings, edit `_HCAS`, `_PAIRS_A` and `_PAIRS_B` and regenerate that overlay's `SHA256SUMS`.
  - It only engages at TP=4; any other group size disables it.
- **The overlays are tied to this image.** They are copies of the image's own vLLM files, and the launcher refuses
  any other image ID. A new image means re-deriving every overlay from that image's copies.
- **The image's loader is closed source** (myllmbox's `libmbx_loader.so`, `--load-format mbx`). The myllmbox recipe
  repository carried no licence file when we checked; we use their public image and none of their files.
- **Tool calls:** strict `tool_choice` does not get a grammar (xgrammar 0.2.7 in the image).
- **FP8 KV is the default since 2026-10-03.**
  - It doubles the pool to 7.93M tokens and decodes faster at long context (−10 % / −16 % engine step at ~500K /
    ~1M), with needles exact to ~1M.
  - The costs:
    - +~0.003 corpus NLL;
    - hot-turn TTFT +6-15 % and cold prefill +6-10 % at 250K-1M;
    - roughly neutral decode at short context.
  - Use `launcher/profile.bf16-20261001.env` where those costs matter more than long-context decode or capacity.
- **1M context and memory:**
  - KV 30 GB per rank leaves 32-35 GiB MemAvailable per node after boot.
  - The boot transient dips to ~10.3 GiB on rank 0 and ~12.6 GiB on the others; the lowest reading while serving
    was ~22 GiB on rank 0 and ~25 GiB on the others.
  - The host memory guard stops its own rank below 4 GiB, or after three samples in a row below 6 GiB. It does not
    stop the peers; run `ring-up.sh down` after a trip.
  - The 7.93M-token pool (FP8 KV) holds 7.56 concurrent 1M-token requests (bf16: 3.96M, 3.78). Many long sessions
    at once will queue, and each extra concurrent session slows the others (see the two-session table above).
  - Larger batched-token budgets did not fit (see above).
- **Measure only on freshly rebooted nodes.** GB10 unified memory fragments, which can cost a large share of
  throughput.
- **Speculation:** MTP K=3 suits code-heavy output. Prose-heavy output may prefer K=2, code-only output DFlash.
- **vllm-project/vllm#46669** (corrupted tokens with async scheduling + MTP at concurrency > 1) did not reproduce on this build:
  0 of 128 at C32. Async scheduling stays on; the garbage probe is in `bench/`.
- **llm-inference-bench's loop guard** tripped more often on RoCEnante boots in its duration-bound cells at 64K
  context: 8 of 25 cells vs 1 of 17 on NCCL boots.
  - The collectives were verified correct (shadow check and stress test above).
  - NCCL boots looped on the same cells too, only less often.
  - At equal output length (`bench/loop-probe.py`, 40 × 6,000 tokens) neither engine looped.
  - We read it as a length confound of duration-bound cells on a faster engine. It stays recorded as not fully
    explained.
- **Scope of the numbers:**
  - one cluster, single boots per configuration;
  - the receipts stay on our side; this README summarises them;
  - NLL figures are on a private corpus, so compare arms with your own (`bench/ppl-probe.py`).

## Reproducing the numbers

Run everything from rank 0's node, pinned to the A725 cores, after a clean reboot and `ring-up.sh up`. Results land
in `$HOME/mimo26-bench/<label>/`.

```bash
KEYS=/etc/mimo26/api-keys; URL=http://127.0.0.1:8015
# "today" column (64 seats: larger concurrency lists, rotated multi-turn instructions, 4 garbage rounds)
CONC=1,4,16,32,64 RUNGS=1,8,16,32,64 LONG_ROTATE=1 GARBAGE_ROUNDS=4 \
  taskset -c 0-4,10-14 bench/ab-suite.sh today $URL mimo-v2.6-flash $KEYS gates,agent,decode,ladder,longctx,garbage,prefill
bench/qualify.sh $HOME/mimo26-bench/today-qual $KEYS          # needles, concurrent long prompts, image
bench/lib-bench.sh today $KEYS                                 # llm-inference-bench @ a50025a3
# "launch" column: boot launcher/profile.launch-20260929.env after a reboot, then the same with the defaults
taskset -c 0-4,10-14 bench/ab-suite.sh launch $URL mimo-v2.6-flash $KEYS gates,agent,decode,ladder,longctx,garbage,prefill
python3 bench/compare-arms.py $HOME/mimo26-bench               # one table across arms
# long context (2026-10-03): one or two sessions, cold ~prefix then two hot turns; ~0.84 tokens per target unit,
# so LONG_PREFIX 300000 / 600000 / 1135000 give ~250K / ~500K / ~950K
LONG_CONC=1 LONG_ROTATE=1 LONG_PREFIX=600000 taskset -c 0-4,10-14 bench/ab-suite.sh lc500k $URL mimo-v2.6-flash $KEYS longctx
```

| table row | script / cell |
|---|---|
| needles, concurrent long prompts, image | `bench/qualify.sh` (uses `bench/niah.py`) |
| cold prefill | `ab-suite.sh` cell `prefill` (`bench/prefill.py`) |
| RigMark single stream and short-code aggregates | cell `decode` (needs a RigMark checkout at `RM`, default `~/rigmark-othexmr`) |
| sampled-prose aggregates | cell `ladder` (`bench/conc-ladder.py`) |
| ~77K multi-turn | cell `longctx` (`bench/shaped.py`, multi-turn). Compare `step_ms` rather than decode tok/s: a repeated instruction is drafted at ~100 % acceptance. |
| agent loop | cell `agent` (`bench/agent-loop-probe.py`) |
| garbage-token probe | cell `garbage` (`bench/garbage-probe.py`) |
| llm-inference-bench | `bench/lib-bench.sh`, summary with `bench/lib-summary.py` |
| decode-step trace | cell `profile` (`bench/profile-decode.py`). Needs `--profiler-config.profiler=torch,--profiler-config.torch_profiler_dir=/cache/profiles,--profiler-config.torch_profiler_with_stack=false` appended to `VLLM_EXTRA`; analyse the trace with `bench/analyze-trace.py`. |
| NLL | cell `ppl` with `PPL_FILES=<your corpus>`, then `ppl-probe.py compare` |
| loop rate at equal length | `bench/loop-probe.py` |
| tool-parser bug and fix | `bench/tools-unit.sh` |
| long-context step / TTFT | cell `longctx` with `LONG_CONC` and `LONG_PREFIX` (see above) |
| FP8 / bf16 attention launch sweeps | `kernel-tuning/attn_fp8_dec_sweep.py`, `kernel-tuning/attn_bf16_dec_sweep.py` (one GPU, `stable_timer.py`) |
| collectives, kernels | `kernel-tuning/` (see its README) |

## Layout

```
README.md  LICENSE (MIT)  LICENSES/Apache-2.0.txt  .gitattributes (keeps the patch and overlays byte-exact)
launcher/      vllm-rank.sh (one rank), ring-up.sh (owner: up/check/down/status), load-hosts.sh,
               hosts.env.example, profile.env (production, FP8 KV), profile.bf16-20261001.env,
               profile.rollback-20260930.env, profile.launch-20260929.env, memory-guard.py, liveness-vllm.py,
               fetch-checkpoint.py
overlays/      vllm-patches  vllm-patches-attn2  vllm-patches-fp8kv  vllm-patches-fp8cfg  vllm-patches-tools
               vllm-patches-roce  NOTICE
nccl/          nccl-2.30.7-dual-pci-domain.patch + dual-pci-domain.json (SparkRing, unchanged), build-nccl.sh,
               PROVENANCE.md
bench/         ab-suite.sh qualify.sh post-boot-check.sh lib-bench.sh tools-unit.sh + the Python probes
kernel-tuning/ attention, FP8 GEMM, NCCL and RoCEnante microbenchmarks + README
```

## Credits

- **Xiaomi MiMo team:** MiMo-V2.6-Flash-MOPD.
- **myllmbox:** the `mimo-v26-flash-cluster-vllm` image and recipe that this stack starts from (mbx loader, split-KV
  spec-verify attention, non-chained MTP; their recipe flags at TP=4).
- **vLLM project and contributors:**
  - the engine, and the files the overlays modify;
  - ported pull requests vllm-project/vllm#58019 (MiMo tool parser) and vllm-project/vllm#58235 (MiMo ViT sink);
  - the FP8 KV cache for Triton DiffKV, vllm-project/vllm#58128, which `vllm-patches-fp8kv` ports;
  - related work vllm-project/vllm#58177, vllm-project/vllm#59085 and vllm-project/vllm#58141.
- **local-inference-lab:** [b12x](https://github.com/local-inference-lab/b12x) and its RoCEnante RDMA collectives,
  which our recursive doubling composes; and llm-inference-bench.
- **SparkRing ([FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring)):** the switchless-cycle and
  dual-PCI-domain NCCL patch.
- **NVIDIA:** NCCL.
- **RiNGSiDE (othexmr):** the pattern of preloading a patched NCCL into vLLM on a switchless ring (`LD_PRELOAD` +
  `VLLM_NCCL_SO_PATH`).
- **RigMark** (alexellis/rigmark; we ran the othexmr fork).

## Licence

Our own files are licensed under the MIT License ([`LICENSE`](LICENSE)). That covers the launcher, the bench scripts,
our kernel-tuning scripts and the documentation.

Third-party material keeps its licence. The Apache-2.0 text is in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
- **`overlays/`:** modified vLLM files (Apache-2.0, see `overlays/NOTICE`). Our changes to them, including the new
  `vllm-patches-roce/.../mimo_roce_rd.py`, are Apache-2.0 as well.
- **vLLM's own files in `kernel-tuning/`:** `benchmark_w8a8_block_fp8.py`, and `triton_unified_attention_diffkv_qksplit.py`
  (a modified copy of vLLM's DiffKV kernel). Both are Apache-2.0.
- **`nccl/`:** SparkRing's patch (Apache-2.0) against NVIDIA NCCL, which carries its own `LICENSE.txt`.
- **Not included:** b12x, the checkpoint, the image and the benchmark tools keep their own terms.
