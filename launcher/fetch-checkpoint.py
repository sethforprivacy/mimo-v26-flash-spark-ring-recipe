#!/usr/bin/env python3
"""Fetch the pinned MiMo-V2.6-Flash-MOPD checkpoint once and verify it against the Hub's published digests.

  python3 -m venv ~/hf-venv && ~/hf-venv/bin/pip install -q huggingface_hub hf_xet
  ~/hf-venv/bin/python fetch-checkpoint.py [--dest /srv/models] [--verify-only]

Downloads XiaomiMiMo/MiMo-V2.6-Flash-MOPD at revision 2479e2d0029eca9a34cc7e7f55a121925f81908e to
<dest>/MiMo-V2.6-Flash-MOPD/<revision>/ (the launcher's MODEL_HOST), re-hashes every file against the digest the Hub
publishes for that exact revision (LFS files: sha256; small files: git-blob sha1), and writes a complete sha256
manifest next to the directory. Copy the directory to the other three nodes (rsync over the LAN or the fabric links)
and check each copy before its first boot:

  cd <dest>/MiMo-V2.6-Flash-MOPD/<revision> && sha256sum -c --quiet ../<revision>.sha256

--verify-only re-hashes an existing directory without downloading. Set HF_TOKEN if your account needs one.
Always pin the revision: two nodes with different weights under one name fail in ways a gate catches only late.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import pathlib
import sys
import time

REPO = "XiaomiMiMo/MiMo-V2.6-Flash-MOPD"
REVISION = "2479e2d0029eca9a34cc7e7f55a121925f81908e"


def digest_of(path, published):
    if len(published) == 64:          # LFS: sha256 of the content
        d = hashlib.sha256()
    else:                             # small file: git blob sha1 (header + content)
        d = hashlib.sha1()
        d.update(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            d.update(block)
    return d.hexdigest()


def sha256_of(path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            d.update(block)
    return d.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="/srv/models")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    import huggingface_hub as h

    started = time.time()
    target = pathlib.Path(a.dest) / a.repo.split("/")[-1] / a.revision
    info = h.HfApi().model_info(a.repo, revision=a.revision, files_metadata=True)
    if info.sha is not None and info.sha != a.revision:
        raise SystemExit(f"revision resolved to {info.sha}, asked for {a.revision}")
    want = {}
    for s in info.siblings:
        rel = pathlib.PurePosixPath(s.rfilename)
        if ".." in rel.parts or rel.is_absolute():
            raise SystemExit(f"refusing unsafe path from the Hub: {s.rfilename}")
        want[s.rfilename] = s.lfs.sha256 if s.lfs else s.blob_id
    print(f"{len(want)} published files at {a.repo}@{a.revision}", flush=True)
    if not a.verify_only:
        target.mkdir(parents=True, exist_ok=True)
        h.snapshot_download(a.repo, revision=a.revision, local_dir=target, max_workers=8)

    def check(name):
        p = target / name
        if not p.is_file():
            return f"MISSING {name}"
        if not want[name]:
            return f"NO PUBLISHED DIGEST {name}"
        return None if digest_of(p, want[name]) == want[name] else f"HASH MISMATCH {name}"

    with cf.ThreadPoolExecutor(max_workers=3) as pool:   # bound by storage I/O; a small pool is enough
        problems = [r for r in pool.map(check, sorted(want)) if r]
    if problems:
        print("\n".join(problems[:20]), file=sys.stderr)
        raise SystemExit(f"VERIFICATION FAILED: {len(problems)} of {len(want)} files")
    manifest = target.parent / f"{a.revision}.sha256"
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        names = sorted(want)
        lines = [f"{d}  {n}\n" for n, d in zip(names, pool.map(lambda n: sha256_of(target / n), names))]
    manifest.write_text("".join(lines))
    print(json.dumps({"repository": a.repo, "revision": a.revision, "target": str(target), "verified_files": len(want),
                      "bytes": sum((target / n).stat().st_size for n in want), "manifest": str(manifest),
                      "elapsed_s": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main()
