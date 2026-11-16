#!/usr/bin/env python3
"""Pull one image from Docker Hub without docker and unpack it into a directory (a root file system).

    python3 scripts/ink/oci_pull.py REPO@sha256:DIGEST ROOTFS [--arch amd64]

Anonymous pull (public images only; no credential is used or accepted). Layers are applied in order with OCI
whiteouts (`.wh.NAME` removes NAME of a lower layer, `.wh..wh..opq` empties the directory). The digest pins the
image: a tag is resolved once and printed, so a run records exactly what it unpacked. Standard library + tar.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ACCEPT = ",".join(["application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json",
                   "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"])


def token(repo: str) -> str:
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    return json.load(urllib.request.urlopen(url, timeout=60))["token"]


def fetch(repo: str, path: str, tok: str, accept: str = ACCEPT):
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/{path}", headers={"Authorization": f"Bearer {tok}", "Accept": accept})
    return urllib.request.urlopen(req, timeout=300)


def blob(repo: str, digest: str, target: Path) -> Path:
    for attempt in range(5):
        try:
            tok = token(repo)
            h = hashlib.sha256()
            with fetch(repo, f"blobs/{digest}", tok, "*/*") as r, open(target, "wb") as out:
                while chunk := r.read(8 << 20):
                    h.update(chunk); out.write(chunk)
            if "sha256:" + h.hexdigest() != digest:
                raise IOError(f"digest mismatch for {digest}")
            return target
        except Exception as exc:  # noqa: BLE001
            print(f"retry {digest[:19]} after {exc}", file=sys.stderr); time.sleep(5 * (attempt + 1))
    raise SystemExit(f"could not fetch {digest}")


def main() -> None:
    ref, rootfs = sys.argv[1], Path(sys.argv[2])
    arch = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else "amd64"
    repo, _, digest = ref.partition("@")
    if not digest:
        repo, _, tag = ref.partition(":"); digest = tag or "latest"
    tok = token(repo)
    man = json.load(fetch(repo, f"manifests/{digest}", tok))
    if "manifests" in man:
        pick = [m for m in man["manifests"] if m.get("platform", {}).get("architecture") == arch][0]
        print(f"MK_ENV image {repo} index {digest} -> {arch} {pick['digest']}")
        man = json.load(fetch(repo, f"manifests/{pick['digest']}", tok))
    layers = man["layers"]
    print(f"MK_ENV image {repo} layers={len(layers)} compressed_gb={sum(l['size'] for l in layers) / 1e9:.2f}")
    cache = rootfs.parent / (rootfs.name + ".blobs"); cache.mkdir(parents=True, exist_ok=True)
    config = json.load(fetch(repo, f"blobs/{man['config']['digest']}", tok, "*/*"))
    (rootfs.parent / (rootfs.name + ".config.json")).write_text(json.dumps(config, indent=1))
    t = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        files = list(ex.map(lambda l: blob(repo, l["digest"], cache / l["digest"].split(":")[1]), layers))
    print(f"MK_ENV image {repo} downloaded in {time.time() - t:.0f} s")
    t = time.time()
    rootfs.mkdir(parents=True, exist_ok=True)
    for layer, path in zip(layers, files):
        mt = layer["mediaType"]
        flag = "--zstd" if "zstd" in mt else "-z" if "gzip" in mt else ""
        listed = subprocess.run(["tar", *([flag] if flag else []), "-tf", str(path)], capture_output=True, text=True)
        if listed.returncode != 0:
            if layer["size"] < 1024:  # the well-known empty layer (a gzipped empty archive, often listed several times)
                continue
            raise SystemExit(f"cannot list layer {layer['digest']}: {listed.stderr[-300:]}")
        names = listed.stdout.splitlines()
        for name in names:
            base = os.path.basename(name.rstrip("/"))
            if not base.startswith(".wh."):
                continue
            parent = rootfs / os.path.dirname(name.rstrip("/"))
            if base == ".wh..wh..opq":
                if parent.is_dir():
                    for child in parent.iterdir():
                        shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
            else:
                victim = parent / base[len(".wh."):]
                if victim.is_dir() and not victim.is_symlink():
                    shutil.rmtree(victim)
                elif victim.exists() or victim.is_symlink():
                    victim.unlink()
        subprocess.run(["tar", *([flag] if flag else []), "-xf", str(path), "-C", str(rootfs), "--exclude=.wh.*",
                        "--no-same-owner", "--delay-directory-restore"], check=True)
    shutil.rmtree(cache, ignore_errors=True)
    env = config["config"].get("Env", [])
    print(f"MK_ENV image {repo} unpacked in {time.time() - t:.0f} s into {rootfs}")
    print("MK_ENV image_env " + json.dumps(env)[:3000])


if __name__ == "__main__":
    main()
