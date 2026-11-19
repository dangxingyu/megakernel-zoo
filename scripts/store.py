#!/usr/bin/env python
"""Directories on the the cloud object store, from inside a machine, with every byte checked.

    python scripts/cw/store.py put LOCAL_DIR s3://bucket/prefix [LEDGER_ROOT]   # upload, MANIFEST.json last
    python scripts/cw/store.py get s3://bucket/prefix LOCAL_DIR      # download and check against the manifest
    python scripts/cw/store.py has s3://bucket/prefix                # exit 0 when a complete manifest is there
    python scripts/cw/store.py roundtrip s3://bucket/prefix          # self test: put, get, compare, timings
    python scripts/cw/store.py ledger s3://bucket/root               # what the ledger says is stored, and the total
    python scripts/cw/store.py reconcile s3://bucket/root            # add ledger records for complete prefixes without one
    python scripts/cw/store.py du s3://bucket/root                   # what a listing of the root finds, and the total
    python scripts/cw/store.py uploads s3://bucket/root [SECONDS]    # unfinished multipart uploads and the bytes of
                                                                     # their parts; abort those older than SECONDS
    python scripts/cw/store.py verify s3://bucket/prefix             # read a stored prefix back: sizes and sha256
    python scripts/cw/store.py verify-all s3://bucket/root           # the same for every complete prefix under a root

Accounting: `put ... LEDGER_ROOT` records the prefix, its file count and bytes as one small object
LEDGER_ROOT/ledger/<prefix with __ for />.json, after the manifest. `ledger` sums those records; `du` lists the
root itself (only our own prefix) and sums object sizes, so the two can be compared.

The endpoint does not verify checksums and uploads have hung with no error, so nothing here trusts a call:

- every part (64 MiB) is sent with its own MD5 kept locally; the ETag the store returns for the part must equal
  it, and the object's final ETag must equal the multipart ETag computed from the local file;
- MANIFEST.json (path, size, sha256 per file) is written only after every file passed, so a prefix without it
  is an unfinished upload and readers ignore it; `get` recomputes sha256 of what it downloaded;
- every request has socket timeouts, every part a wall-clock limit, and a part is retried a few times before
  the command fails (exit 77), so a stalled store ends the stage instead of the job's hours.

Runs in its own small venv with boto3 (scripts/cw/entry.sh), not in the project environment. Credentials and
the endpoint come from the machine's environment (AWS_*), which this never prints.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PART = 64 << 20
PART_SECONDS = 300.0  # a healthy 64 MiB part takes seconds
ATTEMPTS = 4
THREADS = 16
MANIFEST = "MANIFEST.json"
NO_WRITE_BUCKETS = ("<bucket>",)  # the main shared bucket, nearly full: we read there at most
EXIT_STALLED = 77


_OPEN: dict[tuple[str, str], str] = {}  # (bucket, key) -> id of a multipart upload this process has open


def _abort_open(s3=None) -> None:
    """Abort every multipart upload this process still has open (their parts hold space no listing shows)."""
    s3 = s3 or client()
    for (bucket, key), upload in list(_OPEN.items()):
        try:
            s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload)
            print(json.dumps({"STORE_ABORTED_UPLOAD": key}), flush=True)
        except Exception as error:
            print(json.dumps({"STORE_ABORT_FAILED": key, "error": type(error).__name__}), flush=True)
        _OPEN.pop((bucket, key), None)


def client():
    import boto3
    from botocore.config import Config

    # Virtual-host addressing (the endpoint refuses path style); no automatic CRC trailers, which
    # S3-compatible stores may reject; socket timeouts so a dead connection raises.
    return boto3.client("s3", config=Config(
        s3={"addressing_style": "virtual"}, connect_timeout=20, read_timeout=120,
        retries={"max_attempts": 3, "mode": "standard"}, max_pool_connections=4 * THREADS,
        request_checksum_calculation="when_required", response_checksum_validation="when_required"))


def split(url: str) -> tuple[str, str]:
    bucket, _, key = url.removeprefix("s3://").partition("/")
    return bucket, key.rstrip("/")


def _timed(pool, fn, *args, what: str):
    """fn(*args) with a wall-clock limit and retries; a call that neither returns nor raises ends the process."""
    for attempt in range(ATTEMPTS):
        future = pool.submit(fn, *args)
        try:
            return future.result(timeout=PART_SECONDS)
        except concurrent.futures.TimeoutError:
            print(json.dumps({"STORE_STALLED": what, "attempt": attempt + 1}), flush=True)
            if attempt == ATTEMPTS - 1:
                # The stalled call cannot be cancelled, so the process ends; first give back the parts already sent.
                aborter = concurrent.futures.ThreadPoolExecutor(1)
                try:
                    aborter.submit(_abort_open).result(timeout=120)
                except Exception:
                    print(json.dumps({"STORE_ABORT_FAILED": "timeout"}), flush=True)
                os._exit(EXIT_STALLED)
        except Exception as error:
            print(json.dumps({"STORE_RETRY": what, "attempt": attempt + 1, "error": f"{type(error).__name__}: {str(error)[:200]}"}), flush=True)
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(2 * (attempt + 1))


def file_digests(path: Path) -> tuple[str, list[bytes]]:
    sha, parts = hashlib.sha256(), []
    with open(path, "rb") as handle:
        while chunk := handle.read(PART):
            sha.update(chunk)
            parts.append(hashlib.md5(chunk, usedforsecurity=False).digest())
    return sha.hexdigest(), parts or [hashlib.md5(b"", usedforsecurity=False).digest()]


def put_file(s3, pool, path: Path, bucket: str, key: str) -> dict:
    size = path.stat().st_size
    sha, part_md5 = file_digests(path)
    if size <= PART:
        data = path.read_bytes()
        etag = _timed(pool, lambda: s3.put_object(Bucket=bucket, Key=key, Body=data)["ETag"].strip('"'), what=key)
        if etag != part_md5[0].hex():
            raise RuntimeError(f"{key}: ETag {etag} is not the body's MD5")
        return {"size": size, "sha256": sha}
    upload = _timed(pool, lambda: s3.create_multipart_upload(Bucket=bucket, Key=key), what=key)["UploadId"]
    _OPEN[(bucket, key)] = upload

    def send(number: int) -> dict:
        with open(path, "rb") as handle:
            handle.seek((number - 1) * PART)
            data = handle.read(PART)
        etag = s3.upload_part(Bucket=bucket, Key=key, UploadId=upload, PartNumber=number, Body=data)["ETag"].strip('"')
        if etag != part_md5[number - 1].hex():
            raise RuntimeError(f"{key} part {number}: ETag {etag} is not the part's MD5")
        return {"PartNumber": number, "ETag": etag}

    try:
        with concurrent.futures.ThreadPoolExecutor(THREADS) as senders:
            done = list(senders.map(lambda n: _timed(pool, send, n, what=f"{key}#{n}"), range(1, len(part_md5) + 1)))
        final = _timed(pool, lambda: s3.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload,
                                                                  MultipartUpload={"Parts": done}), what=key)
    except BaseException:  # a part failed its check, kept failing, or the run was interrupted: no parts stay behind
        try:
            s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload)
            print(json.dumps({"STORE_ABORTED_UPLOAD": key}), flush=True)
        except Exception as error:
            print(json.dumps({"STORE_ABORT_FAILED": key, "error": type(error).__name__}), flush=True)
        _OPEN.pop((bucket, key), None)
        raise
    _OPEN.pop((bucket, key), None)
    expected = hashlib.md5(b"".join(part_md5), usedforsecurity=False).hexdigest() + f"-{len(part_md5)}"
    head = _timed(pool, lambda: s3.head_object(Bucket=bucket, Key=key), what=key)
    if final["ETag"].strip('"') != expected or head["ETag"].strip('"') != expected or head["ContentLength"] != size:
        raise RuntimeError(f"{key}: stored ETag/size {head['ETag']}/{head['ContentLength']} != {expected}/{size}")
    return {"size": size, "sha256": sha}


def put(local: str, url: str, ledger_root: str | None = None) -> dict:
    s3, (bucket, prefix), root = client(), split(url), Path(local)
    if bucket in NO_WRITE_BUCKETS:
        raise SystemExit(f"{url}: nothing of ours is written to {bucket}")
    files = sorted(p for p in root.rglob("*") if p.is_file())  # symlinks to files are followed (HF cache layout)
    began, manifest = time.time(), {}
    with concurrent.futures.ThreadPoolExecutor(4 * THREADS) as pool, concurrent.futures.ThreadPoolExecutor(4) as outer:
        results = outer.map(lambda p: put_file(s3, pool, p, bucket, f"{prefix}/{p.relative_to(root).as_posix()}"), files)
        for path, entry in zip(files, results):
            manifest[path.relative_to(root).as_posix()] = entry
        body = json.dumps({"files": manifest, "written_at": time.time()}, indent=1, sort_keys=True).encode()
        etag = _timed(pool, lambda: s3.put_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}", Body=body)["ETag"].strip('"'), what=MANIFEST)
        if etag != hashlib.md5(body, usedforsecurity=False).hexdigest():
            raise RuntimeError("manifest ETag mismatch")
    total = sum(e["size"] for e in manifest.values())
    seconds = time.time() - began
    report = {"STORE_PUT": url, "files": len(manifest), "gb": round(total / 1e9, 3), "seconds": round(seconds, 1),
              "mb_s": round(total / 1e6 / max(seconds, 1e-6), 1)}
    if ledger_root:
        lbucket, lprefix = split(ledger_root)
        record = {"prefix": url.rstrip("/"), "files": len(manifest), "bytes": total, "written_at": time.time(),
                  "job": os.environ.get("QZ_JOB"), "cluster": os.environ.get("QZ_CLUSTER")}
        name = url.rstrip("/").removeprefix(ledger_root.rstrip("/") + "/").removeprefix("s3://").replace("/", "__")
        s3.put_object(Bucket=lbucket, Key=f"{lprefix}/ledger/{name}.json", Body=json.dumps(record, sort_keys=True).encode())
    print(json.dumps(report), flush=True)
    return report


def _list(s3, bucket: str, prefix: str):
    token = None
    while True:
        page = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, **({"ContinuationToken": token} if token else {}))
        yield from page.get("Contents", [])
        token = page.get("NextContinuationToken")
        if not token:
            return


def ledger(root: str) -> dict:
    """Sum of the ledger records: what was stored on purpose, by first path component."""
    s3, (bucket, prefix) = client(), split(root)
    rows = [json.loads(s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()) for o in _list(s3, bucket, f"{prefix}/ledger/")]
    groups: dict[str, list[int]] = {}
    for row in sorted(rows, key=lambda r: r["prefix"]):
        rel = row["prefix"].removeprefix(root.rstrip("/") + "/")
        print(f"LEDGER {row['bytes'] / 1e9:10.3f} GB {row['files']:6d} files  {rel}  ({row.get('cluster')}, {row.get('job')})")
        entry = groups.setdefault(rel.split("/", 1)[0], [0, 0])
        entry[0] += row["bytes"]
        entry[1] += 1
    report = {"LEDGER_TOTAL": root, "entries": len(rows), "gb": round(sum(r["bytes"] for r in rows) / 1e9, 3),
              "by_group_gb": {k: round(v[0] / 1e9, 3) for k, v in sorted(groups.items())}}
    print(json.dumps(report), flush=True)
    return report


def reconcile(root: str) -> None:
    """Write the ledger record of every complete prefix under the root that has none (prefixes stored before
    the ledger existed, or by a machine that died between the manifest and the record)."""
    s3, (bucket, prefix) = client(), split(root)
    have = {o["Key"] for o in _list(s3, bucket, f"{prefix}/ledger/")}
    added = 0
    for obj in _list(s3, bucket, prefix + "/"):
        if not obj["Key"].endswith("/" + MANIFEST):
            continue
        rel = obj["Key"][len(prefix) + 1: -len(MANIFEST) - 1]
        key = f"{prefix}/ledger/{rel.replace('/', '__')}.json"
        if key in have:
            continue
        files = json.loads(s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())["files"]
        record = {"prefix": f"s3://{bucket}/{prefix}/{rel}", "files": len(files), "bytes": sum(f["size"] for f in files.values()),
                  "written_at": obj["LastModified"].timestamp(), "job": "reconciled", "cluster": None}
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(record, sort_keys=True).encode())
        added += 1
    print(json.dumps({"LEDGER_RECONCILED": root, "added": added}), flush=True)


def du(root: str) -> dict:
    """Sum of object sizes under the root (a listing of our own prefix only), by the first two path components."""
    s3, (bucket, prefix) = client(), split(root)
    groups: dict[str, list[int]] = {}
    count = total = 0
    for obj in _list(s3, bucket, prefix + "/"):
        rel = obj["Key"][len(prefix) + 1:].split("/")
        entry = groups.setdefault("/".join(rel[:2]) if len(rel) > 2 else rel[0], [0, 0])
        entry[0] += obj["Size"]
        entry[1] += 1
        count += 1
        total += obj["Size"]
    for key, (size, n) in sorted(groups.items()):
        if size >= 1e7:
            print(f"DU {size / 1e9:10.3f} GB {n:6d} objects  {key}")
    report = {"DU_TOTAL": root, "objects": count, "gb": round(total / 1e9, 3)}
    print(json.dumps(report), flush=True)
    return report


def prefixes(s3, bucket: str, prefix: str) -> list[str]:
    """Every complete stored prefix under a root (those with a manifest), relative to the root."""
    return sorted(o["Key"][len(prefix) + 1: -len(MANIFEST) - 1] for o in _list(s3, bucket, prefix + "/")
                  if o["Key"].endswith("/" + MANIFEST))


def verify(url: str) -> dict:
    """Read every object of a stored prefix back and compare size and sha256 with its manifest (nothing is kept)."""
    s3, (bucket, prefix) = client(), split(url)
    manifest = read_manifest(s3, bucket, prefix)
    if manifest is None:
        raise SystemExit(f"{url}: no {MANIFEST}")

    def check(item) -> str | None:
        name, entry = item
        sha, size = hashlib.sha256(), 0
        for chunk in s3.get_object(Bucket=bucket, Key=f"{prefix}/{name}")["Body"].iter_chunks(1 << 24):
            sha.update(chunk)
            size += len(chunk)
        return None if (size, sha.hexdigest()) == (entry["size"], entry["sha256"]) else name

    with concurrent.futures.ThreadPoolExecutor(THREADS) as pool:
        bad = [name for name in pool.map(check, manifest.items()) if name]
    report = {"STORE_VERIFY": url, "files": len(manifest), "gb": round(sum(e["size"] for e in manifest.values()) / 1e9, 3),
              "mismatches": bad}
    print(json.dumps(report), flush=True)
    return report


def verify_all(root: str) -> None:
    s3, (bucket, prefix) = client(), split(root)
    reports = [verify(f"s3://{bucket}/{prefix}/{rel}") for rel in prefixes(s3, bucket, prefix)]
    bad = sum(len(r["mismatches"]) for r in reports)
    print(json.dumps({"STORE_VERIFY_ALL": root, "prefixes": len(reports), "files": sum(r["files"] for r in reports),
                      "gb": round(sum(r["gb"] for r in reports), 3), "mismatches": bad}), flush=True)
    if bad:
        raise SystemExit(1)


def uploads(root: str, abort_older_than_s: float | None = None) -> dict:
    """Unfinished multipart uploads under the root: their parts hold space that object listings do not show.
    With a limit, those begun longer ago than it are aborted (a live upload of ours finishes a file in minutes)."""
    s3, (bucket, prefix) = client(), split(root)
    found, kwargs, now = [], {}, time.time()
    while True:
        page = s3.list_multipart_uploads(Bucket=bucket, Prefix=prefix + "/", **kwargs)
        found += page.get("Uploads", [])
        if not page.get("IsTruncated"):
            break
        kwargs = {"KeyMarker": page["NextKeyMarker"], "UploadIdMarker": page["NextUploadIdMarker"]}
    total = aborted = 0
    for up in found:
        size, marker = 0, 0
        while True:
            parts = s3.list_parts(Bucket=bucket, Key=up["Key"], UploadId=up["UploadId"], PartNumberMarker=marker)
            size += sum(p["Size"] for p in parts.get("Parts", []))
            if not parts.get("IsTruncated"):
                break
            marker = parts["NextPartNumberMarker"]
        age = now - up["Initiated"].timestamp()
        total += size
        gone = abort_older_than_s is not None and age > abort_older_than_s
        if gone:
            s3.abort_multipart_upload(Bucket=bucket, Key=up["Key"], UploadId=up["UploadId"])
            aborted += 1
        print(f"OPEN_UPLOAD {size / 1e9:8.3f} GB  age {age / 3600:6.1f} h  {'ABORTED' if gone else 'left'}  {up['Key']}", flush=True)
    report = {"OPEN_UPLOADS": root, "count": len(found), "gb_in_parts": round(total / 1e9, 3), "aborted": aborted}
    print(json.dumps(report), flush=True)
    return report


def lifecycle(root: str) -> None:
    """The bucket's expiry rules, if we may read them: does anything delete objects under our prefix?"""
    s3, bucket = client(), split(root)[0]
    try:
        rules = s3.get_bucket_lifecycle_configuration(Bucket=bucket).get("Rules", [])
        shown = [{"id": r.get("ID"), "status": r.get("Status"), "prefix": (r.get("Filter") or {}).get("Prefix", r.get("Prefix")),
                  "expiration": r.get("Expiration"), "abort_multipart_days": (r.get("AbortIncompleteMultipartUpload") or {}).get("DaysAfterInitiation")}
                 for r in rules]
        print(json.dumps({"LIFECYCLE": bucket, "rules": shown}, default=str), flush=True)
    except Exception as error:
        print(json.dumps({"LIFECYCLE": bucket, "unreadable": f"{type(error).__name__}: {str(error)[:200]}"}), flush=True)


def read_manifest(s3, bucket: str, prefix: str) -> dict | None:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}")["Body"].read())["files"]
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as error:
        if getattr(error, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise


def get_file(s3, pool, bucket: str, key: str, target: Path, entry: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    ranges = [(s, min(s + PART, entry["size"]) - 1) for s in range(0, entry["size"], PART)]
    with open(tmp, "wb") as handle:
        handle.truncate(entry["size"])

    def fetch(span) -> None:
        data = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={span[0]}-{span[1]}")["Body"].read()
        if len(data) != span[1] - span[0] + 1:
            raise RuntimeError(f"{key}: short read of {span}")
        with open(tmp, "r+b") as handle:
            handle.seek(span[0])
            handle.write(data)

    with concurrent.futures.ThreadPoolExecutor(THREADS) as readers:
        list(readers.map(lambda span: _timed(pool, fetch, span, what=f"{key}@{span[0]}"), ranges))
    sha, _ = file_digests(tmp)
    if sha != entry["sha256"] or tmp.stat().st_size != entry["size"]:
        tmp.unlink()
        raise RuntimeError(f"{key}: downloaded sha256 {sha} != manifest {entry['sha256']}")
    os.replace(tmp, target)


def get(url: str, local: str) -> dict:
    s3, (bucket, prefix), root = client(), split(url), Path(local)
    manifest = read_manifest(s3, bucket, prefix)
    if manifest is None:
        raise SystemExit(f"{url}: no {MANIFEST} (absent or unfinished upload)")
    began = time.time()
    with concurrent.futures.ThreadPoolExecutor(4 * THREADS) as pool, concurrent.futures.ThreadPoolExecutor(4) as outer:
        list(outer.map(lambda item: get_file(s3, pool, bucket, f"{prefix}/{item[0]}", root / item[0], item[1]), manifest.items()))
    total = sum(e["size"] for e in manifest.values())
    seconds = time.time() - began
    report = {"STORE_GET": url, "files": len(manifest), "gb": round(total / 1e9, 3), "seconds": round(seconds, 1),
              "mb_s": round(total / 1e6 / max(seconds, 1e-6), 1)}
    print(json.dumps(report), flush=True)
    return report


def has(url: str) -> bool:
    bucket, prefix = split(url)
    return read_manifest(client(), bucket, prefix) is not None


def roundtrip(url: str) -> None:
    """Self test on a small directory: one multipart file, small files, a nested path, an empty file."""
    import shutil
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="cwstore-"))
    src, back = work / "src", work / "back"
    (src / "nested" / "deep").mkdir(parents=True)
    (src / "big.bin").write_bytes(os.urandom(150 << 20))  # 3 parts
    (src / "nested" / "small.json").write_text(json.dumps({"hello": "store"}))
    (src / "nested" / "deep" / "one-part.bin").write_bytes(os.urandom(5 << 20))
    (src / "empty").write_bytes(b"")
    target = f"{url.rstrip('/')}/roundtrip-{os.uname().machine}-{int(time.time())}"
    up = put(str(src), target)
    assert has(target)
    down = get(target, str(back))
    same = all((back / p.relative_to(src)).read_bytes() == p.read_bytes() for p in src.rglob("*") if p.is_file())
    # A prefix whose manifest is missing must read as absent.
    s3, (bucket, prefix) = client(), split(target)
    s3.delete_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}")
    unfinished_ignored = not has(target)
    for key in [o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/").get("Contents", [])]:
        s3.delete_object(Bucket=bucket, Key=key)
    left = s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/").get("KeyCount", 0)
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"STORE_ROUNDTRIP": target, "identical": same, "unfinished_ignored": unfinished_ignored,
                      "cleaned": left == 0, "up_mb_s": up["mb_s"], "down_mb_s": down["mb_s"]}), flush=True)
    if not (same and unfinished_ignored):
        raise SystemExit(1)


def main() -> int:
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "put":
        put(*args)
    elif cmd == "get":
        get(*args)
    elif cmd == "has":
        return 0 if has(*args) else 1
    elif cmd == "roundtrip":
        roundtrip(*args)
    elif cmd == "ledger":
        ledger(*args)
    elif cmd == "reconcile":
        reconcile(*args)
    elif cmd == "du":
        du(*args)
    elif cmd == "lifecycle":
        lifecycle(*args)
    elif cmd == "uploads":
        uploads(args[0], float(args[1]) if len(args) > 1 else None)
    elif cmd == "verify":
        if verify(*args)["mismatches"]:
            return 1
    elif cmd == "verify-all":
        verify_all(*args)
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
