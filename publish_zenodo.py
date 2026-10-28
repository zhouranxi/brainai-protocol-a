#!/usr/bin/env python3
"""Publish the Protocol A release to Zenodo.

Usage
-----
    export ZENODO_TOKEN=...        # personal access token, scopes: deposit:write deposit:actions
    python publish_zenodo.py --files-dir /path/to/upload --metadata zenodo_metadata.json
    python publish_zenodo.py --sandbox ...        # test against sandbox.zenodo.org
    python publish_zenodo.py --publish            # actually publish (otherwise it stops at the draft)

The script is idempotent and resumable: it records the deposition id and the
bucket URL in a state file next to the metadata, so an interrupted upload can be
restarted without re-uploading what already arrived.

Notes
-----
* The access token is read from the ZENODO_TOKEN environment variable. It is
  never written to disk.
* Uploading 2.4 GB over a residential connection can take a long time. The
  script prints progress and can be re-run safely after an interruption.
* By default the script stops after uploading and leaves the deposition in the
  draft state so that the metadata can be reviewed on zenodo.org before the
  record is made public. Pass --publish to publish in one go.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("This script needs `requests`: pip install requests")

PROD = "https://zenodo.org/api"
SANDBOX = "https://sandbox.zenodo.org/api"


def _token_from_config() -> str | None:
    """Read the access token from ~/.config/zenodo.json if it is there."""
    for path in (Path.home() / ".config" / "zenodo.json",
                 Path.home() / ".zenodo.json"):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            tok = data.get("token")
            if tok:
                print(f"Using token from {path}")
                return tok
    return None


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            b = fp.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files-dir", required=True,
                    help="directory containing the files to upload")
    ap.add_argument("--metadata", required=True,
                    help="JSON file with the deposition metadata")
    ap.add_argument("--sandbox", action="store_true",
                    help="target sandbox.zenodo.org instead of zenodo.org")
    ap.add_argument("--publish", action="store_true",
                    help="publish the record (otherwise it stays a draft)")
    ap.add_argument("--state", default=None,
                    help="state file for resumability (default: alongside the metadata)")
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-request timeout in seconds; 0 means no timeout (needed for 2 GB files)")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        token = _token_from_config()
    if not token:
        sys.exit("No Zenodo token found.\n"
                 "Either export ZENODO_TOKEN, or create one at\n"
                 "  https://zenodo.org/account/settings/applications/tokens/new/\n"
                 "with scopes 'deposit:write' and 'deposit:actions', and save it as\n"
                 "  ~/.config/zenodo.json   (a JSON object with a 'token' field)")

    base = SANDBOX if args.sandbox else PROD
    auth = {"Authorization": f"Bearer {token}",
            "User-Agent": "ProtocolA-ZenodoPublisher/1.0 (academic release)"}
    json_headers = {**auth, "Content-Type": "application/json"}
    timeout = args.timeout or None

    files_dir = Path(args.files_dir).resolve()
    meta_path = Path(args.metadata).resolve()
    state_path = Path(args.state) if args.state else meta_path.with_suffix(".state.json")

    files = sorted(p for p in files_dir.iterdir() if p.is_file())
    if not files:
        sys.exit(f"No files to upload in {files_dir}")
    print(f"Target      : {base}")
    print(f"Files       : {len(files)}")
    for p in files:
        print(f"  - {p.name:52s} {human(p.stat().st_size):>10s}")
    print(f"Total       : {human(sum(p.stat().st_size for p in files))}")
    print(f"Metadata    : {meta_path}")
    print()

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print(f"Resuming from state file: deposition {state.get('id')}")
        print(f"  already uploaded: {state.get('uploaded', [])}")
        print()

    uploaded = set(state.get("uploaded", []))

    # ---- 1. create (or reuse) the deposition ----
    if "id" in state and "bucket" in state:
        dep_id, bucket = state["id"], state["bucket"]
        print(f"Using existing deposition {dep_id}")
    else:
        r = requests.post(f"{base}/deposit/depositions", json=metadata,
                          headers=json_headers, timeout=timeout or 60)
        if r.status_code not in (200, 201):
            sys.exit(f"Failed to create deposition: {r.status_code}\n{r.text[:2000]}")
        dep = r.json()
        dep_id = dep["id"]
        bucket = dep["links"]["bucket"]
        state = {"id": dep_id, "bucket": bucket, "uploaded": []}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"Created deposition {dep_id}")
        print(f"  bucket: {bucket}")
        print(f"  draft URL: {base.replace('/api','')}/deposit/{dep_id}")
    print()

    # ---- 2. upload the files ----
    for p in files:
        if p.name in uploaded:
            print(f"[skip] {p.name} (already uploaded)")
            continue
        size = p.stat().st_size
        print(f"[put ] {p.name}  ({human(size)})")
        t0 = time.time()
        try:
            with p.open("rb") as fp:
                r = requests.put(f"{bucket}/{p.name}", data=fp, headers=auth,
                                 timeout=timeout)
        except requests.exceptions.RequestException as e:
            print(f"  !! transfer interrupted: {e}")
            print(f"  !! re-run this script to resume from {p.name}")
            return 2
        if r.status_code not in (200, 201):
            print(f"  !! upload failed: {r.status_code}\n{r.text[:1500]}")
            return 3
        dt = time.time() - t0
        rate = size / max(dt, 1e-6) / (1024 * 1024)
        print(f"  ok in {dt:.0f}s ({rate:.1f} MB/s)")
        uploaded.add(p.name)
        state["uploaded"] = sorted(uploaded)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # ---- 3. confirm the server saw the files ----
    r = requests.get(f"{base}/deposit/depositions/{dep_id}", headers=auth, timeout=60)
    r.raise_for_status()
    listed = r.json().get("files", [])
    print()
    print("Files on the server:")
    ok_all = True
    for f in listed:
        print(f"  {f.get('key')}  {human(f.get('size', 0))}  "
              f"checksum={f.get('checksum', '')[:16]}...")
        local = files_dir / f.get("key", "")
        if local.exists() and f.get("checksum"):
            import hashlib as _h
            h = _h.md5()
            with local.open("rb") as fp:
                for chunk in iter(lambda: fp.read(1 << 20), b""):
                    h.update(chunk)
            same = h.hexdigest() == f["checksum"]
            ok_all = ok_all and same
            print(f"      md5 local {'==' if same else '!='} server  "
                  f"{'OK' if same else 'MISMATCH'}")
    if not ok_all:
        print("  !! checksum mismatch — do not publish until resolved")
        return 6
    if len(listed) != len(files):
        print(f"  !! expected {len(files)} files, server reports {len(listed)}")
        return 4

    print()
    print(f"Deposition metadata (title): {r.json()['metadata'].get('title')}")
    print(f"Draft URL: {base.replace('/api','')}/deposit/{dep_id}")
    print("Checksum of what was uploaded (recomputed locally):")
    for p in files:
        print(f"  {sha256(p)}  {p.name}")

    if not args.publish:
        print()
        print("Not publishing (no --publish flag). Review the draft at the URL above,")
        print("then re-run with --publish, or publish it from the Zenodo web UI.")
        return 0

    print()
    print("Publishing...")
    r = requests.post(f"{base}/deposit/depositions/{dep_id}/actions/publish",
                      headers=auth, timeout=60)
    if r.status_code not in (200, 202):
        print(f"  !! publish failed: {r.status_code}\n{r.text[:1500]}")
        return 5
    out = r.json()
    print("  published.")
    print(f"  DOI : {out.get('doi')}")
    print(f"  URL : {out.get('links', {}).get('html')}")
    print(f"  concept DOI: {out.get('conceptdoi')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
