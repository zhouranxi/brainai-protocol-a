#!/usr/bin/env python3
"""Restore files that were stored as base64 text on the remote.

The GitHub tree API takes a blob's content as plain text and encodes it
itself. An earlier push passed an already base64-encoded string, so the remote
received the encoded text rather than the file. This walks the repository and
puts the affected files back.

Text files go through the Contents API with `encoding: base64`, which is the
correct way to carry arbitrary bytes. Binary files that Git LFS tracks are
left to git, because writing them through the API would replace the LFS
pointer with the raw bytes.

Usage:
    python repair_remote.py
"""
from __future__ import annotations

import base64
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = "zhouranxi/brainai-protocol-a"
MSG = ("Restore the files that were stored as base64 text\n\n"
       "The tree API takes the content of a blob as plain text and encodes it "
       "itself. The earlier push passed an already base64-encoded string, so the "
       "remote received the encoded text rather than the file.")


def api(method, path, data=None, token=""):
    url = f"https://api.github.com/repos/{REPO}{path}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if body:
        req.add_header("Content-Type", "application/json")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code in (502, 503) and attempt < 4:
                time.sleep(3 * (attempt + 1))
                continue
            raise


def main() -> int:
    token = subprocess.run(["gh", "auth", "token"], capture_output=True,
                            text=True).stdout.strip()
    root = pathlib.Path(__file__).resolve().parent

    tree = api("GET", "/git/trees/master?recursive=1", token=token)
    remote = {e["path"]: e for e in tree.get("tree", []) if e["type"] == "blob"}

    broken_text, broken_binary, ok = [], [], 0
    for path in sorted(remote):
        lp = root / path
        if not lp.exists() or lp.is_dir():
            continue
        local = lp.read_bytes()
        try:
            d = api("GET", f"/contents/{path}", token=token)
            remote_raw = base64.b64decode(d["content"])
        except Exception:
            continue
        if remote_raw == local:
            ok += 1
            continue
        # Was the remote file the base64 text of the local file?
        try:
            if base64.b64decode(remote_raw, validate=True) == local:
                (broken_binary if _is_binary(local) else broken_text).append(path)
        except Exception:
            pass

    print(f"intact           : {ok}")
    print(f"broken (text)    : {len(broken_text)}")
    print(f"broken (binary)  : {len(broken_binary)}")

    fixed = 0
    for path in broken_text:
        raw = (root / path).read_bytes()
        try:
            sha = api("GET", f"/contents/{path}", token=token)["sha"]
            api("PUT", f"/contents/{path}", {
                "message": MSG,
                "content": base64.b64encode(raw).decode(),
                "encoding": "base64",
                "sha": sha,
                "branch": "master",
            }, token=token)
            fixed += 1
        except Exception as e:
            print(f"  FAIL {path}: {e}")

    print(f"restored         : {fixed}")
    if broken_binary:
        print("\nLFS-tracked files that need a git push to restore:")
        for p in broken_binary:
            print("  ", p)
    return 0


def _is_binary(raw: bytes) -> bool:
    try:
        raw.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


if __name__ == "__main__":
    sys.exit(main())
