#!/usr/bin/env python3
"""Push the working tree to GitHub through the REST API.

`git push` needs the github.com git endpoint, which is unreachable from this
network (the proxy tunnel returns 502), while api.github.com works. This
script reproduces what a push does using the REST API:

  1. upload every changed blob (large files go through the LFS batch API);
  2. build a tree that mirrors the local index;
  3. create a commit on top of the remote branch;
  4. move the branch ref.

Files are taken from the local git index, so what is uploaded is exactly what
`git commit` would have recorded.

Usage:
    python push_via_api.py [--repo owner/name] [--branch master] [--message "..."]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://api.github.com"


def gh_token() -> str:
    tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
    if not tok:
        sys.exit("gh auth token returned nothing; run `gh auth login` first")
    return tok


class GitHub:
    def __init__(self, repo: str, token: str):
        self.repo = repo
        self.token = token

    def _req(self, method: str, path: str, data=None, raw=False, headers=None):
        url = path if path.startswith("http") else f"{API}{path}"
        body = None
        if data is not None:
            body = data if isinstance(data, bytes) else json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        if body is not None and not isinstance(body, bytes):
            req.add_header("Content-Type", "application/json")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    payload = r.read()
                return payload if raw else json.loads(payload or b"{}")
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:400]
                if e.code in (502, 503, 504) and attempt < 3:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise RuntimeError(f"{method} {url} -> {e.code}: {detail}") from None
            except urllib.error.URLError:
                if attempt < 3:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise

    # ---- LFS ----
    def lfs_upload(self, path: str, oid: str, size: int) -> str:
        """Upload one file through the LFS batch API and return its oid."""
        r = self._req("POST", f"{API}/repos/{self.repo}/git/lfs/objects/batch",
                      {"operation": "upload", "transfers": ["basic"],
                       "objects": [{"oid": oid, "size": size}]})
        objs = r.get("objects") or []
        if not objs:
            return oid
        obj = objs[0]
        if "actions" not in obj:
            return obj.get("oid", oid)
        hdr = obj["actions"]["upload"]["header"]
        url = obj["actions"]["upload"]["href"]
        with open(path, "rb") as fp:
            data = fp.read()
        req = urllib.request.Request(url, data=data, method="PUT")
        for k, v in hdr.items():
            req.add_header(k, v)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    resp.read()
                break
            except urllib.error.HTTPError:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        return obj.get("oid", oid)


def git(*args, binary=False):
    r = subprocess.run(["git", *args], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace')}")
    return r.stdout if binary else r.stdout.decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="zhouranxi/brainai-protocol-a")
    ap.add_argument("--branch", default="master")
    ap.add_argument("--message", default=None)
    args = ap.parse_args()

    gh = GitHub(args.repo, gh_token())

    # ---- what changed, according to the local index ----
    base = git("rev-parse", "HEAD").strip()
    remote = gh._req("GET", f"/repos/{args.repo}/git/ref/heads/{args.branch}")
    remote_sha = remote["object"]["sha"]
    print(f"local HEAD   : {base}")
    print(f"remote HEAD  : {remote_sha}")

    if base == remote_sha:
        print("already up to date")
        return 0

    entries = git("ls-tree", "-r", base).splitlines()
    changed = []
    for line in entries:
        meta, path = line.split("\t", 1)
        mode, otype, oid = meta.split()
        changed.append({"path": path, "mode": mode, "type": otype, "sha": oid})
    print(f"tree entries : {len(changed)}")

    # ---- upload blobs that are not on the remote yet ----
    remote_tree = gh._req("GET", f"/repos/{args.repo}/git/trees/{remote_sha}?recursive=1")
    remote_paths = {e["path"]: e["sha"] for e in remote_tree.get("tree", [])}

    uploaded = 0
    for e in changed:
        if remote_paths.get(e["path"]) == e["sha"]:
            continue
        # LFS-tracked files: upload the real bytes, keep the pointer as the blob.
        if e["type"] == "blob" and e["mode"] != "160000":
            raw = subprocess.run(["git", "cat-file", "-p", e["sha"]],
                                 capture_output=True).stdout
            if raw.startswith(b"version https://git-lfs.github.com/spec"):
                lfs_oid = raw.split(b"oid sha256:", 1)[1].split(b"\n", 1)[0].strip().decode()
                size = int(raw.split(b"size ", 1)[1].split(b"\n", 1)[0].strip())
                local = None
                for cand in (e["path"],):
                    if os.path.exists(cand):
                        local = cand
                if local:
                    print(f"  LFS {e['path']} ({size/1024/1024:.1f} MB)")
                    gh.lfs_upload(local, lfs_oid, size)
        # the blob itself already exists in the local object store, but the
        # remote may not have it -> upload it
        if e["sha"] not in remote_paths.values():
            content = subprocess.run(["git", "cat-file", "-p", e["sha"]],
                                      capture_output=True).stdout
            if not content.startswith(b"version https://git-lfs.github.com/spec"):
                gh._req("POST", f"/repos/{args.repo}/git/blobs",
                        {"content": base64.b64encode(content).decode(), "encoding": "base64"})
                uploaded += 1
    print(f"blobs uploaded: {uploaded}")

    # ---- build the tree on the remote ----
    tree_entries = []
    for e in changed:
        tree_entries.append({"path": e["path"], "mode": e["mode"],
                             "type": e["type"], "sha": e["sha"]})
    tree = gh._req("POST", f"/repos/{args.repo}/git/trees",
                   {"base_tree": remote_sha, "tree": tree_entries})
    print(f"new tree     : {tree['sha']}")

    msg = args.message or git("log", "-1", "--pretty=%B").strip()
    commit = gh._req("POST", f"/repos/{args.repo}/git/commits",
                     {"message": msg, "tree": tree["sha"], "parents": [remote_sha]})
    print(f"new commit   : {commit['sha']}")

    gh._req("PATCH", f"/repos/{args.repo}/git/refs/heads/{args.branch}",
            {"sha": commit["sha"], "force": False})
    print(f"PUSHED to {args.repo}@{args.branch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
