"""分段续传下载 GLUE 论文 PDF（blob 直连单次会被截断）。"""
import os
import sys
import time
import urllib.request

URL = ("https://rcastoragev2.blob.core.windows.net/4ea8e75eb990647f05b8cdbfba58c9d3/"
       "nihpp-2024.02.26.582157v2.PMC11996410.pdf")
DEST = "/mnt/mydisk/xinguan-brainai/code/glue_paper.pdf"
CHUNK = 1_000_000

req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "research"})
with urllib.request.urlopen(req, timeout=40) as r:
    total = int(r.headers["Content-Length"])
print(f"total = {total}", flush=True)

n = (total + CHUNK - 1) // CHUNK
parts = []
for i in range(n):
    start = i * CHUNK
    end = min(start + CHUNK, total) - 1
    p = f"/tmp/gp_{i:03d}.part"
    for attempt in range(4):
        try:
            rq = urllib.request.Request(
                URL, headers={"User-Agent": "research", "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(rq, timeout=60) as r:
                data = r.read()
            if len(data) == end - start + 1:
                with open(p, "wb") as f:
                    f.write(data)
                parts.append(p)
                print(f"  part {i+1}/{n}: {len(data)} bytes", flush=True)
                break
        except Exception as e:
            print(f"  retry {i}.{attempt}: {e}", flush=True)
            time.sleep(3)
    else:
        print(f"  !! part {i} failed")
        sys.exit(1)

with open(DEST, "wb") as out:
    for p in parts:
        with open(p, "rb") as f:
            out.write(f.read())
        os.remove(p)

print("saved", DEST, os.path.getsize(DEST), flush=True)
from pypdf import PdfReader
r = PdfReader(DEST)
print("PAGES OK:", len(r.pages), flush=True)
