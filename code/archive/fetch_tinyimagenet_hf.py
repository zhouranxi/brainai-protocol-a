#!/usr/bin/env python3
"""从 hf-mirror 下载 Tiny-ImageNet（zh-plus 版，parquet）并转 ImageFolder。"""
import argparse
import io
import os
import urllib.request

import numpy as np
from PIL import Image

MIR = "https://hf-mirror.com"
FILES = {
    "train": "datasets/zh-plus/tiny-imagenet/resolve/main/data/train-00000-of-00001-1359597a978bc4fa.parquet",
    "valid": "datasets/zh-plus/tiny-imagenet/resolve/main/data/valid-00000-of-00001-70d52db3c749a935.parquet",
}


def http_get(url, out, timeout=1800):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(out, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
        print()


def build(split, df, out_root, size=32, per_class=100):
    import pandas as pd
    out = os.path.join(out_root, split)
    if os.path.isdir(out) and len(os.listdir(out)) >= 200:
        print(f"{split} already built"); return
    os.makedirs(out, exist_ok=True)
    print(f"{split}: cols={list(df.columns)} rows={len(df)}")
    # zh-plus 版列：image (dict with bytes) + label (int)
    lab = df["label"].values
    img_col = "image" if "image" in df.columns else "img"
    rng = np.random.RandomState(0)
    n = 0
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        idx = rng.permutation(idx)[:per_class]
        d = os.path.join(out, f"{c:03d}")
        os.makedirs(d, exist_ok=True)
        for j, i in enumerate(idx):
            b = df[img_col].values[i]
            if isinstance(b, dict):
                b = b.get("bytes")
            Image.open(io.BytesIO(b)).convert("RGB").resize((size, size)).save(
                os.path.join(d, f"{j:04d}.png"))
            n += 1
    print(f"  -> {n} images, {len(os.listdir(out))} classes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/mydisk/xinguan-brainai/data")
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)
    import pandas as pd

    for split, rel in FILES.items():
        pq = os.path.join(args.root, f"tinyimgnet_{split}.parquet")
        if not os.path.exists(pq):
            print(f"downloading {rel}")
            http_get(f"{MIR}/{rel}", pq)
        df = pd.read_parquet(pq)
        build(split, df, os.path.join(args.root, "tinyimagenet"))
    out = os.path.join(args.root, "tinyimagenet")
    for sp in ["train", "valid"]:
        d = os.path.join(out, sp)
        if os.path.isdir(d):
            print(f"FINAL {sp}: {len(os.listdir(d))} classes, "
                  f"{sum(len(os.listdir(os.path.join(d,c))) for c in os.listdir(d))} imgs")


if __name__ == "__main__":
    main()
