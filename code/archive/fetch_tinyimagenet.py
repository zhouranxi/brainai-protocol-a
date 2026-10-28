#!/usr/bin/env python3
"""下载 Tiny-ImageNet（200 类，第二类级偏移）并转为 ImageFolder。

来源：cs.stanford.edu（官方，237MB）；备选 HF 镜像。
输出：/mnt/mydisk/xinguan-brainai/data/tinyimagenet/{train,test}/<wnid>/*.JPEG
resize 到 32×32 以匹配现有管线的归一化与输入尺寸。
"""
import argparse
import io
import os
import tarfile
import urllib.request

import numpy as np
from PIL import Image

MIRROR_PRIMARY = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
# 备选：HF datasets 的 zh-plus/tiny-imagenet parquet


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


def build_folders(root_src, root_out, size=32):
    """Tiny-ImageNet 的 val 有官方标注；train 有 500 img/类。都转 ImageFolder。"""
    for split, src in [("train", os.path.join(root_src, "train")),
                       ("test", os.path.join(root_src, "val"))]:
        out = os.path.join(root_out, split)
        if os.path.isdir(out) and len(os.listdir(out)) >= 200:
            print(f"{split} already built")
            continue
        os.makedirs(out, exist_ok=True)
        n = 0
        if split == "train":
            for wnid in sorted(os.listdir(src)):
                d = os.path.join(src, wnid, "images")
                if not os.path.isdir(d):
                    continue
                tgt = os.path.join(out, wnid)
                os.makedirs(tgt, exist_ok=True)
                for j, f in enumerate(sorted(os.listdir(d))[:100]):  # 每类取 100 张
                    img = Image.open(os.path.join(d, f)).convert("RGB").resize((size, size))
                    img.save(os.path.join(tgt, f"{j:04d}.png"))
                    n += 1
        else:
            # val/val_annotations.txt: fname wnid ...
            ann = {}
            with open(os.path.join(root_src, "val", "val_annotations.txt")) as f:
                for line in f:
                    p = line.split("\t")
                    if len(p) >= 2:
                        ann[p[0]] = p[1]
            for fname, wnid in ann.items():
                tgt = os.path.join(out, wnid)
                os.makedirs(tgt, exist_ok=True)
                img = Image.open(os.path.join(root_src, "val", "images", fname)).convert("RGB").resize((size, size))
                img.save(os.path.join(tgt, fname.replace(".JPEG", ".png")))
                n += 1
        print(f"  {split}: {n} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/mydisk/xinguan-brainai/data")
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)

    zip_path = os.path.join(args.root, "tiny-imagenet-200.zip")
    src = os.path.join(args.root, "tiny-imagenet-200")
    out = os.path.join(args.root, "tinyimagenet")

    if not os.path.isdir(src):
        if not os.path.exists(zip_path):
            print(f"downloading {MIRROR_PRIMARY}")
            http_get(MIRROR_PRIMARY, zip_path)
        print("extracting...")
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(args.root)
    build_folders(src, out)
    # 清理 zip 省空间
    if os.path.exists(zip_path) and os.path.isdir(src):
        os.remove(zip_path)
        print("zip removed")
    for sp in ["train", "test"]:
        d = os.path.join(out, sp)
        print(f"{sp}: {len(os.listdir(d))} classes, "
              f"{sum(len(os.listdir(os.path.join(d,c))) for c in os.listdir(d))} imgs")


if __name__ == "__main__":
    main()
