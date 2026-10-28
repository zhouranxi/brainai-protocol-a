#!/usr/bin/env python3
"""查 hf-mirror 上 tiny-imagenet 数据集的文件树与速度。"""
import json
import urllib.request

MIR = "https://hf-mirror.com"
CANDS = ["zh-plus/tiny-imagenet", "tanganke/tiny-imagenet",
         "cifar100", "uoft-cs/cifar100"]

for repo in CANDS:
    for kind in ["datasets"]:
        url = f"{MIR}/api/{kind}/{repo}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            files = [s["rfilename"] for s in d.get("siblings", [])]
            print(f"=== {repo} ({len(files)} files) ===")
            for f in files[:12]:
                print("   ", f)
        except Exception as e:
            print(f"=== {repo} ERR: {str(e)[:80]}")
