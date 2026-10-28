#!/usr/bin/env python3
"""在自建 48 个 resnet18 上测 Tiny-ImageNet linear probe（第二类级偏移 OOD）。

协议与 CIFAR-100 probe 一致：冻结特征，TinyImageNet train 训新头，valid 评估。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_id_ood import build_model, extract_features, linear_probe, DATA  # noqa: E402


def make_loader(root, batch):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return DataLoader(datasets.ImageFolder(root, transform=tf),
                      batch_size=batch, shuffle=False, num_workers=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/mnt/mydisk/xinguan-brainai/runs/ckpt_grid")
    ap.add_argument("--out", default="/mnt/mydisk/xinguan-brainai/runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe-epochs", type=int, default=40)
    args = ap.parse_args()

    tiny = "/mnt/mydisk/xinguan-brainai/data/tinyimagenet"
    tr = make_loader(os.path.join(tiny, "train"), 256)
    te = make_loader(os.path.join(tiny, "valid"), 256)
    print(f"tinyimgnet: {len(tr.dataset)} train / {len(te.dataset)} valid", flush=True)

    ckpts = sorted(Path(args.ckpt_dir).glob("*.pt"))
    rows = []
    for i, ck in enumerate(ckpts):
        t0 = time.time()
        model = build_model().to(args.device)
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()
        Xtr, _, ytr = extract_features(model, tr, args.device)
        Xte, _, yte = extract_features(model, te, args.device)
        acc = linear_probe(Xtr, ytr, Xte, yte, num_classes=200,
                           epochs=args.probe_epochs, seed=0)
        rows.append({"ckpt": ck.stem, "tinyimg_probe_acc": acc})
        print(f"  [{i+1}/{len(ckpts)}] {ck.stem}: {acc:.4f} ({time.time()-t0:.0f}s)",
              flush=True)

    # 合并进 ind_validation_results
    ind_fp = Path(args.out) / "ind_validation_results.json"
    if ind_fp.exists():
        ind = json.load(open(ind_fp))
        by = {r["ckpt"]: r for r in ind["rows"]}
        for r in rows:
            if r["ckpt"] in by:
                by[r["ckpt"]]["tinyimg_probe_acc"] = r["tinyimg_probe_acc"]
        ind_fp.write_text(json.dumps(ind, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(f"[merged into] {ind_fp}")
    fp = Path(args.out) / "tinyimg_probe_results.json"
    fp.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"[saved] {fp}")


if __name__ == "__main__":
    raise SystemExit(main())
