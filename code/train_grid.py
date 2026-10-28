#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 ICLR 2026 论文的超参网格训练 CIFAR-10 模型（resnet18）。

协议对齐官方 scripts/train.py：
  - 200 epoch，cosine annealing 到 0
  - SGD momentum 0.9
  - 4×4 的 (lr, weight_decay) 网格 × 3 种子
  - 数据增强：RandomCrop(32,pad=4) + RandomHorizontalFlip
论文的 ResNet 网格：
  lr  = [1.0, 0.5, 0.25, 0.125]
  wd  = [2e-4, 1e-4, 5e-5, 2.5e-5]

目的：构造「ID test accuracy 相近但 OOD 表现差异大」的模型群 ——
这是检验几何标记预测力的必要条件（论文 Section 3 的核心设计）。

输出：runs/ckpt_grid/{arch}_lr{lr}_wd{wd}_s{seed}.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

def _find_root():
    """Locate the package root: the nearest ancestor containing results/ and code/.

    Works both when scripts are run from inside the package (code/xxx.py) and when
    they are run from a flat working directory (xxx.py at top level).
    """
    here = Path(__file__).resolve().parent
    for cand in [here, here.parent, here.parent.parent]:
        if (cand / "results").is_dir() and (cand / "code").is_dir():
            return cand
    return here.parent if (here.parent / "results").is_dir() else here
ROOT = _find_root()
import torch
import torch.nn as nn

DATA = os.environ.get("BRAINAI_DATA_ROOT", "/mnt/mydisk/xinguan-brainai/data")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# 论文 Table 5：ResNet(18/34/50) 的 SGD 网格
LR_GRID = [1.0, 0.5, 0.25, 0.125]
WD_GRID = [2e-4, 1e-4, 5e-5, 2.5e-5]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch="resnet18"):
    from torchvision.models import resnet18, resnet34, resnet50
    fn = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}[arch]
    return fn(num_classes=10)


def loaders(bs=128, workers=6):
    from torchvision import datasets, transforms
    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    te_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    tr = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"), transform=tr_tf)
    te = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"), transform=te_tf)
    return (
        torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True, num_workers=workers,
                                   pin_memory=True, persistent_workers=True, drop_last=False),
        torch.utils.data.DataLoader(te, batch_size=256, shuffle=False, num_workers=workers,
                                   pin_memory=True, persistent_workers=True),
    )


def train_one(arch, lr, wd, seed, epochs, trl, tel, device, outdir, ckpt_every=20):
    set_seed(seed)
    model = build_model(arch).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    tag = f"{arch}_lr{lr}_wd{wd}_s{seed}"
    ck = Path(outdir) / f"{tag}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)

    hist = []
    for ep in range(epochs):
        model.train()
        corr = tot = 0
        for xb, yb in trl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            crit(out, yb).backward()
            opt.step()
            corr += (out.argmax(1) == yb).sum().item()
            tot += yb.numel()
        sched.step()

        model.eval()
        tc = tt = 0
        with torch.no_grad():
            for xb, yb in tel:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                p = model(xb).argmax(1)
                tc += (p == yb).sum().item()
                tt += yb.numel()
        hist.append({"epoch": ep + 1, "train_acc": corr / tot, "test_acc": tc / tt})

        # 断点续跑：每 ckpt_every epoch 存一次；已完成则跳过
        if (ep + 1) % ckpt_every == 0 or ep == epochs - 1:
            torch.save({"model": model.state_dict(), "ep": ep + 1,
                        "lr": lr, "wd": wd, "seed": seed, "arch": arch,
                        "history": hist}, ck)
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    [{tag}] ep{ep+1}/{epochs} train={corr/tot:.3f} test={tc/tt:.3f}",
                  flush=True)
    return tag, hist[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--outdir", default=str(ROOT / "checkpoints" / "resnet18"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr-grid", type=float, nargs="+", default=LR_GRID)
    ap.add_argument("--wd-grid", type=float, nargs="+", default=WD_GRID)
    args = ap.parse_args()

    device = args.device
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/3] 准备数据", flush=True)
    trl, tel = loaders(args.batch, args.workers)

    jobs = [(lr, wd, s) for lr in args.lr_grid for wd in args.wd_grid for s in args.seeds]
    print(f"[2/3] {len(jobs)} 组训练任务（{args.epochs} epoch each）", flush=True)

    done = []
    for i, (lr, wd, s) in enumerate(jobs):
        tag = f"{args.arch}_lr{lr}_wd{wd}_s{s}"
        if (outdir / f"{tag}.pt").exists():
            print(f"  [{i+1}/{len(jobs)}] {tag} — 已存在，跳过", flush=True)
            continue
        t0 = time.time()
        print(f"  [{i+1}/{len(jobs)}] {tag}", flush=True)
        tag, last = train_one(args.arch, lr, wd, s, args.epochs, trl, tel,
                              device, outdir)
        print(f"      done test_acc={last['test_acc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        done.append({"tag": tag, **last})

    print("[3/3] 完成", flush=True)
    fp = outdir / "train_summary.json"
    fp.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
