#!/usr/bin/env python3
"""A0 training + probes (ADR-0024; criteria frozen verbatim from the
2026-11-10 evaluation Table 3-1).

9 runs: {ff, rec, bfm} x seeds {0,1,2}; CIFAR-10, 200 epochs,
0.3 GPU·h per run with 1.5 feedback factor. Probes on the trained
BFM runs (P1/P2/P3) + spectrum diagnostic on b (pre-B gate check).
Ambiguity axis: structured 8x8-block occlusion severity 1-5
(reusing the P16 mask machinery: 25%->100% invisible in steps).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("BRAINAI_DATA_ROOT", "/mnt/mydisk/xinguan-brainai/data")

import numpy as np
import torch
import torch.nn as nn

from models_bfm import BFM, param_report
from train_forcing_p4 import (CIFAR10_MEAN, CIFAR10_STD, DATA,
                              set_seed)

RUNS = Path("/mnt/mydisk/xinguan-brainai/runs/v2_a0")
OUT = RUNS / "a0_results.json"
EPOCHS = 200
SEEDS = (0, 1, 2)


def block_mask(sev: int, rng) -> np.ndarray:
    """8x8-block occlusion; severity 1..5 -> 20/35/50/65/80% hidden."""
    fracs = (0.20, 0.35, 0.50, 0.65, 0.80)
    mask = np.ones((32, 32), np.float32)
    n_blocks = int(round(16 * fracs[sev - 1]))
    idx = rng.permutation(16)[:n_blocks]
    for k in idx:
        r, c = divmod(int(k), 4)
        mask[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = 0.0
    return mask


class Occlude:
    def __init__(self, base, sev):
        self.base, self.sev = base, sev

    def __call__(self, img):
        x = self.base(img)
        if self.sev == 0:
            return x
        rng = np.random.default_rng(1000 + self.sev)
        m = block_mask(self.sev, rng)
        return x * torch.as_tensor(m).unsqueeze(0)


def loaders(mode, bs=128, workers=6):
    from torchvision import datasets, transforms
    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    te_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    tr = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"),
                              transform=Occlude(tr_tf, 3))  # 训练期固定 sev3 歧义
    te = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"),
                              transform=Occlude(te_tf, 3))
    return (torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True,
                                        num_workers=workers, pin_memory=True,
                                        persistent_workers=True),
            torch.utils.data.DataLoader(te, batch_size=256, shuffle=False,
                                        num_workers=workers, pin_memory=True,
                                        persistent_workers=True))


@torch.no_grad()
def eval_sev(model, device, workers=4):
    """Accuracy + b/gate stats across severities 0..5 (test set)."""
    from torchvision import datasets, transforms
    te_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    out = {"acc": {}, "b_var": {}, "gate_mean": {}}
    for sev in range(6):
        ds = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"),
                                  transform=Occlude(te_tf, sev))
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                         num_workers=workers, pin_memory=True)
        model.eval()
        corr = tot = 0
        bs_, gs_ = [], []
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            rec = {}
            logits = model(xb, record=rec) if model.mode == "bfm" else model(xb)
            corr += (logits.argmax(1) == yb).sum().item()
            tot += yb.numel()
            if model.mode == "bfm":
                bs_.append(rec["b"].cpu())
                gs_.append(rec["gates"].cpu())
        out["acc"][sev] = round(corr / tot, 4)
        if model.mode == "bfm":
            ball = torch.cat(bs_)
            out["b_var"][sev] = round(ball.var(0).mean().item(), 5)
            out["gate_mean"][sev] = torch.cat(gs_).mean(0).round(decimals=3).tolist()
    return out


def train_one(mode, seed, device):
    tag = f"a0_{mode}_s{seed}"
    outdir = RUNS / tag
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "result.json"
    if fp.exists():
        print(f"[skip] {tag}", flush=True)
        return json.loads(fp.read_text())
    set_seed(seed)
    model = BFM(mode=mode).to(device)
    report = param_report(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    trl, tel = loaders(mode)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in trl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        if (ep + 1) % 40 == 0:
            model.eval()
            corr = tot = 0
            with torch.no_grad():
                for xb, yb in tel:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    corr += (model(xb).argmax(1) == yb).sum().item()
                    tot += yb.numel()
            print(f"  [{tag}] ep{ep+1} test={corr/tot:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), outdir / "last.pt")
    sev_stats = eval_sev(model, device)
    row = {"tag": tag, "mode": mode, "seed": seed, **report,
           "severity_stats": sev_stats, "secs": round(time.time() - t0, 1)}
    fp.write_text(json.dumps(row, indent=1))
    a = sev_stats["acc"]
    k0, k5 = (0, 5) if 0 in a else ("0", "5")
    print(f"  => {tag}: acc@sev0 {a[k0]} @sev5 {a[k5]} "
          f"| params {report['total_params']}", flush=True)
    return row


if __name__ == "__main__":
    dev = torch.device("cuda:0")
    results = [train_one(m, s, dev)
               for m in ("bfm", "ff", "rec") for s in SEEDS]
    OUT.write_text(json.dumps({"rows": results}, indent=1))
    print(f"written: {OUT}", flush=True)
