#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PC-A0 (ADR-0029): beta x seed grid on occlusion inputs + frozen probes.

Reuses train_a0's Occlude machinery (train fixed at sev-3, same input
conditions as ADR-0024 for comparability). Arms: beta {0.0, 0.1, 1.0} x
seeds {0,1,2} = 9 runs, 200 ep. Frozen probes recorded per run:
severity-0..5 accuracy sweep (PC-P2) + per-layer prediction errors (PC-P1).
Resume-safe per (beta, seed).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/code")
os.environ.setdefault("BRAINAI_DATA_ROOT", "/mnt/mydisk/xinguan-brainai/data")

import numpy as np
import torch
import torch.nn as nn

from train_a0 import DATA, EPOCHS, Occlude, loaders, set_seed
from train_forcing_p4 import CIFAR10_MEAN, CIFAR10_STD
DEVICE = torch.device("cuda:0")
from models_pc import build_pc

OUT_DIR = Path("/mnt/mydisk/xinguan-brainai/runs/v2_pc_a0")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def train_one(beta: float, seed: int, device):
    tag = f"pc_b{beta}_s{seed}"
    out = OUT_DIR / f"{tag}.json"
    if out.exists():
        print(f"skip {tag} (done)", flush=True)
        return json.loads(out.read_text())
    set_seed(seed)
    model = build_pc(beta=beta, seed=seed).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.125, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    tl, _ = loaders(mode="s3")            # train fixed at sev-3 (as ADR-0024)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        hit = n = 0
        for xb, yb in tl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = ce(logits, yb)
            if beta > 0:
                loss = loss + beta * model.pc_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            hit += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        sched.step()
        if (ep + 1) % 40 == 0:
            print(f"  [{tag}] ep{ep+1} acc {hit/n:.4f}", flush=True)

    # ---- frozen probes: severity sweep with per-layer prediction errors ----
    from torchvision import datasets, transforms
    te_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))])
    model.eval()
    sev_stats = {"acc": {}, "err": {}}
    with torch.no_grad():
        for sev in range(6):
            ds = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"),
                                      transform=Occlude(te_tf, sev))
            dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                             num_workers=4, pin_memory=True)
            corr = tot = 0
            errs = []
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                corr += (logits.argmax(1) == yb).sum().item()
                tot += yb.numel()
                if beta > 0:
                    errs.append(model.layer_errors(xb))
            sev_stats["acc"][str(sev)] = round(corr / tot, 4)
            if beta > 0 and errs:
                per_layer = np.array(errs).mean(0)
                sev_stats["err"][str(sev)] = [round(float(e), 5) for e in per_layer]

    row = {"tag": tag, "beta": beta, "seed": seed,
           "secs": round(time.time() - t0), "severity_stats": sev_stats}
    out.write_text(json.dumps(row, indent=1))
    print(f"  => {tag}: acc@s0 {sev_stats['acc']['0']} @s3 {sev_stats['acc']['3']} "
          f"@s5 {sev_stats['acc']['5']}", flush=True)
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", default="0.0,0.1,1.0")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    betas = [float(b) for b in args.betas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    for beta in betas:
        for s in seeds:
            train_one(beta, s, DEVICE)


if __name__ == "__main__":
    main()
