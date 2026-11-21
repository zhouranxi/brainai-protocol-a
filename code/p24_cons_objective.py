#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P24 (memo path B1): does an anti-collapse objective flip the direction?

Arms: CE (baseline, reproduces PC-A0's collapse) vs CE + consistency
(feature alignment between clean and occluded views of the same image).
Grid: objective {ce, ce+cons} x seeds {0,1,2} = 6 runs, 60 epochs (short:
we only need the error-direction flip, not SOTA accuracy). Probes: the
same severity-0..5 per-stage error sweep as P23/PC-A0.

Gate (frozen in memo): rho flips from negative to >=0 on >=2/3 seeds in
the cons arm while the ce arm stays negative -> the collapse is
objective-driven, and a PC-A0 re-run under the cons objective is
triggered (the sixth attempt, now mechanistically motivated).
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
import torch.nn.functional as F

from train_a0 import DATA, EPOCHS, Occlude, set_seed
from train_forcing_p4 import CIFAR10_MEAN, CIFAR10_STD, build_model

OUT_DIR = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p24")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS_SHORT = 60
SEVS = (0, 1, 2, 3, 4, 5)


def build_pair_loaders(seed, workers=4):
    """Train loader yielding (clean, occluded-sev3, y); test per severity."""
    from torchvision import datasets, transforms
    base_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    clean = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"),
                                 transform=base_tf)
    occ = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"),
                               transform=Occlude(base_tf, 3))
    pair = [(c, o, y) for (c, _), (o, y) in zip(clean, occ)]
    tl = torch.utils.data.DataLoader(pair, batch_size=128, shuffle=True,
                                     num_workers=workers, pin_memory=True)
    te_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    tests = {}
    for sev in SEVS:
        ds = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"),
                                  transform=Occlude(te_tf, sev))
        tests[sev] = torch.utils.data.DataLoader(ds, batch_size=256,
                                                 shuffle=False,
                                                 num_workers=workers,
                                                 pin_memory=True)
    return tl, tests


def stage_feats(trunk, x):
    z = trunk.relu(trunk.bn1(trunk.conv1(x)))
    zs = [z]
    for li in (1, 2, 3, 4):
        z = getattr(trunk, f"layer{li}")(z)
        zs.append(z)
    return zs


def cons_loss(trunk, x_clean, x_occ):
    """Feature consistency: align penultimate features between views."""
    zc = stage_feats(trunk, x_clean)[-1]
    zo = stage_feats(trunk, x_occ)[-1]
    return F.mse_loss(zo, zc.detach())


def rho_dir(err_mean_by_sev):
    sevs = [1, 2, 3, 4, 5]
    n = len(sevs)
    mx, my = sum(sevs) / n, sum(err_mean_by_sev) / n
    num = sum((s - mx) * (e - my) for s, e in zip(sevs, err_mean_by_sev))
    den = (sum((s - mx) ** 2 for s in sevs)
           * sum((e - my) ** 2 for e in err_mean_by_sev)) ** 0.5
    return num / den if den else 0.0


def train_one(obj, seed, device):
    tag = f"p24_{obj}_s{seed}"
    out = OUT_DIR / f"{tag}.json"
    if out.exists():
        print(f"skip {tag}", flush=True)
        return json.loads(out.read_text())
    set_seed(seed)
    model = build_model("resnet18").to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.125, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_SHORT)
    tl, tests = build_pair_loaders(seed)
    ce = nn.CrossEntropyLoss()
    lam_cons = 1.0
    t0 = time.time()
    for ep in range(EPOCHS_SHORT):
        model.train()
        hit = n = 0
        for clean, occ, yb in tl:
            clean = clean.to(device, non_blocking=True)
            occ = occ.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(occ)
            loss = ce(logits, yb)
            if obj == "cons":
                loss = loss + lam_cons * cons_loss(model, clean, occ)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            hit += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"  [{tag}] ep{ep+1} acc {hit/n:.4f}", flush=True)

    # ---- probe: per-severity penultimate feature norm + plain acc ----
    model.eval()
    acc_s, norm_s = {}, {}
    with torch.no_grad():
        for sev, dl in tests.items():
            corr = tot = 0
            norms = []
            for clean, occ, yb in dl:
                occ = occ.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                corr += (model(occ).argmax(1) == yb).sum().item()
                tot += yb.numel()
                norms.append(float(stage_feats(model, occ)[-1].pow(2).mean()))
            acc_s[str(sev)] = round(corr / tot, 4)
            norm_s[str(sev)] = round(float(np.mean(norms)), 4)

    # ---- collapse rho via a quick 2-layer predictor fit (P23-style) ----
    # reuse P23's machinery on this freshly trained model
    from p23_collapse_marker import StageExtractor, fit_predictors, sweep_errors
    ex = StageExtractor(model)
    from torchvision import datasets, transforms
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    tr_ds = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"),
                                 transform=base_tf)
    sub = torch.utils.data.Subset(tr_ds, range(0, 50000, 5))
    tr_dl = torch.utils.data.DataLoader(sub, batch_size=256, shuffle=True,
                                        num_workers=4, pin_memory=True)
    preds = fit_predictors(model, ex, tr_dl, device, epochs=50)
    err_sweep = sweep_errors(model, ex, preds, device, sevs=SEVS)
    ex.remove()
    e_mean = [sum(err_sweep[str(s)]) / 4 for s in (1, 2, 3, 4, 5)]
    rho = rho_dir(e_mean)

    row = {"tag": tag, "obj": obj, "seed": seed,
           "secs": round(time.time() - t0),
           "acc": acc_s, "feat_norm": norm_s,
           "err_sweep": err_sweep, "rho_collapse": round(rho, 4)}
    out.write_text(json.dumps(row, indent=1))
    print(f"  => {tag}: rho {rho:+.3f} | acc@s0 {acc_s['0']} @s5 {acc_s['5']} "
          f"| norm@s0 {norm_s['0']} @s5 {norm_s['5']}", flush=True)
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--objs", default="ce,cons")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    for obj in args.objs.split(","):
        for s in [int(x) for x in args.seeds.split(",")]:
            train_one(obj, s, torch.device("cuda:0"))


if __name__ == "__main__":
    main()
