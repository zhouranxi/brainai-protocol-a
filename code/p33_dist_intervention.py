#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P33 (ADR-0033): distribution intervention against the collapse attractor.

P24 showed the objective route (CE+cons) fails (0/3 flips, deeper
contraction). This batch tests the data route with three arms:

  occ  - occluded-only (sev3), reproduces the collapse baseline
  mix  - per-batch 1:1 clean:occluded sample interleaving
  curr - curriculum: epochs 0-29 clean, 30-59 occluded sev3

Frozen gates in paper/adr_0033.md:
  R-D1: mix or curr reaches rho >= 0 on >=2/3 seeds while occ stays < 0
  R-D0: both interventions fail -> causal chain closes

Probes: P23 suite (fit_predictors + sweep_errors), identical rho.
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

from train_a0 import DATA, Occlude, set_seed
from train_forcing_p4 import CIFAR10_MEAN, CIFAR10_STD, build_model

OUT_DIR = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p33")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS_SHORT = 60
CURR_SWITCH = 30
SEVS = (0, 1, 2, 3, 4, 5)


def build_loaders(arm, seed, workers=4):
    """Train loader per arm; test loaders per severity."""
    from torchvision import datasets, transforms
    base_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    te_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    tr_root = os.path.join(DATA, "cifar_full", "train")
    te_root = os.path.join(DATA, "cifar_full", "test")

    if arm == "occ":
        clean_ds = datasets.ImageFolder(tr_root, transform=Occlude(base_tf, 3))
        occ_ds = clean_ds
    else:
        clean_ds = datasets.ImageFolder(tr_root, transform=base_tf)
        occ_ds = datasets.ImageFolder(tr_root, transform=Occlude(base_tf, 3))

    if arm == "mix":
        pair = []
        for i in range(len(clean_ds)):
            c, y = clean_ds[i]
            o, _ = occ_ds[i]
            pair.append((c, o, y))
        tl = torch.utils.data.DataLoader(pair, batch_size=128, shuffle=True,
                                         num_workers=workers, pin_memory=True)
    elif arm == "curr":
        # both loaders over the same indices; switch at CURR_SWITCH by epoch
        tl_clean = torch.utils.data.DataLoader(clean_ds, batch_size=128,
                                               shuffle=False, num_workers=workers,
                                               pin_memory=True)
        tl_occ = torch.utils.data.DataLoader(occ_ds, batch_size=128,
                                             shuffle=False, num_workers=workers,
                                             pin_memory=True)
        tl = ("curr", tl_clean, tl_occ)
    elif arm == "occ":
        tl = torch.utils.data.DataLoader(occ_ds, batch_size=128, shuffle=True,
                                         num_workers=workers, pin_memory=True)
    else:
        raise ValueError(arm)

    tests = {}
    for sev in SEVS:
        ds = datasets.ImageFolder(te_root, transform=Occlude(te_tf, sev))
        tests[sev] = torch.utils.data.DataLoader(ds, batch_size=256,
                                                 shuffle=False,
                                                 num_workers=workers,
                                                 pin_memory=True)
    return tl, tests


def stage_feats(model, x):
    z = model.relu(model.bn1(model.conv1(x)))
    zs = [z]
    for li in (1, 2, 3, 4):
        z = getattr(model, f"layer{li}")(z)
        zs.append(z)
    return zs


def rho_dir(err_mean_by_sev):
    sevs = [1, 2, 3, 4, 5]
    n = len(sevs)
    mx, my = sum(sevs) / n, sum(err_mean_by_sev) / n
    num = sum((s - mx) * (e - my) for s, e in zip(sevs, err_mean_by_sev))
    den = (sum((s - mx) ** 2 for s in sevs)
           * sum((e - my) ** 2 for e in err_mean_by_sev)) ** 0.5
    return num / den if den else 0.0


def train_one(arm, seed, device):
    tag = f"p33_{arm}_s{seed}"
    out = OUT_DIR / f"{tag}.json"
    if out.exists():
        print(f"skip {tag}", flush=True)
        return json.loads(out.read_text())
    set_seed(seed)
    model = build_model("resnet18").to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.125, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_SHORT)
    tl, tests = build_loaders(arm, seed)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    rho_mid = None
    for ep in range(EPOCHS_SHORT):
        model.train()
        hit = n = 0
        if arm == "mix":
            for clean, occ, yb in tl:
                # alternate clean/occluded samples within the batch (1:1)
                half = clean.shape[0] // 2
                x = torch.cat([clean[:half], occ[half:]], 0).to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(x)
                loss = ce(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                hit += (logits.argmax(1) == yb).sum().item()
                n += len(yb)
        elif arm == "curr":
            _, tl_clean, tl_occ = tl
            loader = tl_clean if ep < CURR_SWITCH else tl_occ
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = ce(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                hit += (logits.argmax(1) == yb).sum().item()
                n += len(yb)
        else:
            for xb, yb in tl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = ce(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                hit += (logits.argmax(1) == yb).sum().item()
                n += len(yb)
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"  [{tag}] ep{ep+1} acc {hit/n:.4f}", flush=True)
        if ep + 1 == CURR_SWITCH and arm == "curr":
            rho_mid = quick_rho(model, device)
            print(f"  [{tag}] mid rho {rho_mid:+.3f}", flush=True)

    # ---- probe: per-severity acc + penultimate norm ----
    model.eval()
    acc_s, norm_s = {}, {}
    with torch.no_grad():
        for sev, dl in tests.items():
            corr = tot = 0
            norms = []
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                corr += (model(xb).argmax(1) == yb).sum().item()
                tot += yb.numel()
                norms.append(float(stage_feats(model, xb)[-1].pow(2).mean()))
            acc_s[str(sev)] = round(corr / tot, 4)
            norm_s[str(sev)] = round(float(np.mean(norms)), 4)

    # ---- collapse rho (P23 machinery) ----
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

    row = {"tag": tag, "arm": arm, "seed": seed,
           "secs": round(time.time() - t0),
           "acc": acc_s, "feat_norm": norm_s,
           "err_sweep": err_sweep, "rho_collapse": round(rho, 4)}
    if rho_mid is not None:
        row["rho_mid"] = round(rho_mid, 4)
    out.write_text(json.dumps(row, indent=1))
    print(f"  => {tag}: rho {rho:+.3f} | acc@s0 {acc_s['0']} @s5 {acc_s['5']} "
          f"| norm@s0 {norm_s['0']} @s5 {norm_s['5']}", flush=True)
    return row


def quick_rho(model, device):
    """Mid-training rho estimate (lighter: 25ep predictors, sev 1-5 only)."""
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
    preds = fit_predictors(model, ex, tr_dl, device, epochs=25)
    sweep = sweep_errors(model, ex, preds, device, sevs=(1, 2, 3, 4, 5))
    ex.remove()
    e_mean = [sum(sweep[str(s)]) / 4 for s in (1, 2, 3, 4, 5)]
    return rho_dir(e_mean)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="occ,mix,curr")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    Path("/tmp/p33_done").unlink(missing_ok=True)
    for arm in args.arms.split(","):
        for s in [int(x) for x in args.seeds.split(",")]:
            train_one(arm, s, device)
    Path("/tmp/p33_done").write_text("done")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
