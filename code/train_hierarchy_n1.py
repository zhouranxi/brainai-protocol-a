#!/usr/bin/env python3
"""ADR-0019 N1: train FF/REC/HFB on CIFAR-10 and run the swap readout.

Per run: 100ep standard protocol; final measurements =
  id test (native inference) + off-inference (T=1) for REC/HFB
  + d_ent/GLUE on ID test features + c100 probe (transfer probe).
Probes reuse the ADR-0016 chain (d_ent_of, glue_dimension,
linear_probe_acc, loaders) via import from train_forcing_p4.
Out: runs/v2_hierarchy/n1_{cfg}_s{seed}/n1_result.json
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

from models_hierarchy import HierarchyNet
from train_forcing_p4 import (DATA, CIFAR10_MEAN, CIFAR10_STD,
                               d_ent_of, glue_dimension, linear_probe_acc,
                               set_seed)

RESULTS_KEYS = ("id_test", "off_test", "params_shared", "params_extra")


def loaders(bs=128, workers=6):
    from torchvision import datasets, transforms
    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(2, 10),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    te_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    tr = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"),
                              transform=tr_tf)
    te = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"),
                              transform=te_tf)
    return (torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True,
                                        num_workers=workers, pin_memory=True,
                                        persistent_workers=True),
            torch.utils.data.DataLoader(te, batch_size=256, shuffle=False,
                                        num_workers=workers, pin_memory=True,
                                        persistent_workers=True))


def probe_loaders(workers=4):
    from torchvision import datasets, transforms
    ev_tf = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    mk = lambda ds: torch.utils.data.DataLoader(ds, batch_size=256,
                                                shuffle=False,
                                                num_workers=workers,
                                                pin_memory=True)
    c100_tr = datasets.ImageFolder(os.path.join(DATA, "cifar100", "train"),
                                   transform=ev_tf)
    c100_te = datasets.ImageFolder(os.path.join(DATA, "cifar100", "test"),
                                   transform=ev_tf)
    id_tr = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"),
                                 transform=ev_tf)
    id_te = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"),
                                 transform=ev_tf)
    return {"id_tr": mk(id_tr), "id_te": mk(id_te),
            "c100_tr": mk(c100_tr), "c100_te": mk(c100_te)}


@torch.no_grad()
def eval_acc(model, loader, device, steps=None):
    model.eval()
    corr = tot = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb, steps=steps)
        corr += (out.argmax(1) == yb).sum().item()
        tot += yb.numel()
    return corr / tot


@torch.no_grad()
def extract_features(model, loader, device, max_n=10000):
    model.eval()
    feats, ys = [], []
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        # penultimate = head 前的 pooled 特征：重跑 forward 到 pooled
        x0 = model.stem(xb)
        z1 = model.s1(x0); z2 = model.s2(z1); z3 = model.s3(z2)
        # 迭代步（与 forward 同逻辑）——为测量一致，用原生 steps
        T = model.steps
        for t in range(T - 1):
            if model.mode == "rec":
                z1 = z1 + torch.sigmoid(model.lat1(z1)) * z1
                z2 = z2 + torch.sigmoid(model.lat2(z2)) * z2
                z3 = z3 + torch.sigmoid(model.lat3(z3)) * z3
            elif model.mode == "hfb":
                f2 = torch.nn.functional.interpolate(
                    model.td2(z3), size=z2.shape[-2:], mode="nearest")
                z2 = z2 + torch.tanh(f2)
                f1 = torch.nn.functional.interpolate(
                    model.td1(z2), size=z1.shape[-2:], mode="nearest")
                z1 = z1 + torch.tanh(f1)
        pooled = torch.nn.functional.adaptive_avg_pool2d(z3, 1).flatten(1)
        feats.append(pooled.cpu().numpy())
        ys.append(yb.numpy())
        n += len(yb)
        if max_n and n >= max_n:
            break
    return np.concatenate(feats), np.concatenate(ys)


def run_one(cfg, seed, device, epochs=150):
    set_seed(seed)
    outdir = Path(f"/mnt/mydisk/xinguan-brainai/runs/v2_hierarchy/n1p_{cfg}_s{seed}")
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "n1_result.json"
    if fp.exists():
        print(f"[skip] {cfg}_s{seed}", flush=True)
        return json.loads(fp.read_text())

    model = HierarchyNet(mode=cfg).to(device)
    shared, extra = model.param_count_module()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=2e-3, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    trl, tel = loaders()

    t0 = time.time()
    hist = []
    best_test, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        corr = tot = 0
        for xb, yb in trl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            corr += (model(xb).argmax(1) == yb).sum().item()
            tot += yb.numel()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            test = eval_acc(model, tel, device)
            hist.append({"epoch": ep + 1, "train_acc": round(corr / tot, 4),
                         "test_acc": round(test, 4)})
            if test > best_test:
                best_test = test
                best_state = {k: v.detach().clone() for k, v in
                              model.state_dict().items()}
            if (ep + 1) % 30 == 0 or ep == epochs - 1:
                print(f"  [{cfg}_s{seed}] ep{ep+1}/{epochs} train={corr/tot:.4f} "
                      f"test={test:.4f} best={best_test:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    # 恢复最佳 checkpoint（协议修复：早停保护）
    model.load_state_dict(best_state)

    # ---- 交换读数 + 表征测量 ----
    id_test = eval_acc(model, tel, device)                    # 原配推理
    off_test = (eval_acc(model, tel, device, steps=1)          # 关反馈（T=1）
                if cfg in ("rec", "hfb") else None)
    pl = probe_loaders()
    Xid_tr, yid_tr = extract_features(model, pl["id_tr"], device)
    Xid_te, yid_te = extract_features(model, pl["id_te"], device)
    Xc_tr, yc_tr = extract_features(model, pl["c100_tr"], device)
    Xc_te, yc_te = extract_features(model, pl["c100_te"], device)
    id_probe = linear_probe_acc(Xid_tr, yid_tr, Xid_te, yid_te, 10, device,
                                seed=seed)
    c100 = linear_probe_acc(Xc_tr, yc_tr, Xc_te, yc_te, 100, device, seed=seed)
    d_ent = d_ent_of(torch.as_tensor(Xid_tr))
    gd = glue_dimension(Xid_tr, yid_tr, n_sub=500, seed=seed)

    row = {"cfg": cfg, "seed": seed, "id_test": round(id_test, 4),
           "off_test": round(off_test, 4) if off_test is not None else None,
           "id_probe": round(id_probe, 4), "c100_probe": round(c100, 4),
           "d_ent": round(float(d_ent), 3),
           "glue_d": round(gd[0], 4) if gd else None,
           "glue_r": round(gd[1], 4) if gd else None,
           "glue_u": round(gd[2], 4) if gd else None,
           "glue_timeout": gd is None,
           "params_shared": shared, "params_extra": extra,
           "hist": hist, "secs": round(time.time() - t0, 1)}
    fp.write_text(json.dumps({"rows": [row]}, indent=1))
    print(f"  => {cfg}_s{seed}: id={id_test:.4f} off={off_test} "
          f"c100={c100:.4f} d={d_ent:.1f} ({row['secs']}s)", flush=True)
    return row


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", nargs="+", default=["ff", "rec", "hfb"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    for cfg in args.cfgs:
        for seed in args.seeds:
            run_one(cfg, seed, dev, epochs=args.epochs)
