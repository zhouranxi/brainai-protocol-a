#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P23 (memo path C): collapse-probe marker batch on the 68-ckpt corpus.

For each locked checkpoint: attach lateral predictors, fit them briefly
(50 epochs, head-only) on clean train images so they read out the trained
network's layer-to-layer predictability, then measure prediction errors
across occlusion severities 0..5 on the test set. The collapse direction
rho = corr(severity, mean layer error) is the marker: negative = the
network's features collapse under hard inputs (the default found in
PC-A0), near-zero/positive = collapse-resistant.

Resume-safe; ~1.5 min per checkpoint (fit 50ep head + one eval sweep).
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

from train_a0 import DATA, Occlude, set_seed
from train_forcing_p4 import build_model
from p19_smoothness import arch_from_name

RUNS = Path("/mnt/mydisk/xinguan-brainai/runs")
OUT = RUNS / "v2_p23" / "p23_collapse_marker.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0")

# channels at stage boundaries, per family (probed from torchvision)
FAMILY = {
    "resnet18": [64, 64, 128, 256, 512],
    "resnet34": [64, 64, 128, 256, 512],
    "resnet50": [64, 256, 512, 1024, 2048],
    "regnet_y_400mf": [32, 32, 64, 160, 384],
    "shufflenet_v2_x1_0": [24, 24, 116, 232, 464],
}


def _stage_sequence(trunk):
    """Architecture-agnostic ordered list of stage modules, including stem.
    Returns (modules, out_channels) for the boundaries stem->s1..s4."""
    kids = dict(trunk.named_children())
    if "layer1" in kids:                      # resnet family
        mods = [trunk.conv1]
        chans = [64]
        # resnet50 conv1 out is 64, layer1 out 256 etc.
        return mods, chans
    if "stem" in kids:                        # regnet
        return [kids["stem"]], [32]
    if "conv1" in kids and "stage2" in kids:  # shufflenet
        return [kids["conv1"]], [24]
    raise ValueError(f"unknown trunk {type(trunk).__name__}")


def stage_feats(trunk, x):
    """Return [z_stem, z_s1, z_s2, z_s3, z_s4] for any grid architecture.
    Uses forward hooks on the stage-output modules (architecture-agnostic)."""
    raise NotImplementedError  # replaced by the hook-based extractor below


class StageExtractor:
    """Hook-based stage output capture. Attaches to the modules that produce
    the five boundary features of each architecture family, records their
    outputs during a normal forward pass, and exposes per-stage channels."""

    def __init__(self, model):
        import torch.nn as nn
        self.feats = {}
        self.hooks = []
        kids = dict(model.named_children())
        if "layer1" in kids:  # resnet
            anchors = [model.conv1, model.layer1, model.layer2,
                       model.layer3, model.layer4]
        elif "stem" in kids:  # regnet
            to = model.trunk_output
            blocks = list(to.block1) + list(to.block2) + list(to.block3)
            anchors = [model.stem] + [blocks[-1] for blocks in
                                      (to.block1, to.block2, to.block3)]
        elif "stage2" in kids:  # shufflenet
            anchors = [model.conv1, model.stage2, model.stage3, model.stage4,
                       model.conv5]
        else:
            raise ValueError(f"unknown architecture {type(model).__name__}")
        self.chans = []
        for a in anchors:
            h = a.register_forward_hook(self._mk_hook(len(self.hooks)))
            self.hooks.append(h)
        # probe channels with a dummy pass
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, 32, 32, device=next(model.parameters()).device)
            model(dummy)
        self.n_stages = len(self.feats)

    def _mk_hook(self, idx):
        def hook(_m, _i, out):
            self.feats[idx] = out
        return hook

    def channels(self):
        return [self.feats[i].shape[1] for i in range(self.n_stages)]

    def remove(self):
        for h in self.hooks:
            h.remove()


def fit_predictors(model, extractor, tr_dl, device, epochs=50, lr=1e-3):
    """Train-only lateral heads on frozen trunk features (clean images).
    Predict stage i+1 (avg-pooled to stage i's resolution) from stage i."""
    for p in model.parameters():
        p.requires_grad_(False)
    chans = extractor.channels()
    n_pairs = len(chans) - 1
    preds = nn.ModuleList([
        nn.Conv2d(chans[i], chans[i + 1], 1).to(device) for i in range(n_pairs)
    ])
    opt = torch.optim.Adam(preds.parameters(), lr=lr)
    model.eval()
    for ep in range(epochs):
        tot = 0.0
        for xb, _ in tr_dl:
            xb = xb.to(device, non_blocking=True)
            extractor.feats.clear()
            with torch.no_grad():
                model(xb)
                zs = [extractor.feats[i] for i in range(len(chans))]
            loss = 0.0
            for i, pr in enumerate(preds):
                tgt = zs[i + 1]
                h, w = zs[i].shape[-2:]
                if tgt.shape[-2:] != (h, w):
                    tgt = F.adaptive_avg_pool2d(tgt, (h, w))
                loss = loss + F.mse_loss(pr(zs[i]), tgt)
            loss = loss / n_pairs
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
        if (ep + 1) % 25 == 0:
            print(f"    head ep{ep+1} loss {tot/len(tr_dl):.4f}", flush=True)
    return preds


def sweep_errors(model, extractor, preds, device, sevs=(0, 1, 2, 3, 4, 5)):
    """Mean per-stage error at each severity (frozen heads, frozen trunk)."""
    from torchvision import datasets, transforms
    te_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))])
    out = {}
    chans = extractor.channels()
    n_pairs = len(chans) - 1
    model.eval()
    with torch.no_grad():
        for sev in sevs:
            ds = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"),
                                      transform=Occlude(te_tf, sev))
            dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                             num_workers=4, pin_memory=True)
            errs = []
            for xb, _ in dl:
                xb = xb.to(device, non_blocking=True)
                extractor.feats.clear()
                model(xb)
                zs = [extractor.feats[i] for i in range(len(chans))]
                le = []
                for i, pr in enumerate(preds):
                    tgt = zs[i + 1]
                    h, w = zs[i].shape[-2:]
                    if tgt.shape[-2:] != (h, w):
                        tgt = F.adaptive_avg_pool2d(tgt, (h, w))
                    le.append(float(F.mse_loss(pr(zs[i]), tgt)))
                errs.append(le)
            per_stage = np.array(errs).mean(0)
            out[str(sev)] = [round(float(e), 5) for e in per_stage]
    return out


def main():
    grid = sorted(RUNS.glob("ckpt_grid/*.pt"))
    print(f"grid: {len(grid)} ckpts", flush=True)
    rows = []
    done = set()
    if OUT.exists():
        rows = json.loads(OUT.read_text())["rows"]
        done = {r["tag"] for r in rows}
    # clean train loader (head fitting; subset for speed)
    from torchvision import datasets, transforms
    tr_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))])
    tr_ds = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"),
                                 transform=tr_tf)
    sub = torch.utils.data.Subset(tr_ds, range(0, 50000, 5))  # 10k imgs
    tr_dl = torch.utils.data.DataLoader(sub, batch_size=256, shuffle=True,
                                        num_workers=4, pin_memory=True)

    for j, ck in enumerate(grid):
        tag = ck.stem
        if tag in done:
            print(f"skip {tag}", flush=True)
            continue
        t0 = time.time()
        set_seed(123)
        model = build_model(arch_from_name(tag)).to(DEVICE)
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()
        extractor = StageExtractor(model)
        preds = fit_predictors(model, extractor, tr_dl, DEVICE)
        err_sweep = sweep_errors(model, extractor, preds, DEVICE)
        extractor.remove()
        # collapse direction rho over sev 1..5 (sev0 excluded: clean baseline)
        sevs = [1, 2, 3, 4, 5]
        e_mean = [sum(err_sweep[str(s)]) / 4 for s in sevs]
        def _rho(x, y):
            n = len(x)
            mx, my = sum(x) / n, sum(y) / n
            num = sum((a - mx) * (b - my) for a, b in zip(x, y))
            den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
            return num / den if den else 0.0
        rho = _rho(sevs, e_mean)
        rows.append({"tag": tag, "rho_collapse": round(rho, 4),
                     "err_mean_sev": {str(s): round(e, 5) for s, e in zip(sevs, e_mean)},
                     "secs": round(time.time() - t0)})
        OUT.write_text(json.dumps({"rows": rows}, indent=1))
        print(f"[{j+1}/{len(grid)}] {tag} rho={rho:+.3f} "
              f"({rows[-1]['secs']}s)", flush=True)
        del model, preds
        torch.cuda.empty_cache()
    print(f"written: {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
