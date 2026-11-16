#!/usr/bin/env python3
"""P14 (ADR-0021): corruption-caliper audit of E4 forcing gains.

Load the P8 s0 seed2 dose-axis checkpoints (lam .03/.1/1/3) plus
the s0 base (ckpt_grid) and run CIFAR-10-C (19x5). Report
delta_c10c vs the archived delta_c100 per dose.
T1 frozen in ADR-0021.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("BRAINAI_DATA_ROOT", "/mnt/mydisk/xinguan-brainai/data")

import numpy as np
import torch
from torchvision import transforms

from train_forcing_p4 import (CIFAR10_MEAN, CIFAR10_STD, DATA,
                              PenultimateHook, build_model,
                              extract_features, linear_probe_acc,
                              set_seed)

RUNS = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p4_forcing")
OUT = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p4_forcing/p14_c10c_audit.json")
LAMS = ("0.03", "0.1", "1.0", "3.0")
MODEL = "resnet18_lr0.125_wd0.0001_s0"


def _eval_np(model, imgs, ys, tf, device, bs=512):
    model.eval()
    corr = tot = 0
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            batch = imgs[i:i + bs]
            x = torch.stack([tf(im) for im in batch]).to(device)
            yb = ys[i:i + bs].to(device)
            corr += (model(x).argmax(1) == yb).sum().item()
            tot += len(yb)
    return corr / tot


def main() -> int:
    device = torch.device("cuda:0")
    root = os.path.join(DATA, "CIFAR-10-C")
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    labels_np = np.load(os.path.join(root, "labels.npy"))
    corruptions = sorted(p.stem for p in Path(root).glob("*.npy")
                         if p.stem != "labels")

    targets = {"base": Path("/mnt/mydisk/xinguan-brainai/runs/ckpt_grid") / f"{MODEL}.pt"}
    for lam in LAMS:
        targets[f"lam{lam}"] = (RUNS / f"p8_s0_seed2_lam{lam}" /
                                f"{MODEL}__whiten_lam{lam}.pt")

    rows = []
    for name, ck in targets.items():
        set_seed(123)
        model = build_model("resnet18").to(device)
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        accs = {}
        for c in corruptions:
            imgs = np.load(os.path.join(root, f"{c}.npy"))
            ys = torch.as_tensor(labels_np[:len(imgs)]).long()
            accs[c] = round(_eval_np(model, imgs, ys, tf, device), 4)
        c10c = round(float(np.mean(list(accs.values()))), 4)
        rows.append({"point": name, "c10c_mean": c10c,
                     "per_corruption": accs})
        print(f"{name}: c10c mean {c10c:.4f}", flush=True)

    OUT.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
