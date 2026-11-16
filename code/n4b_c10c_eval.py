#!/usr/bin/env python3
"""N4b: CIFAR-10-C corruption evaluation (J3 frozen).

Load rec/ff checkpoints (seeds 6-8, n4_*_fd0.0 dirs), evaluate
mean accuracy over the 19 corruption types x 5 severities.
J3: REC-FF mean diff >= +1.5pp and >=13/19 corruptions same
sign -> effect generalizes to real OOD.
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
from torchvision import datasets, transforms

from models_hierarchy import HierarchyNet
from train_forcing_p4 import CIFAR10_MEAN, CIFAR10_STD, set_seed

RUNS = Path("/mnt/mydisk/xinguan-brainai/runs/v2_hierarchy")
OUT = Path("/mnt/mydisk/xinguan-brainai/runs/v2_hierarchy/n4b_c10c.json")


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    corr = tot = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        corr += (model(xb).argmax(1) == yb).sum().item()
        tot += yb.numel()
    return corr / tot




def _eval_np(model, imgs, ys, tf, device, bs=512):
    """imgs (N,32,32,3) uint8 -> accuracy over all severities."""
    import torch as T
    model.eval()
    corr = tot = 0
    with T.no_grad():
        for i in range(0, len(imgs), bs):
            batch = imgs[i:i + bs]
            x = T.stack([tf(im) for im in batch]).to(device)
            yb = ys[i:i + bs].to(device)
            corr += (model(xb := x).argmax(1) == yb).sum().item()
            tot += len(yb)
    return corr / tot


def main() -> int:
    device = torch.device("cuda:0")
    root = os.path.join(os.environ["BRAINAI_DATA_ROOT"], "CIFAR-10-C")
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    # 官方格式：<corruption>.npy，(50000, 32, 32, 3)：5 强度 × 10000
    labels_np = np.load(os.path.join(root, "labels.npy"))
    corruptions = sorted(p.stem for p in Path(root).glob("*.npy")
                         if p.stem != "labels")
    print(f"{len(corruptions)} corruptions", flush=True)
    rows = []
    for cfg in ("rec", "ff"):
        for s in (6, 7, 8):
            ck = RUNS / f"n4_{cfg}_fd0.0_s{s}" / f"{cfg}_last.pt"
            set_seed(123)
            model = HierarchyNet(mode=cfg).to(device)
            model.load_state_dict(torch.load(ck, map_location="cpu",
                                             weights_only=False))
            accs = {}
            for c in corruptions:
                imgs = np.load(os.path.join(root, f"{c}.npy"))
                ys = torch.as_tensor(labels_np[:len(imgs)]).long()
                accs[c] = round(_eval_np(model, imgs, ys, tf, device), 4)
            mean_acc = round(float(np.mean(list(accs.values()))), 4)
            rows.append({"cfg": cfg, "seed": s, "c10c_mean": mean_acc,
                         "per_corruption": accs})
            print(f"{cfg}_s{s}: c10c mean {mean_acc:.4f}", flush=True)
    OUT.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
