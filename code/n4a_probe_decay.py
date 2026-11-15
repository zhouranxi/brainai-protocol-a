#!/usr/bin/env python3
"""N4a P-arm: post-training feature-noise robustness decay (frozen).

For REC and FF checkpoints (N1pp/N2b pool), inject Gaussian noise
into penultimate features (sigma in {0, .1, .3}, unit-variance
normalized per-feature) and measure the c100 probe accuracy decay.
J1: REC decay at sigma=0.3 slower than FF by >= 2pp -> robustness
channel.
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

from models_hierarchy import HierarchyNet
from train_hierarchy_n1 import (extract_features, probe_loaders,
                                linear_probe_acc)
from train_forcing_p4 import set_seed

RUNS = Path("/mnt/mydisk/xinguan-brainai/runs/v2_hierarchy")
OUT = Path("/mnt/mydisk/xinguan-brainai/runs/v2_hierarchy/n4a_probe_decay.json")
SIGMAS = (0.0, 0.1, 0.3)


def main() -> int:
    rows = []
    for cfg in ("rec", "ff"):
        for s in (0, 1, 2):
            ck = RUNS / f"n1p_{cfg}_s{s}" / f"{cfg}_last.pt"
            if not ck.exists():
                # 训练脚本没存 last ckpt——重建：用记录里的架构 + 重新评估不可行。
                # fallback：直接查有没有已存权重（N4 发射脚本会先补存）。
                print(f"[missing] {ck}")
                continue
            set_seed(123)
            model = HierarchyNet(mode=cfg).cuda()
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state_dict(sd)
            model.eval()
            pl = probe_loaders()
            Xc_tr, yc_tr = extract_features(model, pl["c100_tr"],
                                            torch.device("cuda"))
            Xc_te, yc_te = extract_features(model, pl["c100_te"],
                                            torch.device("cuda"))
            scale = torch.as_tensor(Xc_tr).std(0).clamp_min(1e-6).numpy()
            for sig in SIGMAS:
                g = np.random.default_rng(123 + int(sig * 100))
                Xn_tr = Xc_tr + sig * scale * g.standard_normal(Xc_tr.shape)
                Xn_te = Xc_te + sig * scale * g.standard_normal(Xc_te.shape)
                acc = linear_probe_acc(Xn_tr, yc_tr, Xn_te, yc_te, 100,
                                       torch.device("cuda"), seed=123)
                rows.append({"cfg": cfg, "seed": s, "sigma": sig,
                             "c100_noisy": round(acc, 4)})
                print(f"{cfg}_s{s} sigma={sig}: {acc:.4f}", flush=True)
    OUT.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
