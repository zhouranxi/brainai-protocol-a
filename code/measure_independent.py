#!/usr/bin/env python3
"""独立验证测量：在自建 48 组模型上测公开几何指标 + ID/OOD 性能。

对应 ADR-0011 第一层。不依赖 GLUE 私有代码。
输出 JSON 供 analyze_independent.py 分析。
"""
import argparse
import json
import os
import sys
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo_markers import compute_all          # noqa: E402
from r1_id_ood import (build_model, extract_features, linear_probe,  # noqa: E402
                       eval_acc, subsample_per_class, make_loader, DATA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "resnet18"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--m-per-class", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--probe-epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/4] 准备数据", flush=True)
    id_loader = make_loader(os.path.join(DATA, "cifar_full", "test"), args.batch)
    c100_tr = make_loader(os.path.join(DATA, "cifar100", "train"), args.batch)
    c100_te = make_loader(os.path.join(DATA, "cifar100", "test"), args.batch)

    # CIFAR-10C：按 corruption 分别评估（官方 T5 是按场景细分的）
    c10c_root = Path(DATA) / "cifar10c" / "test"
    corr_loaders = {}
    if c10c_root.is_dir():
        for corr in sorted(os.listdir(c10c_root)):
            if corr.startswith("_") or not os.path.isdir(c10c_root / corr):
                continue
            corr_loaders[corr] = make_loader(str(c10c_root / corr), args.batch)
    print(f"      corruption 场景: {len(corr_loaders)}", flush=True)

    ckpts = sorted(Path(args.ckpt_dir).glob("*.pt"))
    if args.limit:
        ckpts = ckpts[:args.limit]
    if not ckpts:
        print("!! no checkpoints", flush=True)
        return 2
    print(f"[2/4] {len(ckpts)} 个 checkpoint", flush=True)

    rows = []
    for i, ck in enumerate(ckpts):
        t0 = time.time()
        name = ck.stem
        model = build_model().to(device)
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()

        rec = {
            "ckpt": name,
            "epoch": int(sd.get("ep", -1)),
            "lr": float(sd.get("lr", np.nan)),
            "wd": float(sd.get("wd", np.nan)),
            "seed": int(sd.get("seed", -1)),
            "arch": sd.get("arch", "resnet18"),
        }

        # ID：特征 + 准确率 + 几何标记
        Xid, Lid, yid = extract_features(model, id_loader, device)
        rec["id_acc"] = eval_acc(Lid, yid)
        Xg, yg = subsample_per_class(Xid, yid, args.m_per_class, 10)
        m = compute_all(Xg, yg, 10, args.m_per_class)
        rec.update({f"id_{k}": v for k, v in m.items()})

        # OOD-C：cifar100 linear probe（官方协议：冻结特征，OOD 上训新头）
        Xtr, _, ytr = extract_features(model, c100_tr, device)
        Xte, _, yte = extract_features(model, c100_te, device)
        rec["c100_probe_acc"] = linear_probe(
            Xtr, ytr, Xte, yte, num_classes=100,
            epochs=args.probe_epochs, seed=args.seed)

        # OOD-K：CIFAR-10C（标签空间不变 → 原分类头），按 corruption 分场景
        for corr, ld in corr_loaders.items():
            _, Lo, yo = extract_features(model, ld, device)
            rec[f"corr_{corr}_acc"] = eval_acc(Lo, yo)

        rows.append(rec)
        print(f"  [{i+1}/{len(ckpts)}] {name}: id={rec['id_acc']:.4f} "
              f"c100probe={rec['c100_probe_acc']:.4f} "
              f"PR={rec['id_pr_perclass']:.2f} NC1={rec['id_nc1']:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    print("[3/4] 写结果", flush=True)
    fp = outdir / "ind_validation_results.json"
    fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      {fp}", flush=True)
    print("[4/4] 完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
