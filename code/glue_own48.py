#!/usr/bin/env python3
"""在自建 48 个 resnet18 模型上用自实现 GLUE 计算 D_eff / R_eff / Ψ_eff。

用途（ADR-0013 路线 A 的正确定位）：
  - **相对比较与符号检验**——同一实现在同一批模型上的相对排序有意义
  - 不做绝对值对比（自实现与官方存在系统性偏差，无第二 checkpoint 可自验）

流程：加载 ckpt → 抽 ID 特征（penultimate，500/class）→ downsample 50 →
glue_impl.glue_analysis → 记录三指标 → 存 JSON。
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
from glue_impl import glue_analysis, downsample_manifolds  # noqa: E402
from r1_id_ood import build_model, extract_features, subsample_per_class, DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "resnet18"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--m-per-class", type=int, default=500)
    ap.add_argument("--n-points", type=int, default=50)
    ap.add_argument("--n-t", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"), transform=tf)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)

    ckpts = sorted(Path(args.ckpt_dir).glob("*.pt"))
    if args.limit:
        ckpts = ckpts[:args.limit]
    print(f"{len(ckpts)} checkpoints", flush=True)

    rows = []
    for i, ck in enumerate(ckpts):
        t0 = time.time()
        name = ck.stem
        model = build_model().to(device)
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()

        X, L, y = extract_features(model, loader, device)
        Xg, yg = subsample_per_class(X, y, args.m_per_class, 10)
        mans = [Xg[c * args.m_per_class:(c + 1) * args.m_per_class].T for c in range(10)]
        sub = downsample_manifolds(mans, n_points=args.n_points, seed=args.seed)
        g = glue_analysis(sub, n_t=args.n_t, seed=args.seed)

        rows.append({
            "ckpt": name,
            "lr": float(sd.get("lr", np.nan)),
            "wd": float(sd.get("wd", np.nan)),
            "seed": int(sd.get("seed", -1)),
            "glue_dimension": g["dimension"],
            "glue_radius": g["radius"],
            "glue_utility": g["utility"],
        })
        print(f"  [{i+1}/{len(ckpts)}] {name}: dim={g['dimension']:.4f} "
              f"rad={g['radius']:.4f} util={g['utility']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    fp = outdir / "glue_own48_results.json"
    # 合并已有的 ind_validation_results（含 id_acc / c100_probe_acc）
    ind_fp = outdir / "ind_validation_results.json"
    if ind_fp.exists():
        ind = json.load(open(ind_fp))
        by_ckpt = {r["ckpt"]: r for r in ind["rows"]}
        for r in rows:
            o = by_ckpt.get(r["ckpt"])
            if o:
                r["id_acc"] = o["id_acc"]
                r["c100_probe_acc"] = o["c100_probe_acc"]
    fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"[saved] {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
