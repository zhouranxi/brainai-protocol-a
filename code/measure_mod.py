#!/usr/bin/env python3
"""测量 ckpt_mod 中的新架构模型（shufflenet_v2 / regnet）：
几何指标 + CIFAR-100/TinyImageNet probe + 19 种 corruption。

与 measure_independent.py 同协议，差异：
  - 模型按 ckpt 内 arch 字段构建（train_grid.build_model）
  - 特征抽取用通用 avgpool hook（三个架构都有该属性，输出 4D）
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
from geo_markers import compute_all                                    # noqa: E402
from r1_id_ood import (extract_features, linear_probe, eval_acc,       # noqa: E402
                       subsample_per_class, make_loader, DATA)
from train_grid import build_model                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "newarch"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--m-per-class", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="substring filter on ckpt stem")
    args = ap.parse_args()

    device = args.device
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    id_loader = make_loader(os.path.join(DATA, "cifar_full", "test"), args.batch)
    c100_tr = make_loader(os.path.join(DATA, "cifar100", "train"), args.batch)
    c100_te = make_loader(os.path.join(DATA, "cifar100", "test"), args.batch)
    tiny_tr = make_loader(os.path.join(DATA, "tinyimagenet", "train"), args.batch)
    tiny_te = make_loader(os.path.join(DATA, "tinyimagenet", "valid"), args.batch)

    c10c_root = Path(DATA) / "cifar10c" / "test"
    corr_loaders = {}
    if c10c_root.is_dir():
        for corr in sorted(os.listdir(c10c_root)):
            if os.path.isdir(c10c_root / corr):
                corr_loaders[corr] = make_loader(str(c10c_root / corr), args.batch)

    ckpts = sorted(Path(args.ckpt_dir).glob("*.pt"))
    if args.only:
        ckpts = [c for c in ckpts if args.only in c.stem]
    if args.limit:
        ckpts = ckpts[:args.limit]
    print(f"{len(ckpts)} checkpoints", flush=True)

    rows = []
    for i, ck in enumerate(ckpts):
        t0 = time.time()
        sd = torch.load(ck, map_location="cpu")
        arch = sd.get("arch", "resnet18")
        model = build_model(arch).to(device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()

        rec = {"ckpt": ck.stem, "arch": arch,
               "epoch": int(sd.get("ep", -1)),
               "lr": float(sd.get("lr", np.nan)),
               "wd": float(sd.get("wd", np.nan)),
               "seed": int(sd.get("seed", -1))}

        Xid, Lid, yid = extract_features(model, id_loader, device)
        rec["id_acc"] = eval_acc(Lid, yid)
        Xg, yg = subsample_per_class(Xid, yid, args.m_per_class, 10)
        m = compute_all(Xg, yg, 10, args.m_per_class)
        rec.update({f"id_{k}": v for k, v in m.items()})

        Xtr, _, ytr = extract_features(model, c100_tr, device)
        Xte, _, yte = extract_features(model, c100_te, device)
        rec["c100_probe_acc"] = linear_probe(
            Xtr, ytr, Xte, yte, num_classes=100,
            epochs=args.probe_epochs, seed=0)

        Xtr2, _, ytr2 = extract_features(model, tiny_tr, device)
        Xte2, _, yte2 = extract_features(model, tiny_te, device)
        rec["tinyimg_probe_acc"] = linear_probe(
            Xtr2, ytr2, Xte2, yte2, num_classes=200,
            epochs=args.probe_epochs, seed=0)

        for corr, ld in corr_loaders.items():
            _, Lo, yo = extract_features(model, ld, device)
            rec[f"corr_{corr}_acc"] = eval_acc(Lo, yo)

        rows.append(rec)
        print(f"  [{i+1}/{len(ckpts)}] {ck.stem}: id={rec['id_acc']:.4f} "
              f"c100={rec['c100_probe_acc']:.4f} tiny={rec['tinyimg_probe_acc']:.4f} "
              f"PR={rec['id_pr_perclass']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    fp = outdir / "mod_arch_results.json"
    # 断点续存：若已有结果则合并
    if fp.exists():
        try:
            old = json.load(open(fp))
            seen = {r["ckpt"] for r in old["rows"]}
            rows = old["rows"] + [r for r in rows if r["ckpt"] not in seen]
        except Exception:
            pass
    fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"[saved] {fp} ({len(rows)} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
