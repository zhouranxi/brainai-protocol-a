#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R1 核心：在同一批模型上同时测 ID 与 OOD 的几何标记与性能。

检验的命题（源自 ICLR 2026 论文的空白）：
  **几何标记的预测力是分布偏移专属的** ——
  在同分布（ID）任务上，几何标记无法区分质量相当的模型；
  只有存在分布偏移（OOD）时，它们才提供超越 ID accuracy 的预测信息。

设计：
  同一模型 → 三种评估
    1. ID   : CIFAR-10 test        （同分布）
    2. OOD-C: CIFAR-100 test       （类别级偏移，论文主场景）
    3. OOD-K: CIFAR-10C severity=5 （corruption 偏移，论文对照组：
                                    在此场景 ID accuracy 反而最强）
  几何标记**只从 ID 数据上计算**（这是"预后性"predictive 的定义：
  不看未来就知道未来的失败）。

输出：runs/r1_id_ood_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo_markers import compute_all

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
DATA = os.environ.get("BRAINAI_DATA_ROOT", str(ROOT / "data" / "raw"))
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


# --------------------------------------------------------------------------- #
def build_model():
    """与 Phase 0 训练脚本**完全一致**的 resnet18。

    Phase 0（exp_c2_quality.py）用的是原生 torchvision resnet18(num_classes=10)：
      - conv1 = 7x7/stride2（未改成 CIFAR 常用的 3x3）
      - maxpool 保留
    checkpoint 的 conv1.weight 形状 [64,3,7,7] 已核实。
    ⚠️ 不可改成 CIFAR 风格 stem，否则 state_dict 不匹配。
    """
    from torchvision.models import resnet18
    return resnet18(num_classes=10)


def make_loader(root, batch=256, workers=4, subset=None):
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = datasets.ImageFolder(root=root, transform=tf)
    if subset is not None:
        idx = np.random.RandomState(0).choice(len(ds), subset, replace=False)
        ds = torch.utils.data.Subset(ds, idx.tolist())
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                       num_workers=workers, pin_memory=True)


class _Grab:
    """抓取 penultimate 特征：fc 的输入（展平后），适配任意分类器结构。

    优先 hook 分类头的 forward-pre-input；对 resnet/regnet/shufflenet 均适用。
    """

    def __init__(self, mod):
        self.mod, self.out, self.h = mod, None, None

    def __enter__(self):
        def hook(m, args):
            # forward_pre_hook 签名: hook(module, args)；args 是位置参数 tuple
            self.out = args[0] if isinstance(args, tuple) and len(args) == 1 else args
        self.h = self.mod.register_forward_pre_hook(hook)
        return self

    def __exit__(self, *a):
        if self.h:
            self.h.remove()


def penultimate_hook(model):
    """返回可抓取 fc 输入的 _Grab；无 fc 时退回 avgpool 输出。"""
    if hasattr(model, "fc"):
        return _Grab(model.fc)
    if hasattr(model, "avgpool"):
        return _Grab(model.avgpool)
    raise ValueError(f"no penultimate hook for {type(model).__name__}")


@torch.no_grad()
def extract_features(model, loader, device):
    """返回 (features, logits, labels)，按标签排序。"""
    feats, logits, labels = [], [], []
    with penultimate_hook(model) as g:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            feats.append(torch.flatten(g.out, 1).float().cpu().numpy())
            logits.append(out.float().cpu().numpy())
            labels.append(yb.numpy())
    X = np.concatenate(feats)
    L = np.concatenate(logits)
    y = np.concatenate(labels)
    order = np.argsort(y, kind="stable")
    return X[order], L[order], y[order]


def linear_probe(Xtr, ytr, Xte, yte, num_classes, epochs=40, lr=1e-2, wd=0.0, seed=0):
    """OOD 线性探针（论文协议：冻结特征提取器，在 OOD 数据上训新线性分类头）。

    论文 Section 3.1：'the network's feature extractor was frozen. A new linear
    classifier was then trained on top of these features using the OOD dataset.
    The test accuracy of this linear probe served as our measure of OOD performance.'

    实现用 torch 在 GPU 上做全批量 LBFGS-free 的 logistic regression（SGD + momentum），
    与 sklearn LogisticRegression 相比避免了 50 类的 CPU 瓶颈。
    """
    torch.manual_seed(seed)
    dev = Xtr.device
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    yte_t = torch.as_tensor(yte, dtype=torch.long, device=dev)

    # 标准化
    mu = Xtr_t.mean(0, keepdim=True)
    sd = Xtr_t.std(0, keepdim=True) + 1e-6
    Xtr_t = (Xtr_t - mu) / sd
    Xte_t = (Xte_t - mu) / sd

    clf = torch.nn.Linear(Xtr_t.shape[1], num_classes).to(dev)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = torch.nn.CrossEntropyLoss()

    n = Xtr_t.shape[0]
    bs = min(512, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = crit(clf(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        sched.step()

    with torch.no_grad():
        pred = clf(Xte_t).argmax(1)
        return float((pred == yte_t).float().mean())


def eval_acc(logits, labels):
    return float((logits.argmax(1) == labels).mean())


def subsample_per_class(X, y, M, num_classes, seed=0):
    """每类取 M 个样本，返回按类连续的子集。"""
    rng = np.random.RandomState(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], M, replace=False)
                          for c in range(num_classes)])
    idx = np.sort(idx)
    return X[idx], y[idx]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "resnet18"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--m-per-class", type=int, default=500,
                    help="每类用于几何测量的样本数（论文用 500）")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--probe-epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 数据加载器 ----
    print("[1/4] 准备数据", flush=True)
    loaders = {
        "id": make_loader(os.path.join(DATA, "cifar_full", "test"), args.batch),
    }
    # CIFAR-10C：标签空间不变 → 直接用原分类头评估
    c10c_root = Path(DATA) / "cifar10c" / "test"
    if c10c_root.is_dir():
        merged = Path(DATA) / "cifar10c_merged"
        if not merged.is_dir():
            import shutil
            for corr in sorted(os.listdir(c10c_root)):
                for cls in sorted(os.listdir(c10c_root / corr)):
                    tgt = merged / cls
                    tgt.mkdir(parents=True, exist_ok=True)
                    for f in os.listdir(c10c_root / corr / cls):
                        shutil.copy(c10c_root / corr / cls / f, tgt / f"{corr}_{f}")
        loaders["ood_c10c"] = make_loader(str(merged), args.batch)
    # CIFAR-100：类别不重合 → 需 linear probe（train 训探针，test 评估）
    c100_tr = make_loader(os.path.join(DATA, "cifar100", "train"), args.batch)
    c100_te = make_loader(os.path.join(DATA, "cifar100", "test"), args.batch)
    print(f"      loaders: {list(loaders.keys())} + cifar100(train/test, linear probe)", flush=True)

    # ---- 模型 ----
    ckpts = sorted(Path(args.ckpt_dir).glob("*.pt"))
    if not ckpts:
        print("!! no checkpoints found", flush=True)
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

        rec = {"ckpt": name, "epoch": int(sd.get("ep", -1))}

        # --- ID：抽特征 + 测标记 + 测准确率 ---
        Xid, Lid, yid = extract_features(model, loaders["id"], device)
        rec["id_acc"] = eval_acc(Lid, yid)

        Xg, yg = subsample_per_class(Xid, yid, args.m_per_class, 10)
        m = compute_all(Xg, yg, 10, args.m_per_class)
        rec.update({f"id_{k}": v for k, v in m.items()})

        # --- OOD-C: CIFAR-100 linear probe（冻结特征，在 OOD 上训新头） ---
        Xtr, _, ytr = extract_features(model, c100_tr, device)
        Xte, _, yte = extract_features(model, c100_te, device)
        rec["ood_c100_probe_acc"] = linear_probe(
            Xtr, ytr, Xte, yte, num_classes=100,
            epochs=args.probe_epochs, seed=args.seed)

        # --- OOD-K: CIFAR-10C 标签空间不变 → 直接用原分类头 ---
        if "ood_c10c" in loaders:
            _, Lo, yo = extract_features(model, loaders["ood_c10c"], device)
            rec["ood_c10c_acc"] = eval_acc(Lo, yo)

        rows.append(rec)
        print(f"  [{i+1}/{len(ckpts)}] {name}: id={rec['id_acc']:.4f} "
              f"c100probe={rec['ood_c100_probe_acc']:.4f} "
              f"c10c={rec.get('ood_c10c_acc', float('nan')):.4f} "
              f"PR={rec['id_pr_perclass']:.2f} NC1={rec['id_nc1']:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    print("[3/4] 写结果", flush=True)
    fp = outdir / "r1_id_ood_results.json"
    fp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      {fp}", flush=True)
    print("[4/4] 完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
