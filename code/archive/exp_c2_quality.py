#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 决定性实验：表征几何指标能否作为模型质量的独立预测量。

核心设计（见 docs/prd.md §C2、docs/adr/0002-几何量须排除信息量混淆.md）：
  - 同一架构、同一数据集、**同一 clean 输入条件**（输入信息量恒定 → 排除信息量混淆）
  - 唯一差异来自**训练配方** → 制造可观的泛化质量梯度
  - 对每个模型测 clean 条件下的几何指标，与 test accuracy 求相关
  - 非平凡判据：几何指标与性能的 Spearman 相关 < 1.0（否则只是性能的代理）

质量梯度的构造（刻意覆盖不同失效机理，避免只在单一维度上变化）：
  augment      : 随机裁剪+翻转（提升泛化的经典手段）
  label_smooth : 标签平滑（改变输出几何）
  label_noise  : 随机翻转标签比例（直接破坏监督信号）
  epochs       : 训练时长（早停 vs 过拟合）
  wd           : 权重衰减（正则强度）

用法：
  python exp_c2_quality.py --recipes recipes.json --epochs 30 --out runs/
  python exp_c2_quality.py --recipes recipes.json --seeds 0 1 2 --out runs/
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DATA_ROOT = os.environ.get("BRAINAI_DATA_ROOT", "/mnt/mydisk/xinguan-brainai/data")
CIFAR_ROOT = os.path.join(DATA_ROOT, "cifar_full")   # 完整 CIFAR-10（训练用）


# ---------------------------------------------------------------------------- #
# 数据
# ---------------------------------------------------------------------------- #
def build_cifar(root: str, train: bool, augment: bool, noise: float, seed: int):
    """构建 CIFAR-10 训练/测试集（从 ImageFolder 目录读，避开 torchvision 的下载依赖）。

    关键：augment / label_noise 只作用于**训练集**，测试集始终是干净完整分布。
    这保证了"输入信息量"在测试侧恒定 —— 排除信息量混淆的前提。
    """
    from torchvision import datasets, transforms

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    if train:
        ops = [transforms.RandomCrop(32, padding=4)] if augment else []
        ops += [transforms.RandomHorizontalFlip()] if augment else []
        ops += [transforms.ToTensor(), normalize]
    else:
        ops = [transforms.ToTensor(), normalize]

    split = 'train' if train else 'test'
    ds = datasets.ImageFolder(root=os.path.join(root, split),
                              transform=transforms.Compose(ops))

    if train and noise > 0:
        # 确定性标签噪声：固定种子选样本翻转到别的类
        rng = np.random.RandomState(seed)
        n = len(ds)
        targets = np.array(ds.targets)
        flip = np.where(rng.rand(n) < noise)[0]
        if len(flip):
            targets[flip] = (targets[flip] + rng.randint(1, 10, size=len(flip))) % 10
        ds.targets = targets.tolist()
        # ImageFolder 的 __getitem__ 读 self.targets[index]，改 targets 即生效
    return ds


def loader(ds, bs, shuffle, workers=4):
    return torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, drop_last=False
    )


# ---------------------------------------------------------------------------- #
# 模型
# ---------------------------------------------------------------------------- #
def make_model(arch: str) -> nn.Module:
    if arch == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(num_classes=10)
    elif arch == "resnet50":
        from torchvision.models import resnet50
        m = resnet50(num_classes=10)
    elif arch == "vgg11":
        from torchvision.models import vgg11
        m = vgg11(num_classes=10)
    else:
        raise ValueError(arch)
    return m


# ---------------------------------------------------------------------------- #
# 几何指标（与 smoke_geometry.py 保持同一套定义）
# ---------------------------------------------------------------------------- #
def effective_dim(x: np.ndarray, var_ratio: float = 0.90) -> float:
    """PCA 累计解释方差达 var_ratio 所需主成分数。"""
    x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    if n > 2000:
        idx = np.random.RandomState(0).choice(n, 2000, replace=False)
        x = x[idx]
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    if s.sum() <= 0:
        return float("nan")
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    return float(np.searchsorted(cum, var_ratio) + 1)


def local_smoothness(x: np.ndarray, k: int = 10) -> float:
    """流形局部光滑度代理：k 近邻平均位移（L2 归一化后）。"""
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    n = x.shape[0]
    if n <= k + 1:
        return float("nan")
    sq = (x * x).sum(1)
    dists = []
    for i in range(0, n, 512):
        blk = x[i:i + 512]
        d2 = sq[i:i + 512, None] + sq[None, :] - 2.0 * (blk @ x.T)
        d2[np.arange(blk.shape[0]), np.arange(i, i + blk.shape[0])] = np.inf
        part = np.partition(d2, k, axis=1)[:, :k]
        dists.append(np.sqrt(np.maximum(part, 0)))
    return float(np.concatenate(dists, axis=0).mean())


def silhouette_gap(x: np.ndarray, labels: np.ndarray) -> float:
    """组间方差 / 总方差。"""
    x = x - x.mean(axis=0, keepdims=True)
    total = (x ** 2).sum()
    if total <= 0:
        return float("nan")
    between = 0.0
    for c in np.unique(labels):
        xc = x[labels == c]
        between += len(xc) * ((xc.mean(axis=0)) ** 2).sum()
    return float(between / total)


def linear_probe_acc(x: np.ndarray, labels: np.ndarray) -> float:
    """岭回归线性探针准确率。"""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import train_test_split
    if len(np.unique(labels)) < 2:
        return float("nan")
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, labels, test_size=0.3, random_state=0, stratify=labels)
    clf = RidgeClassifier(alpha=1.0).fit(x_tr, y_tr)
    return float(clf.score(x_te, y_te))


def criticality_proxy(x: np.ndarray) -> float:
    """临界态代理指标：谱间隙（spectral gap）的归一化倒数。

    依据复杂系统科学的临界态判据：系统接近临界态时，关联长度发散、
    谱隙趋近于 0。这里用表征协方差矩阵的特征值谱隙作为代理：
    gap = (λ1 - λ2) / λ1，gap 越小越接近临界态。
    """
    x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    if n > 1500:
        idx = np.random.RandomState(0).choice(n, 1500, replace=False)
        x = x[idx]
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    if len(s) < 2 or s[0] <= 0:
        return float("nan")
    return float((s[0] - s[1]) / s[0])


# ---------------------------------------------------------------------------- #
# 特征抽取
# ---------------------------------------------------------------------------- #
class _Grab:
    def __init__(self, target):
        self.target, self.out, self.handle = target, None, None

    def __enter__(self):
        def hook(mod, inp, outp):
            self.out = outp
        self.handle = self.target.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        if self.handle:
            self.handle.remove()


def pick_stage(model) -> nn.Module:
    cands = []
    for name, mod in model.named_modules():
        if name.split(".")[-1] in ("layer4", "layer3", "features", "blocks", "avgpool"):
            cands.append((name, mod))
    if not cands:
        return None
    cands.sort(key=lambda kv: kv[0].count("."))
    return cands[-1][1]


def pool(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 4:
        return t.mean(dim=(2, 3))
    if t.dim() == 3:
        return t.mean(dim=1)
    if t.dim() == 2:
        return t
    return t.reshape(t.shape[0], -1)


@torch.no_grad()
def extract(model, ds, device, batch=256, workers=4):
    """在**干净测试集**上抽取特征（无退化、无增强）。"""
    target = pick_stage(model)
    dl = loader(ds, batch, shuffle=False, workers=workers)
    feats, labels = [], []
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        if target is None:
            out = model.forward_features(xb)
            pooled = pool(out)
        else:
            with _Grab(target) as g:
                model(xb)
            pooled = pool(g.out if g.out is not None else model.forward_features(xb))
        feats.append(pooled.float().cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ---------------------------------------------------------------------------- #
# 训练
# ---------------------------------------------------------------------------- #
def train_one(cfg: dict, seed: int, device: str, log_dir: Path, ckpt_dir: Path):
    """按配方训练一个模型，返回 (model, test_acc)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr = build_cifar(CIFAR_ROOT, True, cfg.get("augment", False),
                     cfg.get("label_noise", 0.0), seed)
    te = build_cifar(CIFAR_ROOT, False, False, 0.0, seed)
    trl = loader(tr, cfg.get("batch", 128), shuffle=True)
    tel = loader(te, 256, shuffle=False)

    model = make_model(cfg.get("arch", "resnet18")).to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=cfg.get("lr", 0.1),
        momentum=0.9, weight_decay=cfg.get("wd", 5e-4), nesterov=True)
    epochs = cfg.get("epochs", 30)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ls = cfg.get("label_smooth", 0.0)
    crit = nn.CrossEntropyLoss(label_smoothing=ls)

    tag = f"{cfg['name']}_s{seed}"
    ck = ckpt_dir / f"{tag}.pt"
    hist = []

    for ep in range(epochs):
        model.train()
        tot = corr = 0
        for xb, yb in trl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            corr += (out.argmax(1) == yb).sum().item()
            tot += yb.numel()
        sched.step()

        # 每 epoch 存 checkpoint（断点续跑用，顺序写）
        torch.save({"model": model.state_dict(), "ep": ep + 1}, ck)

        model.eval()
        tc = tt = 0
        with torch.no_grad():
            for xb, yb in tel:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                p = model(xb).argmax(1)
                tc += (p == yb).sum().item()
                tt += yb.numel()
        hist.append({"epoch": ep + 1, "train_acc": corr / tot, "test_acc": tc / tt})
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    [{tag}] ep{ep+1}/{epochs} train={corr/tot:.3f} test={tc/tt:.3f}",
                  flush=True)

    return model, hist[-1]["test_acc"], hist


# ---------------------------------------------------------------------------- #
# 主流程
# ---------------------------------------------------------------------------- #
DEFAULT_RECIPES = [
    # —— 目标：造出跨度足够大的泛化质量梯度，且覆盖不同失效机理 ——
    # 机理维度：增强 / 标签平滑 / 标签噪声 / 训练时长 / 正则强度 / 学习率
    {"name": "best",         "augment": True,  "label_smooth": 0.1, "label_noise": 0.0,  "epochs": 40, "wd": 5e-4, "lr": 0.1},
    {"name": "aug_only",     "augment": True,  "label_smooth": 0.0, "label_noise": 0.0,  "epochs": 40, "wd": 5e-4, "lr": 0.1},
    {"name": "aug_ls",       "augment": True,  "label_smooth": 0.1, "label_noise": 0.0,  "epochs": 15, "wd": 5e-4, "lr": 0.1},
    {"name": "plain",        "augment": False, "label_smooth": 0.0, "label_noise": 0.0,  "epochs": 40, "wd": 5e-4, "lr": 0.1},
    {"name": "plain_ls",     "augment": False, "label_smooth": 0.1, "label_noise": 0.0,  "epochs": 40, "wd": 5e-4, "lr": 0.1},
    {"name": "plain_short",  "augment": False, "label_smooth": 0.0, "label_noise": 0.0,  "epochs": 5,  "wd": 5e-4, "lr": 0.1},
    {"name": "noise20",      "augment": False, "label_smooth": 0.0, "label_noise": 0.20, "epochs": 40, "wd": 5e-4, "lr": 0.1},
    {"name": "noise50",      "augment": False, "label_smooth": 0.0, "label_noise": 0.50, "epochs": 40, "wd": 5e-4, "lr": 0.1},
    {"name": "noise20_aug",  "augment": True,  "label_smooth": 0.0, "label_noise": 0.20, "epochs": 40, "wd": 5e-4, "lr": 0.1},
    {"name": "wd0",          "augment": False, "label_smooth": 0.0, "label_noise": 0.0,  "epochs": 40, "wd": 0.0,  "lr": 0.1},
    {"name": "wd_high",      "augment": False, "label_smooth": 0.0, "label_noise": 0.0,  "epochs": 40, "wd": 5e-3, "lr": 0.1},
    {"name": "lr_high",      "augment": False, "label_smooth": 0.0, "label_noise": 0.0,  "epochs": 40, "wd": 5e-4, "lr": 0.4},
    {"name": "lr_low",       "augment": False, "label_smooth": 0.0, "label_noise": 0.0,  "epochs": 40, "wd": 5e-4, "lr": 0.01},
    {"name": "overfit",      "augment": False, "label_smooth": 0.0, "label_noise": 0.0,  "epochs": 120,"wd": 0.0,  "lr": 0.1},
]


def spearman(a, b):
    from scipy.stats import spearmanr, pearsonr
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    r, p = pearsonr(a[m], b[m])
    rho, _ = spearmanr(a[m], b[m])
    return float(r), float(p), float(rho)


# 平凡指标（trivial）：直接使用测试集标签做测量，与 test_acc 本质同源，
# 相关性高是同义反复而非发现。不得作为 M0 的 Go 依据。
# 非平凡指标（non-trivial）：纯几何量，不使用任何标签信息。
TRIVIAL_METRICS = {"linear_probe_acc", "silhouette_gap"}   # 后者虽为方差比但同样用到标签分组
NONTRIVIAL_METRICS = {"effective_dim", "local_smoothness",
                      "criticality_proxy", "spectral_entropy"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", default=None, help="自定义配方 JSON；默认用内置 10 组")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=None, help="覆盖所有配方的 epochs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--feat-n", type=int, default=2000, help="几何测量用的特征样本数")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    logdir = Path(args.log_dir); logdir.mkdir(parents=True, exist_ok=True)
    ckdir = outdir / "ckpt"; ckdir.mkdir(parents=True, exist_ok=True)

    if args.recipes:
        recipes = json.loads(Path(args.recipes).read_text(encoding="utf-8"))
    else:
        recipes = DEFAULT_RECIPES
    if args.epochs:
        for r in recipes:
            r["epochs"] = args.epochs

    print(f"[1/4] 配方 {len(recipes)} 组 × 种子 {args.seeds}", flush=True)
    print(f"      数据根 {CIFAR_ROOT}", flush=True)

    te = build_cifar(CIFAR_ROOT, False, False, 0.0, 0)
    # 几何测量的样本来源：**训练集的干净（无增强）样本**。
    # 理由：所有模型面对的输入分布完全一致（同一批训练图像、同一预处理），
    # 且不触碰测试集标签 —— 从根本上排除信息量混淆与标签泄漏。
    geo_src = build_cifar(CIFAR_ROOT, True, False, 0.0, 0)
    # 标签相关指标（平凡指标）单独用**测试集**算 —— 它们本就与 test_acc 同源，
    # 保留它们是为了量化"同义反复有多强"，不作为 Go 依据。
    te = build_cifar(CIFAR_ROOT, False, False, 0.0, 0)
    rows = []

    for seed in args.seeds:
        for cfg in recipes:
            t0 = time.time()
            print(f"[2/4] 训练 {cfg['name']} seed={seed}", flush=True)
            model, test_acc, hist = train_one(cfg, seed, args.device, logdir, ckdir)

            # 非平凡几何指标：干净训练样本（恒定输入，不用任何标签）
            feats, _ = extract(model, geo_src, args.device)
            if feats.shape[0] > args.feat_n:
                idx = np.random.RandomState(seed).choice(
                    feats.shape[0], args.feat_n, replace=False)
                feats = feats[idx]

            # 平凡指标：测试集（与 test_acc 同源，仅用于量化同义反复强度）
            f_te, l_te = extract(model, te, args.device)

            m = {
                # --- 非平凡：纯几何，不使用标签 ---
                "effective_dim": effective_dim(feats),
                "local_smoothness": local_smoothness(feats),
                "criticality_proxy": criticality_proxy(feats),
                "spectral_entropy": float(_entropy(feats)),
                # --- 平凡：使用标签，与 test_acc 同源 ---
                "silhouette_gap": silhouette_gap(f_te, l_te),
                "linear_probe_acc": linear_probe_acc(f_te, l_te),
            }
            row = {"name": cfg["name"], "seed": seed, "test_acc": test_acc,
                   "recipe": {k: v for k, v in cfg.items() if k != "name"},
                   **m, "train_sec": round(time.time() - t0, 1),
                   "history": hist}
            rows.append(row)
            print(f"      test_acc={test_acc:.4f} dim={m['effective_dim']:.0f} "
                  f"smooth={m['local_smoothness']:.4f} gap={m['silhouette_gap']:.4f} "
                  f"probe={m['linear_probe_acc']:.4f} crit={m['criticality_proxy']:.4f}",
                  flush=True)

    print("[3/4] 相关性分析", flush=True)
    metrics = ["effective_dim", "local_smoothness", "silhouette_gap",
               "linear_probe_acc", "criticality_proxy", "spectral_entropy"]
    analysis = {}
    for name in metrics:
        r, p, rho = spearman([x[name] for x in rows], [x["test_acc"] for x in rows])
        analysis[name] = {"pearson_r": r, "p_value": p, "spearman_rho": rho}

    # M0 判据（ADR-0003）+ 非平凡判据（ADR-0002）
    # Go 候选必须同时满足：
    #   1. 属于非平凡指标（不使用测试标签）
    #   2. |Pearson r| >= 0.7 且 p < 0.05
    #   3. |Spearman rho| < 1.0（不能只是性能的单调重标度）
    go_candidates = []
    rejected_trivial = []
    for m, s in analysis.items():
        if m in TRIVIAL_METRICS:
            rejected_trivial.append(m)
            continue
        if (not np.isnan(s["pearson_r"]) and abs(s["pearson_r"]) >= 0.7
                and s["p_value"] < 0.05
                and (np.isnan(s["spearman_rho"]) or abs(s["spearman_rho"]) < 1.0)):
            go_candidates.append(m)

    verdict = {
        "n_models": len(rows),
        "per_metric": analysis,
        "go_candidates": go_candidates,
        "trivial_metrics_excluded": sorted(rejected_trivial),
        "nontrivial_metrics": sorted(NONTRIVIAL_METRICS),
        "M0_decision": "GO" if go_candidates else "NO-GO",
        "note": "判据见 docs/adr/0003-M0-止损线.md；非平凡判据见 docs/adr/0002",
    }

    for m, s in sorted(analysis.items(), key=lambda kv: -abs(kv[1]["pearson_r"] or 0)):
        if m in go_candidates:
            flag = "GO"
        elif m in TRIVIAL_METRICS:
            flag = "平凡×"      # 相关性高但同义反复，不算数
        else:
            flag = "  "
        print(f"      {flag:4s} {m:22s} r={s['pearson_r']:+.3f} p={s['p_value']:.4f} "
              f"rho={s['spearman_rho']:+.3f}", flush=True)
    print(f"      => M0: {verdict['M0_decision']}  (非平凡候选: {go_candidates})", flush=True)
    if rejected_trivial:
        print(f"      注：已排除平凡指标 {rejected_trivial}（使用测试标签，与 test_acc 同源）",
              flush=True)

    print("[4/4] 写结果", flush=True)
    fp = outdir / f"c2_results_seeds{'-'.join(map(str, args.seeds))}.json"
    fp.write_text(json.dumps({"rows": rows, "analysis": analysis,
                               "verdict": verdict}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"      {fp}", flush=True)
    return 0


def _entropy(feats: np.ndarray) -> float:
    """谱熵：表征能量分布的熵（越高越分散）。"""
    x = feats - feats.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    p = s ** 2
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p)) if len(p) > 1 else float("nan"))


if __name__ == "__main__":
    raise SystemExit(main())
