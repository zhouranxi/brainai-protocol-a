#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""脑启发 AI · Phase 0 冒烟测试

目标：用最小代价验证「表征流形优雅性可测量」这条技术路线是否成立。

验证三件事：
  1. 能在真实模型上批量抽取隐状态（管线跑得通）
  2. 能算出流形几何指标（维度 / 光滑性 / 判别性）
  3. 指标对「条件退化」（遮挡）有可预期的响应方向（优雅性随歧义度变化）

设计要点（对应方案任务三 + 任务四）：
  - 主任务：面孔/物体识别，施加受控遮挡 → 条件可分辨度下降
  - 几何指标：
      * effective_dim   —— PCA 累计方差 90% 所需维度（表征空间"容量"）
      * local_smoothness—— 流形上近邻点位移的几何平均（曲率代理，越大越曲折）
      * silhouette_gap  —— 不同刺激类别的表征可分性（判别性）
      * condition_drop  —— 施加遮挡后线性可分性的下降幅度（优雅性鲁棒性代理）
  - 关键判据：若 condition_drop 在所有模型上同向为正，说明「优雅性可测且有区分度」；
    若同向为负或不可重复，说明该指标构造失败，需换几何量。

用法：
  python smoke_geometry.py --model resnet50 --batch 64 --out runs/
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
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ---------------------------------------------------------------------------- #
# 几何指标
# ---------------------------------------------------------------------------- #
def effective_dim(x: np.ndarray, var_ratio: float = 0.90) -> float:
    """PCA 累计解释方差达 var_ratio 所需的主成分数（表征有效容量）。

    生物学对应：表征空间的有效维数（embedding dimensionality），
    过大说明表征冗余/纠缠，过小说明塌缩。
    """
    x = x - x.mean(axis=0, keepdims=True)
    # 用随机投影近似协方差，避免大 N 时 full SVD 爆炸
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
    """流形局部光滑度代理：k 近邻对的平均位移（越小越平滑）。

    生物学对应：神经流形的平滑性 —— 平滑流形无坑洼/奇点，利于稳定推理。
    注意：位移受尺度影响，故先做 L2 归一化再做距离计算。
    """
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    n = x.shape[0]
    if n <= k + 1:
        return float("nan")
    # 分块算距离，避免 N^2 内存爆掉
    sq = (x * x).sum(1)
    dists = []
    block = 512
    for i in range(0, n, block):
        blk = x[i:i + block]
        d2 = sq[i:i + block, None] + sq[None, :] - 2.0 * (blk @ x.T)
        d2[np.arange(blk.shape[0]), np.arange(i, i + blk.shape[0])] = np.inf
        part = np.partition(d2, k, axis=1)[:, :k]
        dists.append(np.sqrt(np.maximum(part, 0)))
    dist = np.concatenate(dists, axis=0)
    return float(dist.mean())


def silhouette_gap(x: np.ndarray, labels: np.ndarray) -> float:
    """线性可分性代理：单类可解释方差比（ANOVA 式）。

    对每个主成分，计算组间方差 / 总方差，取前 k 个主成分的平均。
    生物学对应：刺激类别在表征空间中的可分性（如 V2/V4 的类别选择性的几何体现）。
    """
    x = x - x.mean(axis=0, keepdims=True)
    classes = np.unique(labels)
    if len(classes) < 2:
        return float("nan")
    total = (x ** 2).sum()
    if total <= 0:
        return float("nan")
    between = 0.0
    for c in classes:
        xc = x[labels == c]
        if len(xc) == 0:
            continue
        between += len(xc) * ((xc.mean(axis=0)) ** 2).sum()
    return float(between / total)


def linear_probe_acc(x: np.ndarray, labels: np.ndarray, n_test: int = 30) -> float:
    """岭回归线性探针准确率（几何指标的行为学金标准）。

    生物学对应：解码（decoding）—— 从神经活动解码出刺激类别。
    """
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import train_test_split

    if len(np.unique(labels)) < 2:
        return float("nan")
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, labels, test_size=0.3, random_state=0, stratify=labels
    )
    clf = RidgeClassifier(alpha=1.0).fit(x_tr, y_tr)
    return float(clf.score(x_te, y_te))


# ---------------------------------------------------------------------------- #
# 数据与前向
# ---------------------------------------------------------------------------- #
def build_batches(model_name: str, n_per_class: int, device: str, img_size: int = 32):
    """构造受控遮挡条件下的分类任务。

    条件设计（对应"部分可观测 / 遮挡歧义"）：
      cond='clean'  —— 完整图像
      cond='occ30'  —— 遮挡 30%（中等歧义）
      cond='occ70'  —— 遮挡 70%（高歧义）
      cond='noise'  —— 高斯噪声（低结构、随机歧义，对照组）

    遮挡为随机方块（保留边缘信息），比全黑更接近生物视觉的遮挡补全情境。
    """
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(root=os.environ.get("SMOKE_DATASET_ROOT", "./data/smoke"),
                              transform=tf)
    # 只取前若干类别，控制冒烟规模
    classes = sorted(set(ds.targets))
    keep = classes[:8]
    idx_by_class = {}
    for i, t in enumerate(ds.targets):
        if t in keep:
            idx_by_class.setdefault(t, []).append(i)
    xs, ys = [], []
    for ci, c in enumerate(keep):
        for i in idx_by_class[c][:n_per_class]:
            xs.append(ds[i][0])
            ys.append(ci)
    x = torch.stack(xs)
    y = np.array(ys)
    return x, y


def apply_condition(x: torch.Tensor, cond: str, generator) -> torch.Tensor:
    """施加受控退化条件。"""
    if cond == "clean":
        return x
    b, c, h, w = x.shape
    if cond.startswith("occ"):
        ratio = int(cond[3:]) / 100.0
        out = x.clone()
        # 每样本随机切一块方块并置零（模拟遮挡）
        bh = int(h * ratio ** 0.5)
        bw = int(w * ratio ** 0.5)
        for i in range(b):
            top = int(torch.randint(0, max(h - bh, 1), (1,), generator=generator))
            left = int(torch.randint(0, max(w - bw, 1), (1,), generator=generator))
            out[i, :, top:top + bh, left:left + bw] = 0.0
        return out
    if cond == "noise":
        g = torch.Generator(device="cpu").manual_seed(1234)
        noise = torch.randn(x.shape, generator=g).to(x.device)
        return x + noise * 0.5
    raise ValueError(cond)


def _pick_intermediate_hook(model):
    """通用地抓取"倒数第二 stage 的输出"（残差块末），适配 ResNet / ViT / ConvNet。

    做法：遍历 named_modules，取最后一个名为 layer4 / blocks.11 / layers.3 之类的
    顶层 stage 模块；找不到就退化为最后一个 Conv2d/Linear 之前的 block。
    抓到的张量统一做 (B, C, H, W) -> (B, C) 的空间平均池化。
    """
    # 优先用 timm 的 features_info / forward_features 语义
    candidates = []
    for name, mod in model.named_modules():
        n = name.split(".")[-1]
        if n in ("layer4", "layer3", "blocks", "stages", "features", "layer"):
            candidates.append((name, mod))
    # 取层级最深的那个 stage
    if candidates:
        candidates.sort(key=lambda kv: kv[0].count("."))
        target = candidates[-1][1]
    else:
        target = None
    return target


class _Grab:
    """forward hook：抓取目标模块输出。"""

    def __init__(self, target):
        self.target = target
        self.out = None
        self.handle = None

    def __enter__(self):
        def hook(mod, inp, outp):
            self.out = outp
        self.handle = self.target.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        if self.handle:
            self.handle.remove()


def _pool(t: torch.Tensor) -> torch.Tensor:
    """把不同形状的中间张量统一压成 (B, C) 向量。"""
    if t.dim() == 4:          # (B, C, H, W)
        return t.mean(dim=(2, 3))
    if t.dim() == 3:          # (B, N, C) —— ViT token 序列
        return t.mean(dim=1)
    if t.dim() == 2:          # (B, C)
        return t
    return t.reshape(t.shape[0], -1)


@torch.no_grad()
def extract_features(model, x: torch.Tensor, cond: str, layer: str,
                     batch_size: int, device: str, seed: int = 0) -> np.ndarray:
    """抽取指定层的隐状态（取该层输出的空间平均池化后的向量）。"""
    gen = torch.Generator().manual_seed(seed)
    target = _pick_intermediate_hook(model)
    feats = []
    for i in range(0, x.shape[0], batch_size):
        xb = apply_condition(x[i:i + batch_size], cond, gen).to(device)
        if target is None:
            out = model.forward_features(xb)
            pooled = _pool(out)
        else:
            with _Grab(target) as g:
                model(xb)
            if g.out is None:
                out = model.forward_features(xb)
                pooled = _pool(out)
            else:
                pooled = _pool(g.out)
        feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--timm-tag", default=None,
                    help="timm 权重 tag；默认用 pretrained=True")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--n-per-class", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.dataset_root:
        os.environ["SMOKE_DATASET_ROOT"] = args.dataset_root
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 载入模型: {args.model}", flush=True)
    import timm
    t0 = time.time()
    kw = {}
    if args.timm_tag:
        kw["pretrained_cfg_overlay"] = {"tag": args.timm_tag}
    model = timm.create_model(args.model, pretrained=True, **kw).eval().to(args.device)
    print(f"      参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M"
          f"  载入 {time.time()-t0:.1f}s", flush=True)

    print("[2/5] 准备数据", flush=True)
    try:
        x, y = build_batches(args.model, args.n_per_class, args.device, args.img_size)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"      数据准备失败: {e}", flush=True)
        return 2
    print(f"      样本 {x.shape[0]} 张 (N,C,H,W)={tuple(x.shape)}, 类别 {len(np.unique(y))}", flush=True)

    conditions = ["clean", "occ30", "occ70", "noise"]
    results = {}

    for cond in conditions:
        print(f"[3/5] 抽取隐状态: cond={cond}", flush=True)
        t0 = time.time()
        f = extract_features(model, x, cond, "layer", args.batch, args.device, seed=args.seed)
        dt = time.time() - t0

        m = {
            "effective_dim": effective_dim(f),
            "local_smoothness": local_smoothness(f),
            "silhouette_gap": silhouette_gap(f, y),
            "linear_probe_acc": linear_probe_acc(f, y),
            "n": int(f.shape[0]),
            "dim": int(f.shape[1]),
            "extract_sec": round(dt, 2),
        }
        results[cond] = m
        print(f"      dim={m['effective_dim']:.0f} smooth={m['local_smoothness']:.4f} "
              f"gap={m['silhouette_gap']:.4f} probe={m['linear_probe_acc']:.3f} "
              f"({dt:.1f}s)", flush=True)

    # 关键判据：优雅性随歧义度的单调性
    print("[4/5] 优雅性响应方向检查", flush=True)
    clean = results["clean"]["linear_probe_acc"]
    drops = {c: round(clean - results[c]["linear_probe_acc"], 4) for c in conditions if c != "clean"}
    dim_changes = {c: round(results[c]["effective_dim"] - results["clean"]["effective_dim"], 2)
                   for c in conditions if c != "clean"}
    verdict = {
        "accuracy_drops": drops,
        "effective_dim_changes": dim_changes,
        "monotonic_acc_drop_with_occlusion":
            bool(results["occ30"]["linear_probe_acc"] <= clean + 1e-9
                 and results["occ70"]["linear_probe_acc"] <= results["occ30"]["linear_probe_acc"] + 1e-9),
        "dim_shrinks_with_occlusion":
            bool(results["occ70"]["effective_dim"] <= results["clean"]["effective_dim"] + 1e-9),
    }
    for k, v in verdict.items():
        print(f"      {k}: {v}", flush=True)

    payload = {
        "model": args.model,
        "n_samples": int(x.shape[0]),
        "conditions": results,
        "verdict": verdict,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    tag = args.model.replace("/", "_")
    fp = outdir / f"smoke_{tag}.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[5/5] 结果已写入 {fp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
