#!/usr/bin/env python3
"""第四格：官方宿主协议 × 自建 48 模型（ADR-0016 附录二十一预注册）。

设计（跑前写下，见附录二十一「对第四格实验的设计决定」）：
  宿主协议 = 官方 geo.py 的 wrapper 链：
    官方模型构造（vendor models/resnet.py）
    → 官方预处理（ToTensor+Normalize，无增广）
    → 官方特征钩子（avgpool flatten）
    → 官方采样（np.random.default_rng(42)，每类 500）
    → 官方 downsample（n_points=min(50, M)）
    → 官方 PR 公式（PCA explained_variance）
    → 官方 NC1 公式（S_B 截断伪逆）
    + 我们的 GLUE 数值核（glue_impl.glue_analysis，n_t=200）

  对照：第三格（glue_own48_results.json，全自实现链）

预注册判据：
  Q1 排序一致：逐指标 Spearman ρ ≥ 0.95
  Q2 符号稳定：radius 对 c100 保持正号（第三格 +0.602）
  Q3 参考锚点：官方 example 上读数 = 本实现 A 变体值

输出：runs/v2_p4_forcing/../glue_cell4_own48.json
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CODE = Path("/mnt/mydisk/xinguan-brainai/code")
VENDOR = Path("/mnt/mydisk/xinguan-brainai/vendor/ood-generalization-geometry")
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(VENDOR / "models"))

from glue_impl import glue_analysis, downsample_manifolds  # noqa: E402
import resnet as official_resnet  # noqa: E402


def official_nc1(features_T, num_classes, M, rcond=1e-3):
    """官方 geo.py 的 neural_collapse_1（逐字复刻）。"""
    X_dict = {c: features_T[:, M * c:M * (c + 1)] for c in range(num_classes)}
    X_centered = {c: X_dict[c] - X_dict[c].mean(axis=1, keepdims=True)
                  for c in range(num_classes)}
    Sigma_dict = {c: X_centered[c] @ X_centered[c].T for c in range(num_classes)}
    Sigma_W = sum(Sigma_dict.values()) / (num_classes * M)
    mu_dict = {c: np.mean(X_dict[c], axis=1) for c in range(num_classes)}
    mu_G = np.mean(features_T, axis=1)
    Sigma_B = sum(np.outer(mu_dict[c] - mu_G, mu_dict[c] - mu_G)
                  for c in range(num_classes)) / num_classes
    eigvals, eigvecs = np.linalg.eigh(Sigma_B)
    lam_max = eigvals.max()
    keep = eigvals > rcond * lam_max
    if not np.any(keep):
        return 0.0
    ev_tr = eigvals[keep]
    evec_tr = eigvecs[:, keep]
    pinv = (evec_tr / ev_tr) @ evec_tr.T
    pinv = 0.5 * (pinv + pinv.T)
    return float(np.trace(Sigma_W @ pinv) / num_classes)


def official_pr(X):
    """官方 compute_participation_ratio（PCA，逐字复刻）。"""
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(X)
    ev = pca.explained_variance_
    return float((np.sum(ev) ** 2) / np.sum(ev ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/mnt/mydisk/xinguan-brainai/runs/ckpt_grid")
    ap.add_argument("--out", default="/mnt/mydisk/xinguan-brainai/runs/glue_cell4_own48.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-sub", type=int, default=500)
    ap.add_argument("--n-points", type=int, default=50)
    ap.add_argument("--n-t", type=int, default=200)
    args = ap.parse_args()

    device = args.device
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

    # 官方数据加载（parquet 版 CIFAR-10 test，官方预处理）
    import pyarrow.parquet as pq
    table = pq.read_table("/mnt/mydisk/xinguan-brainai/data/cifar10_test.parquet").to_pylist()

    # 自建 ckpt 列表（48 个，与第三格同清单）
    ckpts = sorted(Path(args.ckpt_dir).glob("resnet18_*.pt"))
    # 排除 P4 forcing 产物（非 base 模型）：第三格清单是 14 配方×3种子减半再并
    # 实际上第三格 48 = 16 组合×3 种子。核对第三格 JSON 的 ckpt 名单。
    cell3 = json.load(open("/mnt/mydisk/xinguan-brainai/runs/glue_own48_results.json"))
    cell3_names = {r["ckpt"] for r in cell3["rows"]}
    ckpts = [c for c in ckpts if c.stem in cell3_names]
    print(f"第四格模型清单：{len(ckpts)} 个（与第三格对齐）", flush=True)

    rows = []
    t0 = time.time()
    for i, ck in enumerate(ckpts):
        # 官方模型构造（自建 ckpt 是 torchvision resnet18 结构？
        # ——第三格用的 r1_id_ood.build_model。这里必须与 ckpt 结构匹配。
        # 自建 ckpt 是按 torchvision resnet18 训的，官方 models/resnet.py
        # 的结构不同（state_dict 键不兼容）。宿主协议的可换部分只有
        # 数据/特征/采样/PR/NC1 —— 模型构造必须跟 ckpt 走。
        sys.path.insert(0, str(CODE))
        from r1_id_ood import build_model
        m = build_model()                 # torchvision resnet18(num_classes=10)
        ckpt = torch.load(ck, map_location="cpu")
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        m.load_state_dict(sd, strict=True)
        m.eval().to(device)

        # 官方特征钩子：avgpool flatten
        feat_buf = []
        handle = m.avgpool.register_forward_hook(
            lambda mo, inp, out: feat_buf.append(torch.flatten(out, 1).detach().cpu()))
        ys = []
        with torch.no_grad():
            for b in range(0, len(table), 512):
                batch = []
                for r in table[b:min(b + 512, len(table))]:
                    im = Image.open(io.BytesIO(r["img"]["bytes"])).convert("RGB")
                    a = np.asarray(im, dtype=np.float32) / 255.
                    batch.append(((a - mean) / std).transpose(2, 0, 1))
                x = torch.from_numpy(np.stack(batch)).to(device)
                m(x)
                ys.extend([r["label"] for r in table[b:min(b + 512, len(table))]])
        handle.remove()
        X_all = torch.cat(feat_buf).numpy()
        y = np.array(ys)

        # 官方采样协议：class-ordered + rng(42) 每类 500
        order = np.argsort(y, kind="stable")
        X_cls = X_all[order]
        counts = np.bincount(y)
        assert counts.min() >= args.n_sub, "每类样本不足"
        rng = np.random.default_rng(42)
        sub_idx = np.concatenate([
            rng.choice(counts[c], args.n_sub, replace=False) + c * counts[c]
            for c in range(10)])
        X_sub = X_cls[sub_idx]            # (5000, 512) class-ordered

        # 官方 downsample + 官方 PR/NC1 + 我们的 GLUE 数值核
        manifolds = [X_sub[c * args.n_sub:(c + 1) * args.n_sub].T
                     for c in range(10)]
        sub50 = downsample_manifolds(manifolds, n_points=args.n_points, seed=0)
        g = glue_analysis(sub50, n_t=args.n_t, seed=0)
        pr = official_pr(X_sub)
        nc1 = official_nc1(X_sub.T, 10, args.n_sub)

        row = {
            "ckpt": ck.stem,
            "official_pr": pr,
            "official_nc1": nc1,
            "glue_dimension": g["dimension"],
            "glue_radius": g["radius"],
            "glue_utility": g["utility"],
            "secs": round(time.time() - t0, 1),
        }
        # 挂第三格的 id/c100（同模型同种子，量不变）
        by = {r["ckpt"]: r for r in cell3["rows"]}
        if ck.stem in by:
            row["id_acc"] = by[ck.stem]["id_acc"]
            row["c100_probe_acc"] = by[ck.stem]["c100_probe_acc"]
        rows.append(row)
        print(f"  [{i+1}/{len(ckpts)}] {ck.stem}: "
              f"dim={g['dimension']:.4f} rad={g['radius']:.4f} "
              f"util={g['utility']:.4f} pr={pr:.3f} nc1={nc1:.4f}", flush=True)
        json.dump({"rows": rows}, open(args.out, "w"), indent=1)

    print(f"done: {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
