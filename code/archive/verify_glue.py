#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用官方仓库的 example_checkpoint.pth 验证 GLUE 实现。

官方 example.ipynb 的参考输出（n_sub=500/class, 10 类, M=500）：
    participation_ratio : 10.2586
    neural_collapse     : 0.2104
    dimension           : 1.3009
    radius              : 0.5240
    utility             : 0.1900
    test accuracy       : 94.87%
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glue_impl import (compute_neural_collapse,
                       compute_participation_ratio_global,
                       compute_participation_ratio_perclass,
                       downsample_manifolds, glue_analysis)

VENDOR = "/mnt/mydisk/xinguan-brainai/vendor/ood-generalization-geometry"
DATA = os.environ.get("BRAINAI_DATA_ROOT", "/mnt/mydisk/xinguan-brainai/data")
REF = {"participation_ratio": 10.2586, "neural_collapse": 0.2104,
       "dimension": 1.3009, "radius": 0.5240, "utility": 0.1900}


def main():
    sys.path.insert(0, VENDOR)
    from models import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(f"{VENDOR}/data/example_checkpoint.pth", map_location="cpu")
    model = build_model("ResNet18", dataset="cifar").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"checkpoint epoch={ckpt.get('epoch')} acc={ckpt.get('acc')}")

    # 官方 notebook 的数据设置：CIFAR10_ROOT 指向含 cifar-10-batches-py 的目录
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # 优先用我们已转换好的 ImageFolder；否则用 torchvision
    id_root = os.path.join(DATA, "cifar_full", "test")
    if os.path.isdir(id_root):
        ds = datasets.ImageFolder(root=id_root, transform=tf)
    else:
        ds = datasets.CIFAR10(root=f"{VENDOR}/data/cifar10", train=False,
                               download=True, transform=tf)
    print(f"test set: {len(ds)} samples")

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)

    feats, labels = [], []
    buf = []
    handle = model.avgpool.register_forward_hook(
        lambda m, i, o: buf.append(torch.flatten(o, 1).detach().cpu()))
    with torch.no_grad():
        for xb, yb in dl:
            model(xb.to(device))
            labels.append(yb.numpy())
    handle.remove()
    # 注意：buf 按 batch 累积，需与 labels 对齐
    X = torch.cat(buf).numpy()
    y = np.concatenate(labels)
    order = np.argsort(y, kind="stable")
    X, y = X[order], y[order]
    print(f"features: {X.shape}")

    n_sub = 500
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(1000, n_sub, replace=False) + c * 1000
                          for c in range(10)])
    Xs = X[idx]
    ys = y[idx]

    print("\n--- 纯 numpy 部分（与官方逐字对照）---")
    pr_global = compute_participation_ratio_global(Xs)
    pr_per = compute_participation_ratio_perclass(Xs, 10, n_sub)
    nc1 = compute_neural_collapse(Xs.T, 10, n_sub)   # 官方约定: (N_feat, N_samples)
    print(f"participation_ratio (global) = {pr_global:.4f}   [官方 {REF['participation_ratio']}]")
    print(f"participation_ratio (perclass) = {pr_per:.4f}")
    print(f"neural_collapse = {nc1:.4f}   [官方 {REF['neural_collapse']}]")

    print("\n--- GLUE 部分（独立实现）---")
    manifolds = [Xs[c * n_sub:(c + 1) * n_sub].T for c in range(10)]
    sub = downsample_manifolds(manifolds, n_points=min(50, n_sub), seed=0)
    print(f"manifolds: {len(sub)} x {sub[0].shape}")
    res = glue_analysis(sub, n_t=200, seed=0)
    for k, v in res.items():
        ref = REF.get(k)
        mark = f"   [官方 {ref}]" if ref else ""
        print(f"{k:12s} = {v:.4f}{mark}")


if __name__ == "__main__":
    main()
