#!/usr/bin/env python3
"""P19 Phase 0 (ADR-0025): smoothness battery on locked checkpoints.

Zero training: extract penultimate features for 68 ckpt_grid baselines
plus the P8/P10 whiten-forcing dose axis, compute three frozen
smoothness metrics, correlate with the in-library three-caliber reads.
Frozen adjudication F-a/F-b/F-c in ADR-0025.
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

from train_forcing_p4 import (CIFAR10_MEAN, CIFAR10_STD, DATA,
                              PenultimateHook, build_model, set_seed)

RUNS = Path("/mnt/mydisk/xinguan-brainai/runs")
OUT = RUNS / "v2_p19" / "p19_phase0.json"
DEVICE = torch.device("cuda:0")
K = 10          # kNN 参数（冻结）
FEAT_N = 4000   # 每模型特征采样数（冻结，控制内存）


def extract_features(model, split="test"):
    """Penultimate features + labels on cifar_full test (or train)."""
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = datasets.ImageFolder(os.path.join(DATA, "cifar_full", split),
                              transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                     num_workers=4, pin_memory=True)
    from train_forcing_p4 import extract_features
    Fs, Ys = extract_features(model, dl, DEVICE)
    F = np.asarray(Fs, dtype=np.float32)
    Y = np.asarray(Ys)
    # 固定子采样（seed=123）
    rng = np.random.default_rng(123)
    idx = rng.permutation(len(F))[:FEAT_N]
    return F[idx], Y[idx]


def smoothness_metrics(F, Y):
    """Three frozen metrics on features (N, D)."""
    from sklearn.neighbors import NearestNeighbors
    N = len(F)
    # ---- S-loc: local PCA spectral entropy over kNN neighborhoods ----
    nn = NearestNeighbors(n_neighbors=K + 1).fit(F)
    _, idx = nn.kneighbors(F)
    H_loc = []
    for i in range(0, N, 4):          # 每 4 个样本一个邻域（冻结，提速）
        nb = F[idx[i, 1:]]             # (K, D)
        nb = nb - nb.mean(0, keepdims=True)
        sv = np.linalg.svd(nb, compute_uv=False)
        sv = np.maximum(sv, 1e-12)
        p = sv / sv.sum()
        H_loc.append(-(p * np.log(p)).sum())
    H_loc = float(np.median(H_loc))
    # ---- S-curv: triplet midpoint deviation over kNN graph ----
    devs = []
    rng = np.random.default_rng(7)
    for i in range(0, N, 4):
        nb = idx[i, 1:]
        a, b = nb[rng.permutation(K)[:2]]
        m = (F[a] + F[b]) / 2
        # 到最近样本的距离（不含自身）
        d, j = nn.kneighbors(m.reshape(1, -1), n_neighbors=1)
        edge = np.linalg.norm(F[a] - F[b])
        if edge > 1e-9:
            devs.append(float(d[0, 0]) / edge)
    S_curv = float(np.median(devs))
    # ---- S-conf: class margin ratio ----
    classes = np.unique(Y)
    cents = np.stack([F[Y == c].mean(0) for c in classes])
    # 类内 kNN 距离中位
    intra = []
    for c in classes[:8]:             # 冻结：前 8 类（提速）
        Fc = F[Y == c]
        if len(Fc) > K + 1:
            nnc = NearestNeighbors(n_neighbors=K + 1).fit(Fc)
            dc, _ = nnc.kneighbors(Fc)
            intra.append(np.median(dc[:, 1:]))
    intra = float(np.median(intra))
    inter_d = []
    for i in range(len(classes) - 1):
        for j in range(i + 1, min(i + 4, len(classes))):
            inter_d.append(np.linalg.norm(cents[i] - cents[j]))
    inter = float(np.median(inter_d))
    S_conf = inter / max(intra, 1e-12)
    return {"S_loc": round(H_loc, 4), "S_curv": round(S_curv, 6),
            "S_conf": round(S_conf, 4)}


def placebo(F, Y, n=8):
    """F-c health gate: metric spread under label permutation."""
    rng = np.random.default_rng(42)
    out = {"S_loc": [], "S_curv": [], "S_conf": []}
    for _ in range(n):
        Yp = Y[rng.permutation(len(Y))]
        m = smoothness_metrics(F, Yp)
        for k in out:
            out[k].append(m[k])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}


def arch_from_name(name: str) -> str:
    n = name.lower()
    if "regnet" in n:
        return "regnet_y_400mf"
    if "shufflenet" in n:
        return "shufflenet_v2_x1_0"
    if "resnet50" in n:
        return "resnet50"
    if "resnet34" in n:
        return "resnet34"
    return "resnet18"


def load_model(ck):
    set_seed(123)
    model = build_model(arch_from_name(ck.name)).to(DEVICE)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    return model


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # 1) 68 ckpt_grid 基线
    grid = sorted(RUNS.glob("ckpt_grid/*.pt"))
    print(f"grid: {len(grid)} ckpts", flush=True)
    for k, ck in enumerate(grid):
        tag = ck.stem
        model = load_model(ck)
        F, Y = extract_features(model)
        m = smoothness_metrics(F, Y)
        rows.append({"tag": tag, "family": "grid", **m})
        del model
        torch.cuda.empty_cache()
        if (k + 1) % 10 == 0:
            print(f"[grid {k+1}/{len(grid)}] {tag} {m}", flush=True)
    # 2) P8/P10 剂量轴 forcing
    arms = [("p8_s0_seed2", "lam{}"), ("p10_s0_seed3", "lam{}")]
    for prefix, _ in arms:
        for lam in ("0.03", "0.1", "1.0", "3.0"):
            d = RUNS / "v2_p4_forcing" / f"{prefix}_lam{lam}"
            cks = sorted(d.glob("*whiten*.pt"))
            if not cks:
                continue
            model = load_model(cks[-1])
            F, Y = extract_features(model)
            m = smoothness_metrics(F, Y)
            rows.append({"tag": f"{prefix}_{lam}", "family": "dose", **m})
            del model
            torch.cuda.empty_cache()
            print(f"[dose] {prefix}_{lam} {m}", flush=True)
    # 3) F-c placebo（对最后一个模型做一次，零分布参考）
    F, Y = extract_features(load_model(grid[0]))
    pl = placebo(F, Y)
    result = {"rows": rows, "placebo": pl, "k": K, "feat_n": FEAT_N}
    OUT.write_text(json.dumps(result, indent=1))
    print(f"written: {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
