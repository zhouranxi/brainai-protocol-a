#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 pilot：训练期几何 forcing（预注册附录十六，2026-11-06）。

两篇论文的证据链最后一块：形成**中**直接 forcing 几何，让网络把 D_eff 训到
目标值。「geometry is a label, not a knob」的终审判决。

算子（全部零噪声、确定性，penultimate 特征层，loss = CE + λ·L）：
  whiten      L = Σ_k ||C_k − I·tr(C_k)/d||_F² / d    类内谱拉平 ⇒ D↑
  disperse    L = Σp·log p（尖谱大平谱小，最小化 ⇒ 谱熵增 ⇒ D↑）
  concentrate L = −Σp·log p = −H（尖谱≈0 最小，最小化 ⇒ 谱集中 D↓，阴性对照）
  placebo     L = λ·||W_rand||²（固定随机方向正则）      阴性对照，Δc100 应≈0

其中 C_k 为 batch 内类 k 特征协方差，λ_k 为 pooled 协方差特征值（power
iteration 近似，避免训练中全谱分解）。D 监控用 d_ent 口径（谱熵指数），
端点判定另跑 GLUE dimension（看门狗 900s）。

Pilot 条件（附录十六）：2 配方 × {whiten, disperse, concentrate, placebo}
× λ=0.1 × 1 种子，200 epoch 全程 forcing：
  lr1.0_wd5e-05_s0    GLUE D=0.89 / d_ent=9.35  c100=0.120（网格最低点，forcing 主角）
  lr0.125_wd0.0001_s0 GLUE D=1.79                 c100=0.273（网格最高点，对照）
base 对照臂直接读 runs/ckpt_grid/{tag}.pt 同种子模型。

⚠️ D 双口径：GLUE dimension（E1 锚点 0.89/1.79）与 pooled 谱熵 d_ent（9.35，
  E5d 同构造）不是同一个量。P4-1 操纵检查作用在 **d_ent 口径**（BASE_READ
  里存的也是 d_ent 口径），端点判定以 GLUE dimension 为准。

Pilot GO 判据：操纵检查过（d_ent 移动 ≥0.3）且 ID acc ≥ base − 2pp。

用法（服务器 GPU0）：
  CUDA_VISIBLE_DEVICES=0 ~/venvs/dl/bin/python code/train_forcing_p4.py \
    --arm pilot --out runs/v2_p4_forcing/pilot
断点续跑：按 (model, kind) 跳过已完成条件（输出 JSON 增量写）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glue_impl import glue_analysis, downsample_manifolds  # noqa: E402

DATA = os.environ.get("BRAINAI_DATA_ROOT", "/mnt/mydisk/xinguan-brainai/data")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

PILOT_MODELS = ["resnet18_lr1.0_wd5e-05_s0", "resnet18_lr0.125_wd0.0001_s0"]
PILOT_KINDS = ["whiten", "disperse", "concentrate", "placebo"]
PILOT_LAM = 0.1

# E1 base 读数（runs/v2_e1_full.json，审计锚点）
BASE_READ = {
    "resnet18_lr1.0_wd5e-05_s0": {"d_ent": 9.354, "glue_d": 0.890, "c100": 0.1198},
    "resnet18_lr0.125_wd0.0001_s0": {"d_ent": None, "glue_d": 1.790, "c100": 0.2732},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch="resnet18"):
    from torchvision import models as tvm
    return getattr(tvm, arch)(num_classes=10)


def loaders(bs=128, workers=6):
    from torchvision import datasets, transforms
    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    te_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    tr = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"), transform=tr_tf)
    te = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"), transform=te_tf)
    return (
        torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True, num_workers=workers,
                                    pin_memory=True, persistent_workers=True, drop_last=False),
        torch.utils.data.DataLoader(te, batch_size=256, shuffle=False, num_workers=workers,
                                    pin_memory=True, persistent_workers=True),
    )


class PenultimateHook:
    """抓 resnet18 avgpool 输出（512 维，E1/E2 同款）。"""

    def __init__(self, model):
        self.feats = None
        model.avgpool.register_forward_hook(self._hook)

    def _hook(self, _m, _inp, out):
        self.feats = out.flatten(1)

    def get(self):
        return self.feats


def forcing_loss(feats, labels, kind, num_classes=10, rand_dir=None):
    """batch 级 forcing loss。feats: (B, d) float32（不 detach，梯度穿回 trunk）。

    whiten      : 类内协方差拉平到单位谱（各向同性目标）
    disperse    : pooled 谱熵最大化（幂迭代近似 top 特征值，D 的直接代理）
    concentrate : pooled 谱集中（−Σp² 最大化 ⇒ 谱压 top-1）
    placebo     : feats 在固定随机方向上的投影能量（与谱无关的正则）
    """
    d = feats.shape[1]
    if kind == "placebo":
        # 固定随机方向上的投影能量，按 batch 级 ||x||²/d 归一化到 O(1)：
        # E[(x·u)²] = ||x||²/d，故 E[L]≈1；原始量级 ~700，λ=0.1 会压过 CE~2.3。
        # 分母 detach——梯度只走分子方向。
        with torch.no_grad():
            scale = (feats.pow(2).sum(-1).mean() / d).detach().clamp_min(1e-8)
        return (feats @ rand_dir).pow(2).mean() / scale
    if kind == "whiten":
        # 池化类内协方差 Sw（E5 fit_ops 同口径：Σ_k XccᵀXcc / N）→
        # 与各向同性目标 I·tr/d 的相对 Frobenius 距离。
        # 注意不用逐类 C_k：batch 内每类 ~13 样本秩亏严重，目标会把
        # 秩亏方向的方差拉到 0（等效压掉类内散布），语义变成 D↓。
        mu = feats.mean(0, keepdim=True)
        Sw = feats.new_zeros((d, d))
        for k in range(num_classes):
            Xk = feats[labels == k]
            if Xk.shape[0] < 2:
                continue
            Xcc = Xk - Xk.mean(0, keepdim=True)
            Sw = Sw + Xcc.T @ Xcc
        Sw = Sw / max(feats.shape[0], 1)
        tr = Sw.diagonal().mean().clamp_min(1e-8)
        return ((Sw - torch.eye(d, device=feats.device) * tr) / tr) \
            .pow(2).sum() / (d * d)
    # disperse / concentrate：pooled top-K 谱上的熵。
    # 【v2 修复 2026-11-06】原实现把 8 步幂迭代（matmul/div/QR）整体挂进
    # autograd：特征值简并时特征向量梯度 ~1/eigengap 爆炸，一步 NaN 永久
    # 毒化权重（表现：20ep 内 test 恒 0.10、d_ent 全 NaN；λ=0.1 与 0.005
    # 均复现 ⇒ 非剂量问题，是梯度结构缺陷）。修复：特征向量在
    # no_grad + float64 + CPU eigh 下求解（数值稳健），梯度只走 Rayleigh 商
    #   r_k = v_kᵀ C v_k，  ∂r_k/∂C = v_k v_kᵀ（v 固定 ⇒ 有界）
    # ——特征值梯度的事实上界技巧。谱方向语义与单元测试一致，未改动。
    Xc = feats - feats.mean(0, keepdim=True)
    C = Xc.T @ Xc / max(Xc.shape[0] - 1, 1)
    C = C / C.diagonal().mean().clamp_min(1e-8)   # 尺度归一（谱形状不变量）
    K = min(32, d)
    with torch.no_grad():
        try:
            _, evecs = torch.linalg.eigh(C.detach().double().cpu())
            V = torch.flip(evecs, dims=(1,))[:, :K].to(feats.device).float()
        except Exception:
            return None   # 调用方护栏：本 batch 跳过 forcing，只走 CE
    r = (V * (C @ V)).sum(0)               # Rayleigh 商 r_k = v_kᵀ C v_k（梯度走这里）
    p = r.clamp_min(0)
    p = p / p.sum().clamp_min(1e-8)
    if kind == "disperse":
        return (p * (p + 1e-8).log()).sum()      # Σp·log p（尖谱≈0 最大，平谱最小 ⇒ 最小化 D↑）
    if kind == "concentrate":
        return -(p * (p + 1e-8).log()).sum()     # −Σp·log p = −H（尖谱≈0 最小 ⇒ 最小化 D↓）
    raise ValueError(kind)


@torch.no_grad()
def d_ent_of(feats):
    """谱熵指数 D（d_ent 口径），监控用。

    forcing 下的特征可能数值病态（disperse 把谱推平后条件数恶化，
    cuSOLVER eigh 报 error 498）——失败时退化到 numpy float64 再试，
    再失败返回 NaN 并打点（监控字段允许 NaN，不阻塞训练）。
    """
    Xc = feats - feats.mean(0, keepdim=True)
    C = (Xc.T @ Xc / max(Xc.shape[0] - 1, 1)).double()
    try:
        lam = torch.flip(torch.linalg.eigvalsh(C), dims=(0,)).clamp_min(0)
    except Exception:
        try:
            lam_np = np.linalg.eigvalsh(C.cpu().numpy().astype(np.float64))[::-1]
            lam = torch.as_tensor(np.clip(lam_np, 0, None), device=feats.device)
        except Exception:
            return float("nan")
    p = lam / lam.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    corr = tot = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        p = model(xb).argmax(1)
        corr += (p == yb).sum().item()
        tot += yb.numel()
    return corr / tot


@torch.no_grad()
def extract_features(model, loader, device, max_n=None):
    hook = PenultimateHook(model)
    model.eval()
    X, y = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        model(xb)
        X.append(hook.get().float().cpu().numpy())
        y.append(yb.numpy())
    X = np.concatenate(X)
    y = np.concatenate(y)
    if max_n is not None and len(X) > max_n:
        idx = np.random.default_rng(0).permutation(len(X))[:max_n]
        X, y = X[idx], y[idx]
    return X, y


def linear_probe_acc(Xtr, ytr, Xte, yte, num_classes, device, epochs=40, lr=1e-2,
                     wd=0.0, seed=0):
    """r1_id_ood.linear_probe 同款协议（口径对齐，逐行对应）：

    标准化（train 统计量）→ SGD(momentum=0.9, wd=0) + cosine 40ep → bs=min(512,n)。
    E1/E3 的 c100 数字全部出自这个协议，P4 的对照可比性依赖这里不偏离。
    """
    torch.manual_seed(seed)
    dev = device
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    yte_t = torch.as_tensor(yte, dtype=torch.long, device=dev)

    mu = Xtr_t.mean(0, keepdim=True)
    sd = Xtr_t.std(0, keepdim=True) + 1e-6
    Xtr_t = (Xtr_t - mu) / sd
    Xte_t = (Xte_t - mu) / sd

    clf = nn.Linear(Xtr_t.shape[1], num_classes).to(dev)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    n = Xtr_t.shape[0]
    bs = min(512, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            crit(clf(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        sched.step()

    with torch.no_grad():
        pred = clf(Xte_t).argmax(1)
        return float((pred == yte_t).float().mean().item())


def glue_dimension(X, labels, device=None, n_sub=500, seed=0, watchdog=900):
    """GLUE dimension（端点判定口径）。E1 同款 multiprocessing 看门狗。

    返回 None 表示超时/失败（与 E1 的 glue_timeout 字段一致）。
    signal.alarm 不杀 C 扩展里的死循环（cvxopt QP 挂起教训），
    必须用子进程 + terminate。
    """
    import multiprocessing as mp

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    classes = sorted(set(labels.tolist()))
    manifolds = []
    for c in classes:
        pts = X[labels == c].T.astype(np.float64)      # (d, n_c)
        manifolds.append(pts)
    manifolds = downsample_manifolds(manifolds, n_sub, seed=seed)

    def _work(q):
        try:
            q.put(("ok", glue_analysis(manifolds, n_t=200, seed=seed)))
        except Exception as e:                          # pragma: no cover
            q.put(("err", repr(e)))

    q = ctx.Queue()
    p = ctx.Process(target=_work, args=(q,), daemon=True)
    p.start()
    p.join(watchdog)
    if p.is_alive():
        p.terminate()
        p.join(5)
        return None
    try:
        st, v = q.get(timeout=60)
    except Exception:
        return None
    if st != "ok":
        return None
    return float(v["dimension"]), float(v["radius"]), float(v["utility"])


def train_one_forcing(tag, kind, lam, epochs, trl, tel, probe_loaders, device,
                      outdir, ckpt_every=20, seed=0):
    """全程 forcing 训练一个模型。返回结果行。"""
    set_seed(seed)
    arch = tag.split("_lr")[0]
    lr = float(tag.split("_lr")[1].split("_wd")[0])
    wd = float(tag.split("_wd")[1].split("_s")[0])
    model = build_model(arch).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    hook = PenultimateHook(model)

    rng = np.random.default_rng(seed + 42)
    rand_dir = torch.as_tensor(
        rng.standard_normal(model.fc.in_features).astype(np.float32),
        device=device)
    rand_dir = rand_dir / rand_dir.norm()

    ck = Path(outdir) / f"{tag}__{kind}_lam{lam}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    d_curve = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        corr = tot = fl_sum = n_batches = n_bad_fl = 0
        for xb, yb in trl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            ce = crit(out, yb)
            feats = hook.get()
            fl = forcing_loss(feats, yb, kind, rand_dir=rand_dir)
            # 【护栏 v2】forcing 损失数值异常（eigh 失败返回 None / NaN /Inf）
            # 时本 batch 退化为纯 CE，绝不让毒化梯度进 opt.step()。
            if fl is not None and torch.isfinite(fl):
                loss = ce + lam * fl
            else:
                loss = ce
                n_bad_fl += 1
            loss.backward()
            opt.step()
            corr += (out.argmax(1) == yb).sum().item()
            tot += yb.numel()
            fl_sum += float(fl.item()) if (fl is not None and torch.isfinite(fl)) else 0.0
            n_batches += 1
        sched.step()
        test_acc = eval_acc(model, tel, device)
        hist.append({"epoch": ep + 1, "train_acc": round(corr / tot, 4),
                     "test_acc": round(test_acc, 4),
                     "forcing_loss": round(fl_sum / max(n_batches, 1), 5)})

        if (ep + 1) % ckpt_every == 0 or ep == epochs - 1:
            model.eval()
            xb, _ = next(iter(tel))
            xb = xb.to(device)
            model(xb)
            d_now = d_ent_of(hook.get())
            d_curve.append({"epoch": ep + 1, "d_ent_batch": round(d_now, 3)})
            torch.save({"model": model.state_dict(), "ep": ep + 1, "tag": tag,
                        "kind": kind, "lam": lam, "history": hist,
                        "d_curve": d_curve}, ck)
            print(f"  [{tag}|{kind}@{lam}] ep{ep+1}/{epochs} "
                  f"test={test_acc:.4f} d_ent~{d_now:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # 端点测量：ID acc + c100 探针 + GLUE dimension
    Xid_tr, yid_tr = extract_features(model, probe_loaders["id_tr"], device)
    Xid_te, yid_te = extract_features(model, probe_loaders["id_te"], device)
    Xc_tr, yc_tr = extract_features(model, probe_loaders["c100_tr"], device)
    Xc_te, yc_te = extract_features(model, probe_loaders["c100_te"], device)

    id_probe = linear_probe_acc(Xid_tr, yid_tr, Xid_te, yid_te, 10, device, seed=seed)
    c100 = linear_probe_acc(Xc_tr, yc_tr, Xc_te, yc_te, 100, device, seed=seed)

    # D 端点：d_ent（全 ID train 特征）+ GLUE dimension（端点判定口径）
    # forcing 后特征可能数值病态（eigh 不收敛）——加抖动重试（jitter 是
    # 标准处理：谱在 1e-6 尺度抖动下不变，监控字段允许这个近似）
    Xc_all = Xid_tr - Xid_tr.mean(0, keepdims=True)
    C = (Xc_all.T @ Xc_all / max(len(Xc_all) - 1, 1)).astype(np.float64)
    lam_eig = None
    for jitter in (0.0, 1e-9, 1e-7, 1e-5):
        try:
            Cj = C if jitter == 0.0 else C + jitter * np.trace(C) / C.shape[0] \
                * np.eye(C.shape[0])
            lam_eig = np.linalg.eigvalsh(Cj)[::-1].clip(0)
            break
        except np.linalg.LinAlgError:
            continue
    if lam_eig is None:
        d_ent_full = float("nan")
    else:
        p = lam_eig / max(lam_eig.sum(), 1e-12)
        pp = p[p > 0]
        d_ent_full = float(np.exp(-(pp * np.log(pp)).sum()))

    gd = glue_dimension(Xid_tr, yid_tr, n_sub=500, seed=seed)

    row = {
        "arm": "p4_pilot",
        "model": tag,
        "arch": arch,
        "kind": kind,
        "lambda": lam,
        "seed": seed,
        "lr": lr, "wd": wd,
        "test_acc": round(hist[-1]["test_acc"], 4),
        "id_probe_acc": round(id_probe, 4),
        "c100_probe": round(c100, 4),
        "d_ent_id": round(d_ent_full, 3),
        "glue_d": (round(gd[0], 4) if gd else None),
        "glue_r": (round(gd[1], 4) if gd else None),
        "glue_u": (round(gd[2], 4) if gd else None),
        "glue_timeout": gd is None,
        "d_curve": d_curve,
        "base": BASE_READ.get(tag, {}),
        "secs": round(time.time() - t0, 1),
    }
    return row, ck


def load_base_readings(ckpt_dir, models, probe_loaders, device, seed=0):
    """base 对照臂读数：同种子 ckpt_grid 模型直接测（与 forcing 臂同协议）。"""
    rows = []
    for tag in models:
        fp = Path(ckpt_dir) / f"{tag}.pt"
        if not fp.exists():
            print(f"[base missing] {fp}", flush=True)
            continue
        sd = torch.load(fp, map_location="cpu", weights_only=False)
        model = build_model(tag.split("_lr")[0]).to(device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()
        Xid_tr, yid_tr = extract_features(model, probe_loaders["id_tr"], device)
        Xid_te, yid_te = extract_features(model, probe_loaders["id_te"], device)
        Xc_tr, yc_tr = extract_features(model, probe_loaders["c100_tr"], device)
        Xc_te, yc_te = extract_features(model, probe_loaders["c100_te"], device)
        id_probe = linear_probe_acc(Xid_tr, yid_tr, Xid_te, yid_te, 10, device, seed=seed)
        c100 = linear_probe_acc(Xc_tr, yc_tr, Xc_te, yc_te, 100, device, seed=seed)
        Xc_all = Xid_tr - Xid_tr.mean(0, keepdims=True)
        C = (Xc_all.T @ Xc_all / max(len(Xc_all) - 1, 1)).astype(np.float64)
        try:
            lam_eig = np.linalg.eigvalsh(C)[::-1].clip(0)
        except np.linalg.LinAlgError:
            Cj = C + 1e-7 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
            lam_eig = np.linalg.eigvalsh(Cj)[::-1].clip(0)
        p = lam_eig / max(lam_eig.sum(), 1e-12)
        pp = p[p > 0]
        d_ent_full = float(np.exp(-(pp * np.log(pp)).sum()))
        gd = glue_dimension(Xid_tr, yid_tr, n_sub=500, seed=seed)
        rows.append({
            "arm": "p4_base", "model": tag, "kind": "base", "lambda": 0.0,
            "seed": seed, "test_acc": round(float(sd.get("history", [{}])[-1]
                                                .get("test_acc", float("nan"))), 4),
            "id_probe_acc": round(id_probe, 4), "c100_probe": round(c100, 4),
            "d_ent_id": round(d_ent_full, 3),
            "glue_d": (round(gd[0], 4) if gd else None),
            "glue_timeout": gd is None,
            "secs": 0.0,
        })
        print(f"[base] {tag} d_ent={d_ent_full:.3f} c100={c100:.4f} "
              f"id={id_probe:.4f} glue_d={rows[-1]['glue_d']}", flush=True)
        del model
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--out", default="/mnt/mydisk/xinguan-brainai/runs/v2_p4_forcing/pilot")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lambda", dest="lam", type=float, default=PILOT_LAM)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", nargs="+", default=PILOT_KINDS)
    ap.add_argument("--models", nargs="+", default=PILOT_MODELS)
    ap.add_argument("--skip-base", action="store_true",
                    help="base 对照臂读数已有（断点续跑）")
    args = ap.parse_args()

    device = args.device
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fp_json = outdir / "train_forcing_p4.json"

    rows = []
    if fp_json.exists():
        rows = json.load(open(fp_json))["rows"]
        print(f"[resume] {len(rows)} rows loaded", flush=True)
    done_keys = {(r["model"], r["kind"], r["lambda"]) for r in rows}

    print("[1/4] 数据与探针加载器", flush=True)
    trl, tel = loaders(args.batch, args.workers)

    # 探针加载器：CIFAR-100 train/test + CIFAR-10 train/test（无增强）
    from torchvision import datasets, transforms
    ev_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    c100_tr = datasets.ImageFolder(os.path.join(DATA, "cifar100", "train"), transform=ev_tf)
    c100_te = datasets.ImageFolder(os.path.join(DATA, "cifar100", "test"), transform=ev_tf)
    id_tr = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"), transform=ev_tf)
    id_te = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"), transform=ev_tf)
    probe_loaders = {
        "id_tr": torch.utils.data.DataLoader(id_tr, batch_size=256, shuffle=False,
                                             num_workers=args.workers, pin_memory=True),
        "id_te": torch.utils.data.DataLoader(id_te, batch_size=256, shuffle=False,
                                             num_workers=args.workers, pin_memory=True),
        "c100_tr": torch.utils.data.DataLoader(c100_tr, batch_size=256, shuffle=False,
                                               num_workers=args.workers, pin_memory=True),
        "c100_te": torch.utils.data.DataLoader(c100_te, batch_size=256, shuffle=False,
                                               num_workers=args.workers, pin_memory=True),
    }

    # base 对照臂（同种子 grid checkpoint）
    if not args.skip_base:
        have_base = {(r["model"], r["kind"]) for r in rows if r["arm"] == "p4_base"}
        if not have_base:
            print("[2/4] base 对照臂读数", flush=True)
            base_rows = load_base_readings(
                "/mnt/mydisk/xinguan-brainai/runs/ckpt_grid",
                args.models, probe_loaders, device, seed=args.seed)
            rows.extend(base_rows)
            json.dump({"rows": rows}, open(fp_json, "w"), indent=1)

    print(f"[3/4] forcing 训练：{len(args.models)} 模型 × {len(args.kinds)} 算子 "
          f"× λ={args.lam} × {args.epochs}ep", flush=True)
    for tag in args.models:
        for kind in args.kinds:
            key = (tag, kind, args.lam)
            if key in done_keys:
                print(f"  skip {tag}|{kind}（已完成）", flush=True)
                continue
            print(f"  === {tag} | {kind} @ λ={args.lam} ===", flush=True)
            row, _ = train_one_forcing(tag, kind, args.lam, args.epochs, trl, tel,
                                       probe_loaders, device, outdir, seed=args.seed)
            rows.append(row)
            json.dump({"rows": rows}, open(fp_json, "w"), indent=1)
            b = row["base"] or {}
            print(f"  => d_ent {b.get('d_ent')}->{row['d_ent_id']} "
                  f"c100 {b.get('c100')}->{row['c100_probe']} "
                  f"id={row['id_probe_acc']} glue_d={row['glue_d']} "
                  f"({row['secs']}s)", flush=True)
            del row
            torch.cuda.empty_cache()

    print("[4/4] done:", len(rows), "rows ->", fp_json, flush=True)
    json.dump({"rows": rows}, open(fp_json, "w"), indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
