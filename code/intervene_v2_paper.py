#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务四论文级扩展：算子族 × 架构 × 训练期对照（预注册 v2.0，2026-11-04）。

承接 runs/probe（噪声/投影干预）与 runs/probe2（零噪声谱重着色）的发现：
「跨模型 D_eff–OOD 观察相关不可反演为干预处方；反号效应全部来自噪声注入」。
本脚本按论文化要求一次铺满三个审稿缺口：

  E1 跨算子族（8 族，统一 ID-only 拟合协议，作用丁推理期特征）
     gauss_iso   : X + a·σ̄·ε，ε~N(0,I)     噪声，抬高尾部方向 → D_eff↑
     gauss_aniso : X + a·Lε，LLᵀ=Sw         噪声，Sw 谱形加噪 → D_eff 近不变
     rotate      : 全谱随机正交旋转          确定性，D_eff 严格不变
     scale       : μ 中心整体缩放 s          确定性，D_eff 尺度不变量
     recolor(γ)  : 类内谱重着色（γ<1 白化）  确定性，D_eff↑
     tot_recolor : 总谱对照（谱特异性检查）
     placebo     : 固定单位向量噪声（阴性对照）

  E2 跨架构（resnet34/resnet50/shufflenet_v2_x1_0/mobilenet_v3_small ×4 配方，
     与 E1 的 8 个 resnet18 合并成跨架构表）：D_eff–OOD 观察混杂结构是否
     依赖 resnet18。

  E3 训练期 vs 推理期对照：白化算子改为**训练期**施加（冻结 trunk，白化
     ID train 子集后重训 fc 头），对照推理期 recolor(0)（同一算子、推理期
     施加、不动任何训练数据）。区分「表征形成期几何」与「推理期几何」。

预注册判据（跑前写下，2026-11-04）：
  P1 (E1)  噪声族（gauss_iso/gauss_aniso）逐模型 d(c100)/d(D) 斜率
           median ≤ −0.05 且同号率 ≥ 6/8 ⇒ 「噪声注入有害」跨算子复制；
           零噪声确定性族（recolor）|median 斜率| ≤ 0.02 ⇒ 「几何操纵弱效应」复制。
  P1b(E1)  gauss_aniso 在 D_eff 近不变条件下 c100 仍下降（功率匹配）⇒ 新分离格：
           「D 不变 + 噪声」 vs 「D 不变 + 确定性（rotate/scale）」。
  P2 (E2)  跨架构 base 观察斜率（每架构 4 配方 + resnet18 复用）全负
           （≥4/5 架构），且推理期 recolor 斜率幅度 ≤ 观察 10%。
  P3 (E3)  训练期白化 t-W 的 (c100, D_eff) 落在其 base 邻域，而推理期
           recolor(0) 不动 c100（|Δ|≤0.01）⇒ 形成期与推理期可分离，
           白化是「表征成形工具」而非「推理期修补工具」。

实现要点（相对 probe/probe2 的三处修改）：
  1. arch 从 ckpt 元数据读取，build_model 参照 train_grid.py 名单（7 架构通用）。
  2. GLUE 看门狗：glue_analysis 在子进程中运行，wall-clock 超时（默认 900s）
     即终止并记 glue_timeout=true（修复 probe2 中 resnet18_lr0.5_wd1e-4_s1
     的 cvxopt QP 无限挂起问题）。
  3. E3 训练臂：--arm e3，白化训练子集 + 重训 fc 头（trunk 冻结，无主动
     学习、无数据交换），输出独立行（arm="e3"）。

用法（学校 4 卡服务器 /mnt/mydisk/xinguan-brainai）：
  CUDA_VISIBLE_DEVICES=1 ~/venvs/dl/bin/python code/intervene_v2_paper.py \
    --arm eval --out runs/v2/e1_g1 --shard 0 --num-shards 3
  CUDA_VISIBLE_DEVICES=1 ~/venvs/dl/bin/python code/intervene_v2_paper.py \
    --arm e3 --out runs/v2/e3 --models <E1 的 8 模型>
断点续跑安全：增量写 JSON，按 (model, kind, param, arm) 跳过已完成条件。
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_id_ood import (build_model, extract_features, linear_probe,  # noqa: E402
                       make_loader, subsample_per_class)
from glue_impl import glue_analysis, downsample_manifolds  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "results").is_dir() else HERE.parent

# E1 用的 8 个 resnet18（与 probe/probe2 完全同清单，id_acc 0.851–0.884）
E1_MODELS = [
    "resnet18_lr0.125_wd2.5e-05_s1",
    "resnet18_lr1.0_wd2.5e-05_s0",
    "resnet18_lr1.0_wd5e-05_s0",
    "resnet18_lr0.125_wd0.0001_s0",
    "resnet18_lr0.5_wd5e-05_s2",
    "resnet18_lr0.5_wd0.0002_s2",
    "resnet18_lr0.5_wd0.0001_s1",
    "resnet18_lr0.125_wd0.0002_s2",
]

# E2 新架构（train_grid.py 名单中去掉 resnet18；convnext_tiny 太慢，备选不启用。
# ⚠️ 不用 mobilenet_v3：其 classifier 含非线性（Hardswish(Linear(576→1024))），
# avgpool 钩子抓到的 576 维不是末层 Linear 的输入空间，干预会做错空间。
# regnet_y_400mf 有 .fc，penultimate 钩子干净。）
E2_ARCHS = ["resnet34", "resnet50", "shufflenet_v2_x1_0", "regnet_y_400mf"]
# E2 训练配方：lr 两端 × wd 两端 × 1 seed（跨架构表每架构 4 点）
E2_LR = [0.5, 0.125]
E2_WD = [0.0002, 5e-05]
E2_SEEDS = [0]

# (kind, param)；param=None 即基线或不带参算子
CONDITIONS = (
    [("base", None)]
    + [("gauss_iso", a) for a in (0.5, 1.0, 2.0)]
    + [("gauss_aniso", a) for a in (0.5, 1.0, 2.0)]
    + [("rotate", None)]
    + [("scale", s) for s in (0.5, 2.0)]
    + [("recolor", g) for g in (1.5, 0.75, 0.5, 0.25, 0.0)]
    + [("tot_recolor", g) for g in (0.5, 0.0)]
    + [("placebo", a) for a in (1.0, 2.0)]
)

FAMILIES_FOR_SLOPE = ("gauss_iso", "gauss_aniso", "recolor", "tot_recolor")


# --------------------------------------------------------------------------- #
def _spec_decomp(cov, eps_rel=1e-6):
    """对称谱分解 + eps 地板（防白化爆炸）。返回 (lam_sorted_desc, V)。"""
    lam, V = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    floor = max(float(lam[0]) * eps_rel, 1e-12)
    lam = np.clip(lam, floor, None)
    return lam, V


def build_model_arch(arch):
    """与 train_grid.py 完全一致的架构表（7 种，含 resnet18）。"""
    from torchvision import models as tvm
    names = ["resnet18", "resnet34", "resnet50", "shufflenet_v2_x1_0",
             "regnet_y_400mf", "mobilenet_v3_small", "convnext_tiny"]
    if arch not in names:
        raise ValueError(arch)
    return getattr(tvm, arch)(num_classes=10)


def get_head(model):
    """取最后一层线性头：resnet*/shufflenet/regnet 是 .fc，
    mobilenet_v3 是 .classifier[-1]，convnext_tiny 是 .head。"""
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier[-1]
    if hasattr(model, "head"):
        return model.head
    raise ValueError(f"no head for {type(model).__name__}")


def fit_ops(X, y, num_classes=10):
    """ID-only：类内协方差谱（主）+ 总协方差谱（对照）+ 全局均值。"""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for c in range(num_classes):
        Xc = X[y == c]
        Xcc = Xc - Xc.mean(0)
        Sw += Xcc.T @ Xcc
    Sw /= max(len(X), 1)
    lam_w, V_w = _spec_decomp(Sw)
    St = np.cov(X, rowvar=False) if X.shape[1] < len(X) else Sw
    lam_t, V_t = _spec_decomp(np.asarray(St))
    return {"mu": mu, "lam_w": lam_w, "V_w": V_w,
            "lam_t": lam_t, "V_t": V_t,
            "sigma": float(np.sqrt(max(lam_w.mean(), 1e-12)))}


def transform(X, op, kind, param):
    """把算子作用到任意特征矩阵（类无关的固定映射，ID-only 拟合）。"""
    X = np.asarray(X)
    if kind == "base":
        return X
    if kind == "gauss_iso":
        rng = np.random.default_rng(10_000 + int(round(abs(param) * 10)))
        return X.astype(np.float64) \
            + param * op["sigma"] * rng.standard_normal(X.shape)
    if kind == "gauss_aniso":
        # 噪声协方差 = Sw 谱形：L = V diag(√λ) Vᵀ ⇒ LLᵀ = Sw（eps 截断前）
        rng = np.random.default_rng(11_000 + int(round(abs(param) * 10)))
        eps = rng.standard_normal(X.shape)
        L = (op["V_w"] * np.sqrt(op["lam_w"])) @ op["V_w"].T
        return X.astype(np.float64) + param * (eps @ L.T)
    if kind == "rotate":
        d = X.shape[1]
        rng = np.random.default_rng(12_000)
        M = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(M)
        Q = Q * np.sign(np.diag(R))          # 固定随机正交（det≈+1）
        return (X.astype(np.float64) - op["mu"]) @ Q + op["mu"]
    if kind == "scale":
        return op["mu"] + (X.astype(np.float64) - op["mu"]) * float(param)
    if kind in ("recolor", "tot_recolor"):
        lam, V = (op["lam_w"], op["V_w"]) if kind == "recolor" \
            else (op["lam_t"], op["V_t"])
        gain = lam ** ((float(param) - 1.0) / 2.0)
        dX = (X.astype(np.float64) - op["mu"]) @ V
        return op["mu"] + (dX * gain) @ V.T
    if kind == "placebo":
        rng = np.random.default_rng(20261103)
        u = rng.standard_normal(X.shape[1])
        u /= np.linalg.norm(u)
        return X.astype(np.float64) + float(param) * op["sigma"] * u
    raise ValueError(kind)


@torch.no_grad()
def head_acc(fc, X, y, device):
    xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    return float((fc(xt).argmax(1).cpu().numpy() == np.asarray(y)).mean())


def probe_acc(Xtr, ytr, Xte, yte, num_classes, device, epochs=40, seed=0):
    return linear_probe(
        torch.as_tensor(np.asarray(Xtr), dtype=torch.float32, device=device),
        torch.as_tensor(np.asarray(ytr), dtype=torch.long, device=device),
        torch.as_tensor(np.asarray(Xte), dtype=torch.float32, device=device),
        torch.as_tensor(np.asarray(yte), dtype=torch.long, device=device),
        num_classes=num_classes, epochs=epochs, seed=seed)


# --------------------------------------------------------------------------- #
# GLUE 看门狗：子进程运行，wall-clock 超时即杀（修 probe2 QP 挂起）
# --------------------------------------------------------------------------- #
def glue_with_timeout(Xg, yg, m_per_class, n_points, n_t, seed, timeout=900.0):
    import multiprocessing as mp

    # Linux 用 fork（子进程只跑 numpy/cvxopt，不碰 CUDA，秒级启动）；
    # 其他平台退回 spawn。
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    def _work(q):
        try:
            mans = [Xg[c * m_per_class:(c + 1) * m_per_class].T
                    for c in range(10)]
            sub = downsample_manifolds(mans, n_points=n_points, seed=seed)
            q.put(("ok", glue_analysis(sub, n_t=n_t, seed=seed)))
        except Exception as e:  # pragma: no cover
            q.put(("err", repr(e)))

    q = ctx.Queue()
    p = ctx.Process(target=_work, args=(q,), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        return {"dimension": None, "radius": None, "utility": None,
                "timeout": True}
    try:
        st, v = q.get(timeout=60)
    except Exception:
        return {"dimension": None, "radius": None, "utility": None,
                "timeout": False, "error": "no result from worker"}
    if st == "ok":
        v = dict(v)
        v["timeout"] = False
        return v
    return {"dimension": None, "radius": None, "utility": None,
            "timeout": False, "error": str(v)}


# --------------------------------------------------------------------------- #
# E3：训练期白化（冻结 trunk，白化 ID train 子集，重训 fc 头）
# --------------------------------------------------------------------------- #
def sample_subset_idx(n, frac, seed):
    """确定性子集抽样索引（两臂共用，保证配对一致）。"""
    rng = np.random.default_rng(seed)
    return rng.permutation(n)[:int(round(frac * n))]


def whiten_subset(X, y, op, frac=0.6, seed=0):
    idx = sample_subset_idx(len(X), frac, seed)
    Xw = transform(np.asarray(X)[idx], op, "recolor", 0.0)
    return Xw, np.asarray(y)[idx], idx


def train_head_whitened(model, Xw, yw, device, epochs=30, bs=256, lr=0.05,
                        seed=0, standardize=True):
    """只训分类头（trunk 冻结）。返回 (model, history)。

    standardize=True 时对训练子集做 per-dim z-score（均值/方差从训练子集
    估计，epoch 级固定）。原始 penultimate 特征范数 ~30，直接用 lr=0.05+
    动量训头部会发散（e3 首轮 t_acc 仅 0.15–0.36，e3c 对照臂同样差，
    证明是优化问题而非白化效应）；标准化后与 linear_probe 协议一致。
    """
    torch.manual_seed(seed)
    head = get_head(model)
    Xw = np.asarray(Xw, dtype=np.float64)
    if standardize:
        mu_h = Xw.mean(0)
        sd_h = Xw.std(0) + 1e-8
        Xw = (Xw - mu_h) / sd_h
    else:
        mu_h = sd_h = None
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed + 1)
    Xw_t = torch.as_tensor(np.asarray(Xw), dtype=torch.float32)
    yw_t = torch.as_tensor(np.asarray(yw), dtype=torch.long)
    n = len(Xw)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = corr = 0
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            xb = Xw_t[sel].to(device, non_blocking=True)
            yb = yw_t[sel].to(device, non_blocking=True)
            opt.zero_grad()
            out = head(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot += len(sel)
            corr += int((out.argmax(1) == yb).sum().item())
        sched.step()
        hist.append({"epoch": ep + 1, "train_acc": round(corr / max(tot, 1), 4)})
    return model, hist, (mu_h, sd_h)


def eval_head_standardized(model, mu_h, sd_h, X, y, device):
    """用训练臂的标准化统计量评估头部。mu_h=None 时直接评估。"""
    X = np.asarray(X, dtype=np.float64)
    if mu_h is not None:
        X = (X - mu_h) / sd_h
    return head_acc(get_head(model), X, y, device)


# --------------------------------------------------------------------------- #
def resolve_models(args, ckpt_dir):
    if args.models:
        return [m for m in args.models.split(",") if m]
    out = []
    for arch in [a for a in args.arch.split(",") if a]:
        for lr in E2_LR:
            for wd in E2_WD:
                for s in E2_SEEDS:
                    tag = f"{arch}_lr{lr}_wd{wd}_s{s}"
                    if (Path(ckpt_dir) / f"{tag}.pt").exists():
                        out.append(tag)
    return out


def run_eval(args, device):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "intervene_v2.json"

    models = resolve_models(args, args.ckpt_dir)
    if args.num_shards > 1:
        models = models[args.shard::args.num_shards]
    print(f"[eval] shard {args.shard}/{args.num_shards} models={models}",
          flush=True)

    rows = []
    if fp.exists():
        try:
            rows = json.load(open(fp))["rows"]
        except Exception:
            rows = []
    done = {(r["model"], r["kind"], repr(r.get("param")))
            for r in rows if r.get("arm", "eval") == "eval"}

    data = Path(args.data)
    loaders = {
        "id_tr": make_loader(str(data / "cifar_full" / "train"), 256),
        "id": make_loader(str(data / "cifar_full" / "test"), 256),
        "c100_tr": make_loader(str(data / "cifar100" / "train"), 256),
        "c100_te": make_loader(str(data / "cifar100" / "test"), 256),
    }
    merged = data / "cifar10c_merged"
    if merged.is_dir():
        loaders["c10c"] = make_loader(str(merged), 256)

    for mi, name in enumerate(models):
        ck = Path(args.ckpt_dir) / f"{name}.pt"
        if not ck.exists():
            print(f"[missing] {ck}", flush=True)
            continue
        try:
            sd = torch.load(ck, map_location="cpu", weights_only=False)
        except Exception:
            sd = torch.load(ck, map_location="cpu")
        arch = sd.get("arch") if isinstance(sd, dict) else None
        arch = arch or name.split("_lr")[0]
        model = build_model_arch(arch).to(device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()
        fc = get_head(model)

        t0 = time.time()
        Xtr, _, ytr10 = extract_features(model, loaders["id_tr"], device)
        Xid, _, yid = extract_features(model, loaders["id"], device)
        Xc_tr, _, yc_tr = extract_features(model, loaders["c100_tr"], device)
        Xc_te, _, yc_te = extract_features(model, loaders["c100_te"], device)
        Xk, _, yk = (extract_features(model, loaders["c10c"], device)
                     if "c10c" in loaders else (None, None, None))
        op = fit_ops(Xtr, ytr10)
        max_gain_w = float((op["lam_w"][0] / op["lam_w"][-1]) ** 0.5)
        Xsub, ysub = subsample_per_class(Xid, yid, args.m_per_class, 10,
                                         seed=args.seed)
        print(f"[{mi+1}/{len(models)}] {name} arch={arch} feats "
              f"{time.time()-t0:.0f}s", flush=True)

        for kind, param in CONDITIONS:
            key = (name, kind, repr(param))
            if key in done:
                continue
            tc = time.time()
            Xsub_p = transform(Xsub, op, kind, param)
            Xid_p = transform(Xid, op, kind, param)
            Xtr_p = transform(Xtr, op, kind, param)
            Xc_tr_p = transform(Xc_tr, op, kind, param)
            Xc_te_p = transform(Xc_te, op, kind, param)

            g = glue_with_timeout(Xsub_p, ysub, args.m_per_class,
                                  args.n_points, args.n_t, args.seed,
                                  timeout=args.glue_timeout)
            rec = {
                "arm": "eval", "model": name, "arch": arch,
                "lr": sd.get("lr"), "wd": sd.get("wd"),
                "seed": sd.get("seed"),
                "kind": kind, "param": param,
                "glue_d": (round(float(g["dimension"]), 4)
                           if g["dimension"] is not None else None),
                "glue_r": (round(float(g["radius"]), 4)
                           if g["radius"] is not None else None),
                "glue_u": (round(float(g["utility"]), 4)
                           if g["utility"] is not None else None),
                "glue_timeout": bool(g.get("timeout")),
                "id_fc_acc": round(head_acc(fc, Xid_p, yid, device), 4),
                "id_probe_acc": round(probe_acc(
                    Xtr_p, ytr10, Xid_p, yid, 10, device,
                    epochs=args.probe_epochs, seed=args.seed), 4),
                "c100_probe": round(probe_acc(
                    Xc_tr_p, yc_tr, Xc_te_p, yc_te, 100, device,
                    epochs=args.probe_epochs, seed=args.seed), 4),
                "c10c_acc": (round(head_acc(
                    fc, transform(Xk, op, kind, param), yk, device), 4)
                    if Xk is not None else None),
                "max_gain": round(max_gain_w, 1),
                "secs": round(time.time() - tc, 1),
            }
            rows.append(rec)
            done.add(key)
            print(f"  {kind}={param}: D={rec['glue_d']} "
                  f"idp={rec['id_probe_acc']} c100={rec['c100_probe']} "
                  f"to={rec['glue_timeout']} ({rec['secs']}s)", flush=True)
            fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False,
                                     indent=1), encoding="utf-8")

    summarize_eval(rows, outdir)
    return 0


def run_e3(args, device):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "intervene_v2.json"

    models = resolve_models(args, args.ckpt_dir)
    if not models:
        models = list(E1_MODELS)  # e3 默认跑 E1 的 8 个 resnet18
    rows = []
    if fp.exists():
        try:
            rows = json.load(open(fp))["rows"]
        except Exception:
            rows = []
    e3c = args.arm == "e3c"
    kind_name = "train_head_control" if e3c else "train_whiten"
    done = {(r["model"], r["kind"], repr(r.get("param")))
            for r in rows if r.get("arm") in ("e3", "e3c")}

    data = Path(args.data)
    loaders = {
        "id_tr": make_loader(str(data / "cifar_full" / "train"), 256),
        "id": make_loader(str(data / "cifar_full" / "test"), 256),
        "c100_tr": make_loader(str(data / "cifar100" / "train"), 256),
        "c100_te": make_loader(str(data / "cifar100" / "test"), 256),
    }

    for mi, name in enumerate(models):
        key = (name, kind_name, repr(None))
        if key in done:
            print(f"[skip] {name}", flush=True)
            continue
        ck = Path(args.ckpt_dir) / f"{name}.pt"
        if not ck.exists():
            print(f"[missing] {ck}", flush=True)
            continue
        try:
            sd = torch.load(ck, map_location="cpu", weights_only=False)
        except Exception:
            sd = torch.load(ck, map_location="cpu")
        arch = sd.get("arch") or name.split("_lr")[0]
        model = build_model_arch(arch).to(device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()  # 关键：BN 用 running stats（缺这行曾污染第一轮 E3 特征）

        t0 = time.time()
        Xtr, _, ytr10 = extract_features(model, loaders["id_tr"], device)
        Xid, _, yid = extract_features(model, loaders["id"], device)
        Xc_tr, _, yc_tr = extract_features(model, loaders["c100_tr"], device)
        Xc_te, _, yc_te = extract_features(model, loaders["c100_te"], device)
        op = fit_ops(Xtr, ytr10)

        # (a) 推理期对照：原头 × 白化特征（不动任何训练数据）
        Xid_w = transform(Xid, op, "recolor", 0.0)
        id_fc_inf = head_acc(get_head(model), Xid_w, yid, device)

        # (b) 训练臂：白化/对照两臂都用 linear_probe 协议（重初始化头部、
        #     per-dim 标准化、lr=1e-2、SGD+momentum、cosine）。
        #     首轮教训：从预训练头出发 lr=0.05 微调两臂都发散（t_acc 0.15–0.36）。
        Xtr_w_full = transform(Xtr, op, "recolor", 0.0)
        Xid_w2 = Xid_w
        if e3c:
            idx = sample_subset_idx(len(Xtr), args.train_frac, args.seed)
            Xh = np.asarray(Xtr)[idx]           # 原始特征，同子集
            yh = np.asarray(ytr10)[idx]
        else:
            Xh, yh, _ = whiten_subset(Xtr, ytr10, op, frac=args.train_frac,
                                      seed=args.seed)
        # 主读数（arm 内对照，消除子集差异）：同臂头部 → 两种 ID 特征。
        # 命名即语义：*_orig = 头部自己的训练空间；*_recolor = 同头评在白化空间。
        head_tw_orig = probe_acc(Xh, yh, Xid, yid, 10, device,
                                 epochs=args.probe_epochs, seed=args.seed)
        head_tw_recolor = probe_acc(Xh, yh, Xid_w2, yid, 10, device,
                                    epochs=args.probe_epochs, seed=args.seed)
        id_probe_tw = probe_acc(Xtr_w_full, ytr10, Xid_w2, yid, 10, device,
                                epochs=args.probe_epochs, seed=args.seed)
        c100_tw = probe_acc(transform(Xc_tr, op, "recolor", 0.0), yc_tr,
                            transform(Xc_te, op, "recolor", 0.0), yc_te,
                            100, device, epochs=args.probe_epochs,
                            seed=args.seed)
        c100_tw_ctrl = probe_acc(Xc_tr, yc_tr, Xc_te, yc_te, 100, device,
                                 epochs=args.probe_epochs, seed=args.seed)
        # 白化表征的 D_eff（推理期与训练臂共用同一表征变换）
        Xsub, ysub = subsample_per_class(Xid, yid, args.m_per_class, 10,
                                         seed=args.seed)
        Xsub_w = transform(Xsub, op, "recolor", 0.0)
        g = glue_with_timeout(Xsub_w, ysub, args.m_per_class, args.n_points,
                              args.n_t, args.seed, timeout=args.glue_timeout)

        rec = {
            "arm": "e3c" if e3c else "e3", "model": name, "arch": arch,
            "lr": sd.get("lr"), "wd": sd.get("wd"), "seed": sd.get("seed"),
            "kind": kind_name, "param": None,
            "protocol": "linear_probe_v2",
            "train_frac": args.train_frac,
            "id_fc_acc_inference_recolor": round(id_fc_inf, 4),
            # 重训头部评在自己的训练空间（对照臂=原始空间；白化臂=白化空间）
            "id_fc_acc_orig": round(head_tw_orig, 4),
            # 同一头评在白化空间（跨空间泛化读数）
            "id_fc_acc_recolor": round(head_tw_recolor, 4),
            "id_probe_acc": round(id_probe_tw, 4),
            "c100_probe": round(c100_tw, 4),
            "c100_probe_ctrl": round(c100_tw_ctrl, 4),
            "glue_d": (round(float(g["dimension"]), 4)
                       if g["dimension"] is not None else None),
            "glue_timeout": bool(g.get("timeout")),
            "secs": round(time.time() - t0, 1),
        }
        rows.append(rec)
        print(f"[e3 {mi+1}/{len(models)}] {name}: c100={rec['c100_probe']} "
              f"(ctrl={rec['c100_probe_ctrl']}) idfc_orig={rec['id_fc_acc_orig']} "
              f"(recolor={rec['id_fc_acc_recolor']}) idp={rec['id_probe_acc']} "
              f"D={rec['glue_d']} ({rec['secs']}s)", flush=True)
        fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False,
                                 indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
def summarize_eval(rows, outdir):
    from scipy.stats import pearsonr
    out = {"per_model": [], "per_arch_base": []}
    by_model = {}
    for r in rows:
        if r.get("arm", "eval") != "eval":
            continue
        by_model.setdefault(r["model"], []).append(r)

    for m, pts in sorted(by_model.items()):
        base = next((r for r in pts if r["kind"] == "base"), None)
        if base is None or base["glue_d"] is None:
            continue
        entry = {"model": m, "arch": base.get("arch"),
                 "base_d": base["glue_d"], "base_c100": base["c100_probe"],
                 "base_id_probe": base["id_probe_acc"]}
        for fam in FAMILIES_FOR_SLOPE:
            sel = sorted([r for r in pts if r["kind"] == fam and not
                          r.get("glue_timeout") and r["glue_d"] is not None]
                         + [base],
                         key=lambda r: -(r["param"] if r.get("param")
                                         is not None else 9))
            if len(sel) < 3:
                continue
            D = np.array([r["glue_d"] for r in sel])
            Y = np.array([r["c100_probe"] for r in sel])
            Yp = np.array([r["id_probe_acc"] for r in sel])
            entry[f"{fam}_slope_c100"] = round(float(np.polyfit(D, Y, 1)[0]), 5)
            entry[f"{fam}_slope_idprobe"] = round(
                float(np.polyfit(D, Yp, 1)[0]), 5)
            entry[f"{fam}_manip_r"] = round(float(pearsonr(
                np.arange(len(D)) * -1.0, D)[0]), 3)
        # 无参算子的成对差（D 不变族的关键读数）
        for kind in ("rotate",):
            r0 = next((r for r in pts if r["kind"] == kind), None)
            if r0 and r0["glue_d"] is not None:
                entry[f"{kind}_dD"] = round(r0["glue_d"] - base["glue_d"], 4)
                entry[f"{kind}_dc100"] = round(
                    r0["c100_probe"] - base["c100_probe"], 4)
        out["per_model"].append(entry)

    # 跨架构 base 观察相关（每架构内 D vs c100）
    arch_pts = {}
    for e in out["per_model"]:
        arch_pts.setdefault(e.get("arch") or "resnet18", []).append(
            (e["base_d"], e["base_c100"]))
    for arch, ps in sorted(arch_pts.items()):
        if len(ps) >= 3:
            D = np.array([p[0] for p in ps])
            Y = np.array([p[1] for p in ps])
            out["per_arch_base"].append({
                "arch": arch, "n": len(ps),
                "pearson_d_c100": round(float(pearsonr(D, Y)[0]), 3),
                "slope_d_c100": round(float(np.polyfit(D, Y, 1)[0]), 5),
            })
    (Path(outdir) / "intervene_v2_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["eval", "e3", "e3c"], default="eval")
    ap.add_argument("--ckpt-dir",
                    default="/mnt/mydisk/xinguan-brainai/runs/ckpt_grid")
    ap.add_argument("--data",
                    default=os.environ.get("BRAINAI_DATA_ROOT",
                                           "/mnt/mydisk/xinguan-brainai/data"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "v2"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default="")
    ap.add_argument("--arch", default=",".join(E2_ARCHS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--probe-epochs", type=int, default=40)
    ap.add_argument("--m-per-class", type=int, default=500)
    ap.add_argument("--n-points", type=int, default=50)
    ap.add_argument("--n-t", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--glue-timeout", type=float, default=900.0)
    # E3 专用
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--train-epochs", type=int, default=30)
    ap.add_argument("--head-lr", type=float, default=0.05)
    args = ap.parse_args()

    device = args.device
    if args.arm == "eval":
        return run_eval(args, device)
    return run_e3(args, device)  # e3 与 e3c 共用主循环，按 arm 分支


if __name__ == "__main__":
    raise SystemExit(main())
