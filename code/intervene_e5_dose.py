#!/usr/bin/env python3
"""E5: dose-response battery for P3 revision (ADR-0016 App.8 + 修订一).

Arms (all use existing checkpoints, no new training):
  --arm e5b   : dose-response for the formation-whitening resurrection channel.
                Per model, per dose g in {0.75, 0.0, -0.5} (gain = lam**((g-1)/2):
                0.75 = partial whiten, 0.0 = full whiten = E3 original, -0.5 =
                over-whiten), three sub-arms:
                  B1 probe-space-only: c100 probe trained AND evaluated in the
                     dose-whitened c100 feature space. No ID formation at all.
                     -> H-cond predicts monotone gain here alone.
                  B2 formation readout: 10-class head trained on whitened ID
                     subset (linear_probe protocol), evaluated on ORIGINAL ID
                     test features and on dose-whitened ID test features.
                     -> H-geom predicts independent gain on ORIGINAL features.
                  B3 negative control: c100 probe in rotate space (spectrum
                     preserved, conditioning not improved). H-cond predicts
                     ~zero gain vs ctrl.
                Plus placebo formation-matched control (B2c): head trained on
                original subset + param-matched placebo noise (norm-matched to
                the whitening feature displacement). H-reg predicts comparable
                gain; H-cond/H-geom predict much less.
                Plus glue_d on dose-whitened ID subsample (dose->dD curve).
  --arm e5d   : MLP probe control (2 hidden 512 ReLU) on E1's decisive
                conditions subset {base, recolor 0.0, gauss_aniso 2.0, rotate}
                -> tests whether E1's linear-probe findings survive a
                nonlinear readout (Q7c).

Preregistered interpretation lives in paper/adr_0016.md App.8 修订一
(written before any E5 data). Resume-safe: skips (model, arm, kind, param).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from r1_id_ood import (build_model, extract_features,  # noqa: E402
                       make_loader)
from intervene_v2_paper import (build_model_arch, fit_ops, get_head,  # noqa: E402
                                glue_with_timeout, probe_acc,
                                sample_subset_idx, subsample_per_class,
                                transform)

# doses as recolor param (gain exponent form); 0.0 reproduces E3 exactly
DOSES = (0.75, 0.0, -0.5)
E5B_MODELS = [
    "resnet18_lr0.125_wd2.5e-05_s1",   # low-lr representative (E3 gain +0.19pp)
    "resnet18_lr0.5_wd5e-05_s2",       # mid (E3 gain +3.06pp)
    "resnet18_lr0.125_wd0.0002_s2",    # high-gain representative (+5.62pp)
]
E5B_EXT_MODELS = [
    "shufflenet_v2_x1_0_lr0.125_wd0.0002_s0",  # cross-arch falsifier (ii)
    "shufflenet_v2_x1_0_lr0.5_wd0.0002_s0",
]
E5D_MODELS = [
    "resnet18_lr0.5_wd5e-05_s2",
    "resnet18_lr0.125_wd0.0002_s2",
]
E5D_CONDITIONS = [("base", None), ("recolor", 0.0),
                  ("gauss_aniso", 2.0), ("rotate", None)]


def mlp_probe_acc(Xtr, ytr, Xte, yte, num_classes, device, epochs=40, seed=0):
    """2-hidden-512 ReLU probe, same optimization protocol as linear_probe."""
    import torch.nn as nn
    torch.manual_seed(seed)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32)
    mu, sd = Xtr_t.mean(0, keepdim=True), Xtr_t.std(0, keepdim=True) + 1e-8
    Xtr_t = (Xtr_t - mu) / sd
    Xte_t = (Xte_t - mu) / sd
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.ReLU(),
                        nn.Linear(512, 512), nn.ReLU(),
                        nn.Linear(512, num_classes)).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
    steps = int(np.ceil(len(Xtr_t) / 256)) * epochs
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, len(Xtr_t), (256,), generator=g)
        xb, yb = Xtr_t[idx].to(device), torch.as_tensor(
            np.asarray(ytr)[idx.numpy()], dtype=torch.long, device=device)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(xb), yb)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred = []
        for i in range(0, len(Xte_t), 1024):
            pred.append(net(Xte_t[i:i + 1024].to(device)).argmax(1).cpu())
    return float((torch.cat(pred).numpy() == np.asarray(yte)).mean())


def load_model(name, ckpt_dir, device):
    ck = Path(ckpt_dir) / f"{name}.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    arch = sd.get("arch") or name.split("_lr")[0]
    model = build_model_arch(arch).to(device)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    return model, arch, sd


def run_e5b(args, device):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "intervene_v2.json"
    rows = []
    if fp.exists():
        try:
            rows = json.load(open(fp))["rows"]
        except Exception:
            rows = []
    models = E5B_MODELS + (E5B_EXT_MODELS if args.ext else [])
    done = {(r["model"], r.get("e5", ""), repr(r.get("param"))) for r in rows}

    data = Path(args.data)
    loaders = {
        "id_tr": make_loader(str(data / "cifar_full" / "train"), 256),
        "id": make_loader(str(data / "cifar_full" / "test"), 256),
        "c100_tr": make_loader(str(data / "cifar100" / "train"), 256),
        "c100_te": make_loader(str(data / "cifar100" / "test"), 256),
    }

    for mi, name in enumerate(models):
        need = [(g, s) for g in DOSES for s in ("b1", "b2", "b2c", "b3", "glue")
                if (name, s, repr(g)) not in done]
        if not need:
            print(f"[skip] {name}", flush=True)
            continue
        model, arch, sd = load_model(name, args.ckpt_dir, device)
        t0 = time.time()
        Xtr, _, ytr10 = extract_features(model, loaders["id_tr"], device)
        Xid, _, yid = extract_features(model, loaders["id"], device)
        Xc_tr, _, yc_tr = extract_features(model, loaders["c100_tr"], device)
        Xc_te, _, yc_te = extract_features(model, loaders["c100_te"], device)
        op = fit_ops(Xtr, ytr10)
        Xtr, Xid, Xc_tr, Xc_te = (np.asarray(a) for a in
                                  (Xtr, Xid, Xc_tr, Xc_te))

        # ---- per-dose sub-arms ----
        for g in DOSES:
            # B1: probe-space-only (train + eval c100 in dose-whitened space)
            key = (name, "b1", repr(g))
            if key not in done:
                acc = probe_acc(
                    transform(Xc_tr, op, "recolor", g), yc_tr,
                    transform(Xc_te, op, "recolor", g), yc_te,
                    100, device, epochs=args.probe_epochs, seed=args.seed)
                rows.append({"arm": "e5b", "e5": "b1", "model": name,
                             "arch": arch, "kind": "recolor", "param": g,
                             "protocol": "linear_probe_v2",
                             "lr": sd.get("lr"), "wd": sd.get("wd"),
                             "seed": sd.get("seed"),
                             "c100_probe": round(acc, 4),
                             "secs": round(time.time() - t0, 1)})
                fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False,
                                         indent=1), encoding="utf-8")
                print(f"[e5b b1 {mi+1}/{len(models)}] {name} g={g} "
                      f"c100={acc:.4f}", flush=True)

            # B2: formation readout -- 10-class head on whitened ID subset,
            #     evaluated on ORIGINAL and dose-whitened ID test features
            key = (name, "b2", repr(g))
            if key not in done:
                idx = sample_subset_idx(len(Xtr), args.train_frac, args.seed)
                Xh = transform(Xtr[idx], op, "recolor", g)
                yh = ytr10[idx]
                acc_orig = probe_acc(Xh, yh, Xid, yid, 10, device,
                                     epochs=args.probe_epochs,
                                     seed=args.seed)
                acc_wh = probe_acc(Xh, yh, transform(Xid, op, "recolor", g),
                                   yid, 10, device,
                                   epochs=args.probe_epochs, seed=args.seed)
                rows.append({"arm": "e5b", "e5": "b2", "model": name,
                             "arch": arch, "kind": "recolor", "param": g,
                             "protocol": "linear_probe_v2",
                             "lr": sd.get("lr"), "wd": sd.get("wd"),
                             "seed": sd.get("seed"),
                             "id_fc_acc_orig": round(acc_orig, 4),
                             "id_fc_acc_recolor": round(acc_wh, 4),
                             "secs": round(time.time() - t0, 1)})
                fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False,
                                         indent=1), encoding="utf-8")
                print(f"[e5b b2 {mi+1}/{len(models)}] {name} g={g} "
                      f"id_orig={acc_orig:.4f} id_wh={acc_wh:.4f}", flush=True)

            # B2c: placebo formation-matched control (H-reg arm).
            #      param matched to the feature-norm displacement of dose g:
            #      ||f(x)-x||_rms / sigma
            key = (name, "b2c", repr(g))
            if key not in done:
                disp = np.linalg.norm(
                    transform(Xtr[idx][:256], op, "recolor", g)
                    - Xtr[idx][:256], axis=1).mean()
                sig = float(op["sigma"])
                p_param = round(float(disp) / sig, 4)
                Xh = Xtr[idx] + p_param * sig * _fixed_direction(Xtr.shape[1])
                yh = ytr10[idx]
                acc = probe_acc(Xh, yh, Xid, yid, 10, device,
                                epochs=args.probe_epochs, seed=args.seed)
                rows.append({"arm": "e5b", "e5": "b2c", "model": name,
                             "arch": arch, "kind": "placebo", "param": g,
                             "placebo_param": p_param, "protocol":
                             "linear_probe_v2",
                             "lr": sd.get("lr"), "wd": sd.get("wd"),
                             "seed": sd.get("seed"),
                             "id_fc_acc_orig": round(acc, 4),
                             "secs": round(time.time() - t0, 1)})
                fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False,
                                         indent=1), encoding="utf-8")
                print(f"[e5b b2c {mi+1}/{len(models)}] {name} g={g} "
                      f"p={p_param} id={acc:.4f}", flush=True)

            # B3: rotate negative control for conditioning story
            key = (name, "b3", repr(g))
            if key not in done:
                acc = probe_acc(
                    transform(Xc_tr, op, "rotate", None), yc_tr,
                    transform(Xc_te, op, "rotate", None), yc_te,
                    100, device, epochs=args.probe_epochs, seed=args.seed)
                rows.append({"arm": "e5b", "e5": "b3", "model": name,
                             "arch": arch, "kind": "rotate", "param": g,
                             "protocol": "linear_probe_v2",
                             "lr": sd.get("lr"), "wd": sd.get("wd"),
                             "seed": sd.get("seed"),
                             "c100_probe": round(acc, 4),
                             "secs": round(time.time() - t0, 1)})
                fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False,
                                         indent=1), encoding="utf-8")
                print(f"[e5b b3 {mi+1}/{len(models)}] {name} g={g} "
                      f"c100_rot={acc:.4f}", flush=True)

            # glue_d on dose-whitened ID subsample (dose->dD curve)
            key = (name, "glue", repr(g))
            if key not in done:
                Xsub, ysub = subsample_per_class(Xid, yid, args.m_per_class,
                                                 10, seed=args.seed)
                Xsub_w = transform(Xsub, op, "recolor", g)
                res = glue_with_timeout(Xsub_w, ysub, args.m_per_class,
                                        args.n_points, args.n_t, args.seed,
                                        timeout=args.glue_timeout)
                rows.append({"arm": "e5b", "e5": "glue", "model": name,
                             "arch": arch, "kind": "recolor", "param": g,
                             "lr": sd.get("lr"), "wd": sd.get("wd"),
                             "seed": sd.get("seed"),
                             "glue_d": (round(float(res["dimension"]), 4)
                                        if res["dimension"] is not None
                                        else None),
                             "glue_timeout": bool(res.get("timeout")),
                             "secs": round(time.time() - t0, 1)})
                fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False,
                                         indent=1), encoding="utf-8")
                print(f"[e5b glue {mi+1}/{len(models)}] {name} g={g} "
                      f"D={res['dimension']}", flush=True)

        # ctrl row once per model: c100 probe on original space (= E3 ctrl)
        key = (name, "ctrl", repr(None))
        if key not in done:
            acc = probe_acc(Xc_tr, yc_tr, Xc_te, yc_te, 100, device,
                            epochs=args.probe_epochs, seed=args.seed)
            rows.append({"arm": "e5b", "e5": "ctrl", "model": name,
                         "arch": arch, "kind": "base", "param": None,
                         "protocol": "linear_probe_v2",
                         "lr": sd.get("lr"), "wd": sd.get("wd"),
                         "seed": sd.get("seed"),
                         "c100_probe": round(acc, 4),
                         "secs": round(time.time() - t0, 1)})
            fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False,
                                     indent=1), encoding="utf-8")
            print(f"[e5b ctrl {mi+1}/{len(models)}] {name} c100={acc:.4f}",
                  flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


def _fixed_direction(d):
    rng = np.random.default_rng(20261103)  # same seed as E1 placebo
    u = rng.standard_normal(d)
    return u / np.linalg.norm(u)


def run_e5d(args, device):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "intervene_v2.json"
    rows = []
    if fp.exists():
        try:
            rows = json.load(open(fp))["rows"]
        except Exception:
            rows = []
    done = {(r["model"], r["kind"], repr(r.get("param"))) for r in rows}

    data = Path(args.data)
    loaders = {
        "id_tr": make_loader(str(data / "cifar_full" / "train"), 256),
        "id": make_loader(str(data / "cifar_full" / "test"), 256),
        "c100_tr": make_loader(str(data / "cifar100" / "train"), 256),
        "c100_te": make_loader(str(data / "cifar100" / "test"), 256),
    }
    for mi, name in enumerate(E5D_MODELS):
        model, arch, sd = load_model(name, args.ckpt_dir, device)
        t0 = time.time()
        Xtr, _, ytr10 = extract_features(model, loaders["id_tr"], device)
        Xid, _, yid = extract_features(model, loaders["id"], device)
        Xc_tr, _, yc_tr = extract_features(model, loaders["c100_tr"], device)
        Xc_te, _, yc_te = extract_features(model, loaders["c100_te"], device)
        op = fit_ops(Xtr, ytr10)
        Xtr, Xid, Xc_tr, Xc_te = (np.asarray(a) for a in
                                  (Xtr, Xid, Xc_tr, Xc_te))
        for kind, p in E5D_CONDITIONS:
            key = (name, kind, repr(p))
            if key in done:
                continue
            Xc_tr_p = transform(Xc_tr, op, kind, p)
            Xc_te_p = transform(Xc_te, op, kind, p)
            lin = probe_acc(Xc_tr_p, yc_tr, Xc_te_p, yc_te, 100, device,
                            epochs=args.probe_epochs, seed=args.seed)
            mlp = mlp_probe_acc(Xc_tr_p, yc_tr, Xc_te_p, yc_te, 100, device,
                                epochs=args.probe_epochs, seed=args.seed)
            rows.append({"arm": "e5d", "model": name, "arch": arch,
                         "kind": kind, "param": p,
                         "protocol": "linear_probe_v2",
                         "lr": sd.get("lr"), "wd": sd.get("wd"),
                         "seed": sd.get("seed"),
                         "c100_probe": round(lin, 4),
                         "c100_mlp": round(mlp, 4),
                         "secs": round(time.time() - t0, 1)})
            fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False,
                                     indent=1), encoding="utf-8")
            print(f"[e5d {mi+1}/{len(E5D_MODELS)}] {name} {kind}({p}) "
                  f"lin={lin:.4f} mlp={mlp:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["e5b", "e5d"], required=True)
    ap.add_argument("--ckpt-dir", default=str(ROOT / "runs" / "ckpt_grid"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "v2_e5b"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ext", action="store_true",
                    help="include shufflenet falsifier models")
    ap.add_argument("--probe-epochs", type=int, default=40)
    ap.add_argument("--m-per-class", type=int, default=500)
    ap.add_argument("--n-points", type=int, default=50)
    ap.add_argument("--n-t", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--glue-timeout", type=float, default=900.0)
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.arm == "e5b":
        sys.exit(run_e5b(args, device))
    else:
        args.out = str(ROOT / "runs" / "v2_e5d")
        sys.exit(run_e5d(args, device))


if __name__ == "__main__":
    main()
