"""E5c: direct conditioning measurements for the H-cond hypothesis (appendix).

Round-2 grilling Q9b: the paper says whitening "is consistent with easing
optimization of a 100-class linear head" -- this turns the inference into a
measurement.

For the 3 E5b models x doses gamma in {0.75, 0, -0.5}:
  1. Spectrum conditioning of the CIFAR-100 probe-train covariance in
     ORIGINAL vs dose-whitened space. ReLU features are numerically
     rank-deficient, so a raw lambda_max/lambda_min condition number is
     degenerate (v1 of this script saturated at the eps floor). We instead
     report: numerical rank (r > 1e-10 * lambda_max), the IN-RANK condition
     number lambda_1/lambda_r, and spectral-decay quantiles
     (p50/p90/p99 of lam / lam_max).
  2. Linear-probe training curve under the B1 protocol: train on the 60%
     CIFAR-100 subset, evaluate on full CIFAR-100 test, recording train
     loss + test acc at epochs {1, 2, 4, 8, 16, 24, 32, 40}.

Exploratory (not preregistered) -- appendix figure + numbers only.
Output: runs/v2_e5c/intervene_e5c.json (resumable, row-appends).
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

from intervene_v2_paper import (build_model_arch, fit_ops,  # noqa: E402
                                sample_subset_idx, transform)
from r1_id_ood import extract_features, make_loader  # noqa: E402

DOSES = (0.75, 0.0, -0.5)
MODELS = [
    "resnet18_lr0.125_wd2.5e-05_s1",
    "resnet18_lr0.5_wd5e-05_s2",
    "resnet18_lr0.125_wd0.0002_s2",
]
EPOCH_TICKS = (1, 2, 4, 8, 16, 24, 32, 40)


def spectrum_stats(X, rel_rank_eps=1e-10):
    """Conditioning-relevant statistics of the feature covariance."""
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(0, keepdims=True)
    cov = Xc.T @ Xc / max(len(Xc) - 1, 1)
    lam = np.linalg.eigvalsh(cov)[::-1]
    lam = np.maximum(lam, 0.0)
    lmax = float(lam[0])
    rank = int((lam > rel_rank_eps * lmax).sum())
    in_rank_cond = float(lam[0] / lam[rank - 1]) if rank > 0 else float("inf")
    qs = np.percentile(lam / lmax, [50, 90, 99])
    return {
        "rank": rank,
        "dim": int(len(lam)),
        "in_rank_cond": round(in_rank_cond, 2),
        "spec_p50": round(float(qs[0]), 6),
        "spec_p90": round(float(qs[1]), 6),
        "spec_p99": round(float(qs[2]), 6),
    }


def probe_curve(Xtr, ytr, Xte, yte, num_classes, device, seed=0, epochs=40):
    """Train the shared-protocol linear probe, recording curves.

    Mirrors linear_probe (standardization, lr 1e-2 cosine, batch 256,
    seed-pinned sampling) but records train loss + test acc at ticks.
    """
    import torch.nn as nn
    torch.manual_seed(seed)
    Xtr_t = torch.as_tensor(np.asarray(Xtr), dtype=torch.float32)
    Xte_t = torch.as_tensor(np.asarray(Xte), dtype=torch.float32)
    mu, sd = Xtr_t.mean(0, keepdim=True), Xtr_t.std(0, keepdim=True) + 1e-8
    Xtr_t = (Xtr_t - mu) / sd
    Xte_t = (Xte_t - mu) / sd
    head = nn.Linear(Xtr_t.shape[1], num_classes).to(device)
    steps_per_epoch = int(np.ceil(len(Xtr_t) / 256))
    opt = torch.optim.SGD(head.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * steps_per_epoch)
    ytr_np = np.asarray(ytr)
    yte_np = np.asarray(yte)
    g = torch.Generator().manual_seed(seed + 1)
    curve = []
    head.train()
    for ep in range(1, epochs + 1):
        head.train()
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, len(Xtr_t), (256,), generator=g)
            xb = Xtr_t[idx].to(device)
            yb = torch.as_tensor(ytr_np[idx.numpy()], dtype=torch.long,
                                 device=device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(head(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
        if ep in EPOCH_TICKS:
            head.eval()
            with torch.no_grad():
                tr_loss, ntr, tr_ok = 0.0, 0, 0
                for i in range(0, len(Xtr_t), 4096):
                    xb = Xtr_t[i:i + 4096].to(device)
                    yb = torch.as_tensor(ytr_np[i:i + 4096],
                                         dtype=torch.long, device=device)
                    logits = head(xb)
                    tr_loss += float(nn.functional.cross_entropy(
                        logits, yb, reduction="sum"))
                    tr_ok += int((logits.argmax(1).cpu().numpy()
                                  == ytr_np[i:i + 4096]).sum())
                    ntr += len(xb)
                ok, nte = 0, 0
                for i in range(0, len(Xte_t), 4096):
                    xb = Xte_t[i:i + 4096].to(device)
                    pred = head(xb).argmax(1).cpu().numpy()
                    ok += int((pred == yte_np[i:i + 4096]).sum())
                    nte += len(xb)
            curve.append({"epoch": ep,
                          "train_loss": round(tr_loss / ntr, 4),
                          "train_acc": round(tr_ok / ntr, 4),
                          "c100_acc": round(ok / nte, 4)})
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default=str(ROOT / "runs" / "ckpt_grid"))
    p.add_argument("--data", default=str(ROOT / "data"))
    p.add_argument("--out", default=str(ROOT / "runs" / "v2_e5c"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.6)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "intervene_e5c.json"
    rows = []
    if fp.exists():
        try:
            rows = json.load(open(fp))["rows"]
        except Exception:
            rows = []
    done = {(r["model"], repr(r.get("param")), r["space"]) for r in rows}

    data = Path(args.data)
    loaders = {
        "id_tr": make_loader(str(data / "cifar_full" / "train"), 256),
        "c100_tr": make_loader(str(data / "cifar100" / "train"), 256),
        "c100_te": make_loader(str(data / "cifar100" / "test"), 256),
    }

    for name in MODELS:
        ck = Path(args.ckpt_dir) / f"{name}.pt"
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        arch = sd.get("arch") or name.split("_lr")[0]
        model = build_model_arch(arch).to(device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()
        t0 = time.time()

        Xtr, _, ytr10 = extract_features(model, loaders["id_tr"], device)
        Xc_tr, _, yc_tr = extract_features(model, loaders["c100_tr"], device)
        Xc_te, _, yc_te = extract_features(model, loaders["c100_te"], device)
        Xtr, Xc_tr, Xc_te = (np.asarray(a) for a in (Xtr, Xc_tr, Xc_te))
        op = fit_ops(Xtr, ytr10)

        # probe-train subset: fixed 60% sample, same as the B1 protocol
        idx = sample_subset_idx(len(Xc_tr), args.train_frac, args.seed)
        Xs, ys = Xc_tr[idx], np.asarray(yc_tr)[idx]

        for g in DOSES:
            for space, Xa, Xb in (("orig", Xs, Xc_te),
                                  ("dose", transform(Xs, op, "recolor", g),
                                   transform(Xc_te, op, "recolor", g))):
                key = (name, repr(g), space)
                if key in done:
                    continue
                spec = spectrum_stats(Xa)
                curve = probe_curve(Xa, ys, Xb, np.asarray(yc_te), 100,
                                    device, seed=args.seed)
                rows.append({"arm": "e5c", "model": name, "arch": arch,
                             "kind": "recolor", "param": g, "space": space,
                             **spec, "curve": curve,
                             "lr": sd.get("lr"), "wd": sd.get("wd"),
                             "seed": sd.get("seed"),
                             "secs": round(time.time() - t0, 1)})
                fp.write_text(json.dumps({"rows": rows}, indent=1),
                              encoding="utf-8")
                print(f"[e5c] {name} g={g} {space} rank={spec['rank']} "
                      f"cond={spec['in_rank_cond']:.2e} "
                      f"final={curve[-1]['c100_acc']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    print("done", len(rows), "rows")


if __name__ == "__main__":
    main()
