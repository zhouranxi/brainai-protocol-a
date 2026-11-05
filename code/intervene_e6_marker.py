#!/usr/bin/env python3
"""E6: second-marker check (Q5c) -- does the dissociation replicate for
participation ratio (PR) and effective rank (erank)?

For every E1 model x condition: rebuild the transformed features (deterministic
linear ops from intervene_v2_paper.transform), compute covariance-spectrum
markers on the SAME ID-test subsample GLUE used, and emit one row per
(model, kind, param). No probe training here -- c100 accuracies come from the
existing E1 JSON and are merged in analysis (code/paper3_marker_merge.py).

Pure feature extraction + eigendecomposition: ~1 min per model.
Output: runs/v2_e6_marker/intervene_e6.json (incremental, resume-safe).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/mydisk/xinguan-brainai")
sys.path.insert(0, str(ROOT / "code"))

from r1_id_ood import extract_features, make_loader  # noqa: E402
from intervene_v2_paper import (CONDITIONS, E1_MODELS, build_model_arch,  # noqa: E402
                                fit_ops, subsample_per_class, transform)


def spectrum_markers(X):
    """PR and erank from covariance eigenvalues (X: n x d)."""
    Xc = X - X.mean(0, keepdims=True)
    # eigendecomposition of d x d covariance (d<=512, cheap)
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    s = lam.sum()
    if s <= 0:
        return None, None
    pr = float(s ** 2 / (lam ** 2).sum())
    p = lam / s
    p = p[p > 1e-12]
    erank = float(np.exp(-(p * np.log(p)).sum()))
    return round(pr, 2), round(erank, 3)


def main():
    outdir = ROOT / "runs" / "v2_e6_marker"
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "intervene_e6.json"
    rows = []
    if fp.exists():
        try:
            rows = json.load(open(fp))["rows"]
        except Exception:
            rows = []
    done = {(r["model"], r["kind"], repr(r.get("param"))) for r in rows}

    data = ROOT / "data"
    load_id = make_loader(str(data / "cifar_full" / "test"), 256)
    load_tr = make_loader(str(data / "cifar_full" / "train"), 256)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for mi, name in enumerate(E1_MODELS):
        if all((name, k, repr(p)) in done for k, p in CONDITIONS):
            print(f"[skip] {name}", flush=True)
            continue
        ck = ROOT / "runs" / "ckpt_grid" / f"{name}.pt"
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        arch = sd.get("arch") or name.split("_lr")[0]
        model = build_model_arch(arch).to(device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()
        t0 = time.time()
        Xtr, _, ytr10 = extract_features(model, load_tr, device)
        Xid, _, yid = extract_features(model, load_id, device)
        Xtr, Xid = np.asarray(Xtr), np.asarray(Xid)
        op = fit_ops(Xtr, ytr10)
        # same GLUE subsample protocol for marker comparability
        Xsub, _ = subsample_per_class(Xid, yid, 500, 10, seed=0)
        for kind, param in CONDITIONS:
            key = (name, kind, repr(param))
            if key in done:
                continue
            Xt = transform(Xsub, op, kind, param)
            pr, erank = spectrum_markers(Xt)
            rows.append({"model": name, "arch": arch, "kind": kind,
                         "param": param, "pr": pr, "erank": erank,
                         "secs": round(time.time() - t0, 1)})
            fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False,
                                     indent=1), encoding="utf-8")
        print(f"[e6 {mi+1}/{len(E1_MODELS)}] {name}: {len(CONDITIONS)} conds "
              f"({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    print("E6 done:", len(rows), "rows")


if __name__ == "__main__":
    main()
