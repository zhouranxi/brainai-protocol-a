"""E5d-ceiling: direct test of the saturation / headroom conjectures.

The E3 recipe stratification (causality paper, "Recipe stratification
(post-hoc, exploratory)") left two conjectures explicitly untested:

  C1 (operator saturation): at the lr-1.0 recipe the whitening operator
     saturates -- no headroom above the floor -- so measured probe benefit
     vanishes.  The negative model (lr1.0_wd5e-05_s0, baseline D_eff 0.87,
     the smallest in the grid) is the poster child.
  C2 (high-D headroom): models already sitting at high D_eff have little
     whitening headroom, independent of recipe.

This script measures BOTH directly, using the E5c protocol (in-rank
condition number of the probe-train covariance + 40-epoch probe training
curve, original vs dose-whitened space) on ALL EIGHT E3 models instead of
the three E5b models.  Three of the eight overlap E5c and serve as a
seed-pinned replication check.

Measurable predictions:
  C1: the lr-1.0 model that showed no E3 benefit (lr1.0_wd5e-05_s0) shows
      an anomalously small conditioning improvement and/or no reachable-loss
      reduction under whitening, unlike the sharper-recipe models.
  C2: across the 8 models, baseline D_eff correlates negatively with the
      probe benefit of whitening (gamma=0) at fixed epoch.

Exploratory (not preregistered) -- appendix numbers only.
Output: runs/v2_e5d_ceiling/intervene_e5d_ceiling.json (resumable).
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

from intervene_e5c_cond import DOSES, probe_curve, spectrum_stats  # noqa: E402
from intervene_v2_paper import (build_model_arch, fit_ops,  # noqa: E402
                                sample_subset_idx, transform)
from r1_id_ood import extract_features, make_loader  # noqa: E402

# the eight E3 paired models (v2_e3.json), the conjecture population
E3_MODELS = [
    "resnet18_lr0.125_wd2.5e-05_s1",
    "resnet18_lr1.0_wd2.5e-05_s0",
    "resnet18_lr1.0_wd5e-05_s0",
    "resnet18_lr0.125_wd0.0001_s0",
    "resnet18_lr0.5_wd5e-05_s2",
    "resnet18_lr0.5_wd0.0002_s2",
    "resnet18_lr0.5_wd0.0001_s1",
    "resnet18_lr0.125_wd0.0002_s2",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir",
                   default="/mnt/mydisk/xinguan-brainai/runs/ckpt_grid")
    p.add_argument("--data", default="/mnt/mydisk/xinguan-brainai/data")
    p.add_argument("--out", default=str(ROOT / "runs" / "v2_e5d_ceiling"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.6)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "intervene_e5d_ceiling.json"
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

    for name in E3_MODELS:
        if all((name, repr(g), sp) in done for g in DOSES
               for sp in ("orig", "dose")):
            print(f"[skip] {name}: all conditions present", flush=True)
            continue
        ck = Path(args.ckpt_dir) / f"{name}.pt"
        if not ck.exists():
            print(f"[missing ckpt] {ck}", flush=True)
            continue
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

        # baseline D_eff of the id train features (same construction the
        # GLUE pipeline uses): participation of the top eigenvalues is not
        # needed here -- we record the effective dimension proxy used by
        # the paper's stratification discussion (glue_d comes from e3 runs;
        # here we record the feature-space stable rank as a local analogue)
        Xc = Xtr - Xtr.mean(0, keepdims=True)
        lam = np.linalg.eigvalsh(
            (Xc.T @ Xc / max(len(Xc) - 1, 1)).astype(np.float64))[::-1]
        lam = np.maximum(lam, 0.0)
        stable_rank = float(lam.sum() / lam[0])
        ent = lam / lam.sum()
        ent = ent[ent > 0]
        d_ent = float(np.exp(-(ent * np.log(ent)).sum()))

        idx = sample_subset_idx(len(Xc_tr), args.train_frac, args.seed)
        Xs, ys = Xc_tr[idx], np.asarray(yc_tr)[idx]

        for g in DOSES:
            for space, Xa, Xb in (("orig", Xs, Xc_te),
                                  ("dose", transform(Xs, op, "recolor", g),
                                   transform(Xc_te, op, "recolor", g))):
                key = (name, repr(g), space)
                if key in done:
                    continue
                st = spectrum_stats(Xa)
                curve = probe_curve(Xa, ys, Xb, yc_te, 100, device,
                                    seed=args.seed)
                rows.append({
                    "arm": "e5d_ceiling",
                    "model": name,
                    "arch": arch,
                    "kind": "recolor",
                    "param": g,
                    "space": space,
                    "stable_rank_id": round(stable_rank, 2),
                    "d_ent_id": round(d_ent, 2),
                    **st,
                    "curve": curve,
                    "lr": float(name.split("_lr")[1].split("_wd")[0]),
                    "wd": float(name.split("_wd")[1].split("_s")[0]),
                    "seed": int(name.split("_s")[-1]),
                    "secs": round(time.time() - t0, 1),
                })
                json.dump({"rows": rows}, open(fp, "w"), indent=1)
                print(f"[row] {name} g={g} {space} "
                      f"cond={st['in_rank_cond']:.3g} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    print("done:", len(rows), "rows")


if __name__ == "__main__":
    main()
