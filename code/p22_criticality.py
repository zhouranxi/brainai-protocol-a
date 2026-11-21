#!/usr/bin/env python3
"""P22 (ADR-0027): criticality batch - GLUE quadruples on 68 baselines.

Extracts penultimate features per checkpoint (P19/P21 pipeline),
runs glue_analysis (n_t=200, one_vs_rest, kappa=0), stores the
full quadruple (glue_d / glue_r / glue_u / alpha) plus Phi_hat
= U*D/P. Zero training. Resume-safe.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, '/mnt/mydisk/xinguan-brainai/code')
os.environ.setdefault("BRAINAI_DATA_ROOT", "/mnt/mydisk/xinguan-brainai/data")

import numpy as np
import torch

from train_forcing_p4 import build_model, set_seed, extract_features
from glue_impl import glue_analysis, downsample_manifolds
from p19_smoothness import arch_from_name

RUNS = Path('/mnt/mydisk/xinguan-brainai/runs')
OUT = RUNS / 'v2_p22' / 'p22_phase0.json'
OUT.parent.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda:0')
FEAT_N = 4000
P_CLASSES = 10


def load_model(ck):
    set_seed(123)
    model = build_model(arch_from_name(ck.name)).to(DEVICE)
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    return model


def extract(model):
    """Penultimate features on cifar_full test, subsampled to FEAT_N."""
    from torchvision import datasets, transforms
    from train_forcing_p4 import CIFAR10_MEAN, CIFAR10_STD, DATA
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = datasets.ImageFolder(os.path.join(DATA, 'cifar_full', 'test'),
                              transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                     num_workers=4, pin_memory=True)
    Fs, Ys = extract_features(model, dl, DEVICE)
    F = np.asarray(Fs, dtype=np.float32)
    Y = np.asarray(Ys)
    rng = np.random.default_rng(123)
    idx = rng.permutation(len(F))[:FEAT_N]
    return F[idx], Y[idx]


def main():
    grid = sorted(RUNS.glob('ckpt_grid/*.pt'))
    print(f"grid: {len(grid)} ckpts", flush=True)

    rows = []
    done = set()
    if OUT.exists():
        rows = json.loads(OUT.read_text())['rows']
        done = {r['tag'] for r in rows}

    for j, ck in enumerate(grid):
        tag = ck.stem
        if tag in done:
            print(f"skip {tag}", flush=True)
            continue
        try:
            model = load_model(ck)
            F, Y = extract(model)
            del model
            torch.cuda.empty_cache()
            # manifolds: list of (D, N_mu) per class（glue 约定：特征维 × 点数）
            manifolds = [F[Y == c].T for c in range(P_CLASSES)]
            manifolds = downsample_manifolds(manifolds, 200, seed=0)
            res = glue_analysis(manifolds, n_t=200, kappa=0.0, seed=0)
            d, u = res['dimension'], res['utility']
            row = {'tag': tag, 'family': 'grid',
                   'glue_d': round(float(d), 4),
                   'glue_r': round(float(res['radius']), 4),
                   'glue_u': round(float(u), 4),
                   'alpha': round(float(res['alpha']), 4),
                   'phi_hat': round(float(u) * float(d) / P_CLASSES, 4)}
            rows.append(row)
            OUT.write_text(json.dumps({'rows': rows}, indent=1))
            print(f"[{j+1}/{len(grid)}] {tag} D={row['glue_d']} "
                  f"U={row['glue_u']} Phi={row['phi_hat']}", flush=True)
        except Exception as e:
            print(f"ERROR {tag}: {type(e).__name__} {e}", flush=True)
    print(f"written: {OUT} ({len(rows)} rows)", flush=True)


if __name__ == '__main__':
    main()
