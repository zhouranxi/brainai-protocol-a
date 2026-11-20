#!/usr/bin/env python3
"""P21 (ADR-0026) phenomenology batch: compression proxies on locked corpus.

Zero training. For each of 68 ckpt_grid baselines + 8 dose-axis
checkpoints, extract penultimate features on (a) clean test set and
(b) 3 corruptions x 3 severities of CIFAR-10-C, then compute frozen
proxies:

  L_knn  - non-spectral code length: median kNN distance x N
  dL     - code-length stability: L_knn(corrupt) - L_knn(clean)

Adjudication C1-C4 frozen in ADR-0026 before any run.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, '/mnt/mydisk/xinguan-brainai/code')
os.environ.setdefault("BRAINAI_DATA_ROOT", "/mnt/mydisk/xinguan-brainai/data")

import numpy as np
import torch
from torchvision import transforms

from train_forcing_p4 import (CIFAR10_MEAN, CIFAR10_STD, DATA,
                              build_model, set_seed)
from p19_smoothness import arch_from_name

RUNS = Path('/mnt/mydisk/xinguan-brainai/runs')
OUT = RUNS / 'v2_p21' / 'p21_phase0.json'
OUT.parent.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda:0')
K = 10
FEAT_N = 4000
CORRUPTIONS = ('fog', 'gaussian_blur', 'pixelate')
SEVS = (1, 3, 5)

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def load_model(ck):
    set_seed(123)
    model = build_model(arch_from_name(ck.name)).to(DEVICE)
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    return model


def feats_from_imgs(model, imgs_np, ys_np):
    """Feature extraction from a numpy image array (N,32,32,3) uint8."""
    model.eval()
    feats, labels = [], []
    bs = 512
    with torch.no_grad():
        for i in range(0, len(imgs_np), bs):
            batch = np.stack([np.asarray(im) for im in imgs_np[i:i+bs]])
            x = torch.stack([tf(im) for im in batch]).to(DEVICE)
            f = model_acts(model, x)
            feats.append(f.cpu().numpy())
            labels.append(ys_np[i:i+bs])
    F = np.concatenate(feats).astype(np.float32)
    Y = np.concatenate(labels)
    rng = np.random.default_rng(123)
    idx = rng.permutation(len(F))[:FEAT_N]
    return F[idx], Y[idx]


def model_acts(model, x):
    """Penultimate activations via forward hook on avgpool/conv5."""
    acts = {}
    anchor = getattr(model, 'avgpool', None) or model.conv5
    h = anchor.register_forward_hook(lambda m, i, o: acts.__setitem__(
        'f', o.mean(dim=(2, 3)) if o.dim() == 4 else o.flatten(1)))
    model(x)
    h.remove()
    return acts['f'].float()


def l_knn(F, k=K):
    """Non-spectral code-length proxy: median kNN distance x N."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(F)
    d, _ = nn.kneighbors(F)
    return float(np.median(d[:, k]) * len(F))


def bootstrap_sigma(F, n=8):
    """Noise floor for L_knn under subsampling."""
    rng = np.random.default_rng(7)
    vals = []
    for _ in range(n):
        idx = rng.permutation(len(F))[:len(F) // 2]
        vals.append(l_knn(F[idx]))
    return float(np.std(vals))


def main():
    # data sources
    c10c_root = os.path.join(DATA, 'CIFAR-10-C')
    labels_np = np.load(os.path.join(c10c_root, 'labels.npy'))
    corr_imgs = {c: np.load(os.path.join(c10c_root, f'{c}.npy'))
                 for c in CORRUPTIONS}
    # clean test set
    from torchvision import datasets
    ds = datasets.ImageFolder(os.path.join(DATA, 'cifar_full', 'test'),
                              transform=None)
    clean_imgs = np.array([np.asarray(im) for im, _ in ds])
    clean_ys = np.array([y for _, y in ds])

    rows = []
    done = set()
    if OUT.exists():
        rows = json.loads(OUT.read_text())['rows']
        done = {r['tag'] for r in rows}

    # 1) 68 baselines
    grid = sorted(RUNS.glob('ckpt_grid/*.pt'))
    print(f"grid: {len(grid)} ckpts", flush=True)
    # 2) dose axis
    dose = []
    for prefix in ('p8_s0_seed2', 'p10_s0_seed3'):
        for lam in ('0.03', '0.1', '1.0', '3.0'):
            d = RUNS / 'v2_p4_forcing' / f'{prefix}_lam{lam}'
            cks = sorted(d.glob('*whiten*.pt'))
            if cks:
                dose.append((f'{prefix}_lam{lam}', cks[-1]))
    print(f"dose: {len(dose)} ckpts", flush=True)

    all_jobs = [(ck.stem, ck) for ck in grid] + dose
    for j, (tag, ck) in enumerate(all_jobs):
        if tag in done:
            print(f"skip {tag}", flush=True)
            continue
        try:
            model = load_model(ck)
            # clean features + L_knn
            F, Y = feats_from_imgs(model, clean_imgs, clean_ys)
            L_id = l_knn(F)
            sig = bootstrap_sigma(F)
            # corruption sweep: dL per (corruption, severity)
            dl = {}
            for c in CORRUPTIONS:
                for s in SEVS:
                    imgs = corr_imgs[c][(s - 1) * 10000: s * 10000]
                    ys = labels_np[(s - 1) * 10000: s * 10000]
                    Fc, Yc = feats_from_imgs(model, imgs, ys)
                    dl[f'{c}_s{s}'] = round(l_knn(Fc) - L_id, 3)
            row = {'tag': tag, 'family': ('grid' if j < len(grid) else 'dose'),
                   'L_knn': round(L_id, 3), 'sigma': round(sig, 3), 'dL': dl}
            rows.append(row)
            del model
            torch.cuda.empty_cache()
            OUT.write_text(json.dumps({'rows': rows}, indent=1))
            print(f"[{j+1}/{len(all_jobs)}] {tag} L={L_id:.1f} sig={sig:.1f}", flush=True)
        except Exception as e:
            print(f"ERROR {tag}: {type(e).__name__} {e}", flush=True)
    print(f"written: {OUT} ({len(rows)} rows)", flush=True)


if __name__ == '__main__':
    main()
