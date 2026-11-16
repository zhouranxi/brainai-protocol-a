#!/usr/bin/env python3
"""P16 (ADR-0022): bottleneck testbed for hierarchical feedback.

Three bottlenecks x three configs (FF/REC/HFB from
models_hierarchy.py, pure-modulation revision):
  B1 mask   : structured occlusion, 25% visible blocks (train+test)
  B2 width  : stem+stage widths halved (capacity limit)
  B3 data   : 10% training data
Protocol = N1pp (wd 2e-3, RandAugment [B1 excluded], best-ckpt
early stop, 100ep). Measures: id/off/c100 probe (+c10c post-hoc).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("BRAINAI_DATA_ROOT", "/mnt/mydisk/xinguan-brainai/data")

import numpy as np
import torch
import torch.nn as nn

from models_hierarchy import HierarchyNet
from train_forcing_p4 import (CIFAR10_MEAN, CIFAR10_STD, DATA,
                              linear_probe_acc, set_seed)


def loaders_p16(mask_frac=0.0, data_frac=1.0, randaug=True,
                bs=128, workers=6):
    from torchvision import datasets, transforms
    tr_tf = [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip()]
    if randaug:
        tr_tf.append(transforms.RandAugment(2, 10))
    tr_tf += [transforms.ToTensor(),
              transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    tr_tf = transforms.Compose(tr_tf)
    te_tf = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    tr = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"),
                              transform=MaskTransform(tr_tf, mask_frac))
    te = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"),
                              transform=MaskTransform(te_tf, mask_frac))
    if data_frac < 1.0:  # B3: 类平衡下采样
        import random
        rng = random.Random(0)
        by_class = {}
        for idx, (_, y) in enumerate(tr.samples):
            by_class.setdefault(y, []).append(idx)
        keep = [i for ys in by_class.values()
                for i in rng.sample(ys, int(len(ys) * data_frac))]
        tr.samples = [tr.samples[i] for i in sorted(keep)]
    return (torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True,
                                        num_workers=workers, pin_memory=True,
                                        persistent_workers=True),
            torch.utils.data.DataLoader(te, batch_size=256, shuffle=False,
                                        num_workers=workers, pin_memory=True,
                                        persistent_workers=True))


class MaskTransform:
    """Apply base transform, then structured occlusion (8x8 blocks).

    Deterministic per-index mask via a fixed RNG seeded by index,
    so train and test see a consistent occlusion pattern per image.
    """

    def __init__(self, base, frac):
        self.base = base
        self.frac = frac

    def __call__(self, img):
        x = self.base(img)
        if self.frac <= 0:
            return x
        g = np.random.default_rng(0)  # 单张固定模式（SimpleClass 无法拿 index，用固定 25% 模式集合）
        mask = np.ones((32, 32), np.float32)
        n_blocks = int(16 * self.frac)  # 16 个 8x8 块中遮 n 个
        idx = g.permutation(16)[:n_blocks]
        for k in idx:
            r, c = divmod(int(k), 4)
            mask[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = 0.0
        return x * torch.as_tensor(mask).unsqueeze(0)


@torch.no_grad()
def eval_acc(model, loader, device, steps=None):
    model.eval()
    corr = tot = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb, steps=steps)
        corr += (out.argmax(1) == yb).sum().item()
        tot += yb.numel()
    return corr / tot


def probe_loaders(workers=4):
    from torchvision import datasets, transforms
    ev_tf = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    mk = lambda ds: torch.utils.data.DataLoader(ds, batch_size=256,
                                                shuffle=False,
                                                num_workers=workers,
                                                pin_memory=True)
    c100_tr = datasets.ImageFolder(os.path.join(DATA, "cifar100", "train"),
                                   transform=ev_tf)
    c100_te = datasets.ImageFolder(os.path.join(DATA, "cifar100", "test"),
                                   transform=ev_tf)
    return {"c100_tr": mk(c100_tr), "c100_te": mk(c100_te)}


@torch.no_grad()
def penultimate(model, x):
    z = model.stem(x)
    z = model.s1(z); z = model.s2(z); z = model.s3(z)
    import torch.nn.functional as F
    return F.adaptive_avg_pool2d(z, 1).flatten(1)


def run_one(cfg, seed, bottleneck, device, epochs=100):
    set_seed(seed)
    tag = f"p16_{bottleneck}_{cfg}_s{seed}"
    outdir = Path(f"/mnt/mydisk/xinguan-brainai/runs/v2_hierarchy/{tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "p16_result.json"
    if fp.exists():
        print(f"[skip] {tag}", flush=True)
        return json.loads(fp.read_text())

    kwargs = {"mask_frac": 0.75, "randaug": False} if bottleneck == "mask" \
        else {} if bottleneck == "width" \
        else {"data_frac": 0.1}
    model = HierarchyNet(mode=cfg,
                         widths=(32, 64, 128) if bottleneck == "width"
                         else (64, 128, 256)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=2e-3, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    trl, tel = loaders_p16(**kwargs)

    t0 = time.time()
    hist = []
    best_test, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in trl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            test = eval_acc(model, tel, device)
            hist.append({"epoch": ep + 1, "test_acc": round(test, 4)})
            if test > best_test:
                best_test = test
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            if (ep + 1) % 30 == 0:
                print(f"  [{tag}] ep{ep+1} test={test:.4f} "
                      f"best={best_test:.4f} ({time.time()-t0:.0f}s)",
                      flush=True)
    model.load_state_dict(best_state)

    id_test = eval_acc(model, tel, device)
    off_test = eval_acc(model, tel, device, steps=1) if cfg != "ff" else None
    pl = probe_loaders()
    feats_tr, ys_tr, feats_te, ys_te = [], [], [], []
    with torch.no_grad():
        for xb, yb in pl["c100_tr"]:
            feats_tr.append(penultimate(model, xb.cuda()).cpu().numpy())
            ys_tr.append(yb.numpy())
        for xb, yb in pl["c100_te"]:
            feats_te.append(penultimate(model, xb.cuda()).cpu().numpy())
            ys_te.append(yb.numpy())
    c100 = linear_probe_acc(np.concatenate(feats_tr), np.concatenate(ys_tr),
                            np.concatenate(feats_te), np.concatenate(ys_te),
                            100, device, seed=seed)
    row = {"bottleneck": bottleneck, "cfg": cfg, "seed": seed,
           "id_test": round(id_test, 4),
           "off_test": round(off_test, 4) if off_test is not None else None,
           "c100_probe": round(c100, 4), "hist": hist,
           "secs": round(time.time() - t0, 1)}
    fp.write_text(json.dumps({"rows": [row]}, indent=1))
    print(f"  => {tag}: id={id_test:.4f} off={off_test} c100={c100:.4f}",
          flush=True)
    return row


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bottlenecks", nargs="+",
                    default=["mask", "width", "data"])
    ap.add_argument("--cfgs", nargs="+", default=["ff", "rec", "hfb"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    for b in args.bottlenecks:
        for cfg in args.cfgs:
            for seed in args.seeds:
                run_one(cfg, seed, b, dev, epochs=args.epochs)
