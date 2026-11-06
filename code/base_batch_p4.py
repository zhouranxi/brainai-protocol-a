"""P4 全量批 base 读数驱动：按种子组测 ckpt_grid 全部配方。

用法：
  python base_batch_p4.py --tags <tag...> --seed 0 --out runs/v2_p4_forcing/full_base_s0.json

断点续跑：按 model 去重，已有行跳过。
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from train_forcing_p4 import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DATA,
    load_base_readings,
)


def build_probe_loaders(workers):
    from torchvision import datasets, transforms

    ev_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    c100_tr = datasets.ImageFolder(os.path.join(DATA, "cifar100", "train"), transform=ev_tf)
    c100_te = datasets.ImageFolder(os.path.join(DATA, "cifar100", "test"), transform=ev_tf)
    id_tr = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "train"), transform=ev_tf)
    id_te = datasets.ImageFolder(os.path.join(DATA, "cifar_full", "test"), transform=ev_tf)
    mk = lambda ds: torch.utils.data.DataLoader(  # noqa: E731
        ds, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True)
    return {"id_tr": mk(id_tr), "id_te": mk(id_te),
            "c100_tr": mk(c100_tr), "c100_te": mk(c100_te)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(outp))["rows"] if outp.exists() else []
    done = {r["model"] for r in rows}
    todo = [t for t in args.tags if t not in done]
    print(f"[base_batch] {len(todo)}/{len(args.tags)} todo (seed={args.seed})",
          flush=True)
    if not todo:
        print("[base_batch] all done", flush=True)
        return

    probe_loaders = build_probe_loaders(args.workers)
    ckpt_dir = "/mnt/mydisk/xinguan-brainai/runs/ckpt_grid"
    for i, tag in enumerate(todo):
        print(f"[base_batch {i+1}/{len(todo)}] {tag}", flush=True)
        rows.extend(load_base_readings(ckpt_dir, [tag], probe_loaders,
                                       args.device, seed=args.seed))
        json.dump({"rows": rows}, open(outp, "w"), indent=1)
    print(f"[base_batch] done: {len(rows)} rows -> {outp}", flush=True)


if __name__ == "__main__":
    main()
