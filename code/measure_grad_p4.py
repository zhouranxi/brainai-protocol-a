#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实测各 forcing 算子的梯度模长 vs CE——定 λ 阶梯的依据（预注册修订前置）。"""
import sys

sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/code")
import numpy as np
import torch

from train_forcing_p4 import PenultimateHook, build_model, forcing_loss, loaders

device = "cuda"
model = build_model("resnet18").to(device)
hook = PenultimateHook(model)
trl, _ = loaders(bs=128, workers=2)
xb, yb = next(iter(trl))
xb, yb = xb.to(device), yb.to(device)
model.train()
crit = torch.nn.CrossEntropyLoss()

out = model(xb)
ce = crit(out, yb)
model.zero_grad(set_to_none=True)
ce.backward()
gn_ce = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e12).item()

rng = np.random.default_rng(0)
rd = torch.as_tensor(rng.standard_normal(512).astype(np.float32), device=device)
rd = rd / rd.norm()
for kind in ("whiten", "disperse", "concentrate", "placebo"):
    model.zero_grad(set_to_none=True)
    out = model(xb)
    fl = forcing_loss(hook.get(), yb, kind, rand_dir=rd)
    fl.backward()
    gn_fl = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e12).item()
    # 等效 λ（使 forcing 梯度 = 10% CE 梯度）
    print(f"{kind}: |g_ce|={gn_ce:.3f} |g_f|={gn_fl:.3f} "
          f"-> lambda@10%={0.1 * gn_ce / gn_fl:.2e}", flush=True)
print("GRAD-OK", flush=True)
