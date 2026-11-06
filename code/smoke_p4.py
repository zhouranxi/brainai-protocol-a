#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 冒烟测试：单 batch 验证 forcing_loss 数值 + 梯度回传（不训练）。"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/code")
from train_forcing_p4 import (forcing_loss, PenultimateHook, build_model,
                              d_ent_of, loaders)

device = "cuda"
model = build_model("resnet18").to(device)
hook = PenultimateHook(model)
trl, _tel = loaders(bs=128, workers=2)
xb, yb = next(iter(trl))
xb, yb = xb.to(device), yb.to(device)
model.train()
out = model(xb)
feats = hook.get()
print("feats shape:", tuple(feats.shape), flush=True)

rng = np.random.default_rng(0)
rd = rng.standard_normal(512).astype(np.float32)
rand_dir = torch.as_tensor(rd, device=device)
rand_dir = rand_dir / rand_dir.norm()      # 脚本 train_one_forcing 里做了归一化
for kind in ("whiten", "disperse", "concentrate", "placebo"):
    fl = forcing_loss(feats, yb, kind, rand_dir=rand_dir)
    fl.backward(retain_graph=True)
    g = model.conv1.weight.grad
    gn = "None" if g is None else f"{float(g.norm()):.5f}"
    val = float(fl.detach())
    print(f"{kind}: loss={val:.5f} grad_to_conv1={gn}", flush=True)
    model.zero_grad(set_to_none=True)

model.eval()
with torch.no_grad():
    model(xb)
    print("d_ent(batch) =", round(d_ent_of(hook.get()), 3), flush=True)
print("SMOKE-OK", flush=True)
