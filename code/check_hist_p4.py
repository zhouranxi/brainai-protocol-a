#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查普通训练前 3ep 的 test_acc 基线（判断冒烟 id 崩是 forcing 还是 lr=1.0 正常震荡）。"""
import torch

sd = torch.load("/mnt/mydisk/xinguan-brainai/runs/ckpt_grid/"
                "resnet18_lr1.0_wd5e-05_s0.pt", map_location="cpu",
                weights_only=False)
h = sd["history"]
print("lr1.0 前3ep:", h[:3])
print("lr1.0 ep20:", h[19])
print("lr1.0 末ep:", h[-1])
