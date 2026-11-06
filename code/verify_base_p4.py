#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""前台验证修复版 base 臂读数（一个模型，含 GLUE 端点），通过后再全量启动。"""
import os
import sys

sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/code")
os.chdir("/mnt/mydisk/xinguan-brainai")

import torch
from torchvision import datasets, transforms

from train_forcing_p4 import load_base_readings

device = "cuda"
ev_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
pl = {
    "id_tr": torch.utils.data.DataLoader(
        datasets.ImageFolder("data/cifar_full/train", transform=ev_tf),
        batch_size=256, num_workers=4, pin_memory=True),
    "id_te": torch.utils.data.DataLoader(
        datasets.ImageFolder("data/cifar_full/test", transform=ev_tf),
        batch_size=256, num_workers=4, pin_memory=True),
    "c100_tr": torch.utils.data.DataLoader(
        datasets.ImageFolder("data/cifar100/train", transform=ev_tf),
        batch_size=256, num_workers=4, pin_memory=True),
    "c100_te": torch.utils.data.DataLoader(
        datasets.ImageFolder("data/cifar100/test", transform=ev_tf),
        batch_size=256, num_workers=4, pin_memory=True),
}
rows = load_base_readings("runs/ckpt_grid", ["resnet18_lr1.0_wd5e-05_s0"],
                          pl, device, seed=0)
print("ROW:", rows[0], flush=True)
print("VERIFY-OK", flush=True)
