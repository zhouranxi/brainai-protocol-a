#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PC-A0 (ADR-0029): predictive-coding gate probe architecture.

ResNet-18 trunk with lateral predictors: each layer's features are
predicted by a light 1x1-conv head from the previous layer (target
stop-gradient). Loss = CE + beta * recon. Inference carries NO extra
computation (predictors are train-only).

beta=0 reduces to the FF baseline through the same code path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralPredictor(nn.Module):
    """1x1 conv head predicting the next layer's features; spatial sizes
    differ between stages (32/16/8/4), so the target is spatially pooled to
    the input's resolution before comparison (mean-pool, stop-grad)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, z, tgt):
        # pool target down to the predictor input's spatial size
        h, w = z.shape[-2:]
        if tgt.shape[-2:] != (h, w):
            tgt = F.adaptive_avg_pool2d(tgt, (h, w))
        return F.mse_loss(self.proj(z), tgt.detach())


class PCResNet(nn.Module):
    """ResNet-18 + lateral predictors; train-only PC loss."""

    def __init__(self, beta: float = 0.0, num_classes: int = 10):
        super().__init__()
        from torchvision import models as tvm
        self.trunk = tvm.resnet18(num_classes=num_classes)
        # predictors between successive stages (stem->l1, l1->l2, l2->l3, l3->l4)
        # channel progression: stem 64 -> l1 64 -> l2 128 -> l3 256 -> l4 512
        chans = [64, 64, 128, 256, 512]
        self.predictors = nn.ModuleList([
            LateralPredictor(chans[i], chans[i + 1]) for i in range(4)
        ])
        self.beta = beta

    def forward(self, x, record=None):
        t = self.trunk
        z = t.conv1(x)
        z = t.bn1(z)
        z = t.relu(z)
        zs = [z]                       # pre-layer1
        z = t.layer1(z); zs.append(z)
        z = t.layer2(z); zs.append(z)
        z = t.layer3(z); zs.append(z)
        z = t.layer4(z); zs.append(z)
        pooled = t.avgpool(z).flatten(1)
        logits = t.fc(pooled)
        if self.beta == 0.0 or not self.training:
            return logits
        # PC loss: predictor i maps zs[i] -> zs[i+1] (stop-grad target)
        pc = 0.0
        for i, pred in enumerate(self.predictors):
            pc = pc + pred(zs[i], zs[i + 1])
        pc = pc / len(self.predictors)
        if record is not None:
            record["pc_loss"] = float(pc.detach())
        # pack recon loss so the trainer can add beta * pc to CE
        self._pc = pc
        return logits

    def pc_loss(self):
        return getattr(self, "_pc", torch.tensor(0.0))

    def layer_errors(self, x):
        """Per-layer prediction errors (probe PC-P1; eval mode)."""
        t = self.trunk
        z = t.relu(t.bn1(t.conv1(x)))
        zs = [z]
        z = t.layer1(z); zs.append(z)
        z = t.layer2(z); zs.append(z)
        z = t.layer3(z); zs.append(z)
        z = t.layer4(z); zs.append(z)
        errs = []
        with torch.no_grad():
            for i, pred in enumerate(self.predictors):
                errs.append(float(pred(zs[i], zs[i + 1])))
        return errs  # length 4


def build_pc(beta: float, seed: int = 123):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return PCResNet(beta=beta)
