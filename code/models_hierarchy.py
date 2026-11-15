#!/usr/bin/env python3
"""ADR-0019 N1: three-config hierarchical-feedback testbed (FF/REC/HFB).

Same trunk (3-stage small ResNet, CIFAR-10); the only differences:
  FF   : T=1, no lateral, no top-down
  REC  : T=3 lateral self-connections (no cross-level information)
  HFB  : T=3 top-down 1x1 conv projections (deep -> shallow modulation)

Feedback is switchable at inference (mode.forward(steps=1) == off).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = (nn.Identity() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm2d(cout)))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.short(x))


class HierarchyNet(nn.Module):
    """mode in {'ff','rec','hfb'}; steps = iterative depth at inference."""

    def __init__(self, mode="hfb", widths=(64, 128, 256), num_classes=10,
                 fb_scale=1.0):
        super().__init__()
        assert mode in ("ff", "rec", "hfb")
        self.mode = mode
        self.steps = 1 if mode == "ff" else 3
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        self.s1 = nn.Sequential(Block(widths[0], widths[0]),
                                Block(widths[0], widths[0]))
        self.s2 = nn.Sequential(Block(widths[0], widths[1], 2),
                                Block(widths[1], widths[1]))
        self.s3 = nn.Sequential(Block(widths[1], widths[2], 2),
                                Block(widths[2], widths[2]))
        self.head = nn.Linear(widths[2], num_classes)

        if mode == "rec":
            # 同层横向自连接（不跨层）：1x1 conv 门控残差
            self.lat1 = nn.Conv2d(widths[0], widths[0], 1, bias=False)
            self.lat2 = nn.Conv2d(widths[1], widths[1], 1, bias=False)
            self.lat3 = nn.Conv2d(widths[2], widths[2], 1, bias=False)
        if mode == "hfb":
            # top-down 投影（深层->浅层，跨层级信息）：1x1 conv + 上采样对齐
            self.td2 = nn.Conv2d(widths[2], widths[1], 1, bias=False)  # s3 -> s2
            self.td1 = nn.Conv2d(widths[1], widths[0], 1, bias=False)  # s2 -> s1
            # N2a：可学习反馈缩放（初始化 α，给梯度「调大」通道）
            self.s2_scale = nn.Parameter(torch.tensor(float(fb_scale)))
            self.s1_scale = nn.Parameter(torch.tensor(float(fb_scale)))

    def forward(self, x, steps=None):
        T = self.steps if steps is None else steps
        x0 = self.stem(x)
        z1 = self.s1(x0)
        z2 = self.s2(z1)
        z3 = self.s3(z2)
        for t in range(T - 1):
            if self.mode == "rec":
                # 同层横向自连接（不跨层）：仅门控残差调制，无二次卷积（N1p 教训）
                z1 = z1 + torch.sigmoid(self.lat1(z1)) * z1
                z2 = z2 + torch.sigmoid(self.lat2(z2)) * z2
                z3 = z3 + torch.sigmoid(self.lat3(z3)) * z3
            elif self.mode == "hfb":
                # top-down 调制：深层投影回浅层，仅残差调制（无二次卷积）
                f2 = F.interpolate(self.td2(z3), size=z2.shape[-2:], mode="nearest")
                z2 = z2 + self.s2_scale * torch.tanh(f2)
                f1 = F.interpolate(self.td1(z2), size=z1.shape[-2:], mode="nearest")
                z1 = z1 + self.s1_scale * torch.tanh(f1)
        pooled = F.adaptive_avg_pool2d(z3, 1).flatten(1)
        return self.head(pooled)

    def param_count_module(self):
        """参数量对账：返回非前馈共享部分的参数量。"""
        shared = (sum(p.numel() for n, p in self.named_parameters()
                      if not n.startswith(("lat", "td"))))
        extra = sum(p.numel() for n, p in self.named_parameters()
                    if n.startswith(("lat", "td", "s1_scale", "s2_scale")))
        return shared, extra
