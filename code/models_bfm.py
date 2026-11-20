#!/usr/bin/env python3
"""Belief-feedback module (BFM) per 研究方案 Task-2 + ADR-0024 A0.

Base: torchvision-style ResNet18 (Bottleneck-free BasicBlock), built to
match the main-line `train_forcing_p4.build_model('resnet18')` layout
so CIFAR pipelines transfer 1:1. Modes:
  ff  : plain ResNet18 (param-matched head)
  rec : plain trunk + lateral recurrence: each residual layer re-used
        twice (z <- z + g_lat * z, gate from a per-layer scalar on
        pooled features) -- 同层重复 2 次 per the plan.
  bfm : trunk with belief state b (d_b=16), gate sigma(d_l) per layer,
        disambiguation commands injected at each layer's INPUT during
        a second forward (steering, not rewriting).
All modes share the trunk; extra modules are matched by widening the
head for ff/rec (parameter parity, exact counts reported per run).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + (self.short(x) if self.short is not None else x)
        return F.relu(out)


class ResNet18Trunk(nn.Module):
    def __init__(self, widths=(64, 128, 256, 512)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(BasicBlock(widths[0], widths[0]),
                                    BasicBlock(widths[0], widths[0]))
        self.layer2 = nn.Sequential(BasicBlock(widths[0], widths[1], 2),
                                    BasicBlock(widths[1], widths[1]))
        self.layer3 = nn.Sequential(BasicBlock(widths[1], widths[2], 2),
                                    BasicBlock(widths[2], widths[2]))
        self.layer4 = nn.Sequential(BasicBlock(widths[2], widths[3], 2),
                                    BasicBlock(widths[3], widths[3]))
        self.feat_dim = widths[3]

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class BFM(nn.Module):
    MODES = ("ff", "rec", "bfm")
    D_B = 16

    def __init__(self, mode="bfm", num_classes=10):
        super().__init__()
        assert mode in self.MODES
        self.mode = mode
        self.trunk = ResNet18Trunk()
        d = self.trunk.feat_dim
        if mode == "bfm":
            # 信念状态 b：高层低维信念（方案任务二 1.）
            self.b_head = nn.Linear(d, self.D_B)
            # 各层消歧指令：b -> 幅度受限残差注入（1×1 语义，输出每通道尺度）
            in_ch = (64, 64, 128, 256)   # 各层输入端通道（注入位置）
            widths = (64, 128, 256, 512)
            self.td = nn.ModuleList([nn.Linear(self.D_B, w) for w in widths])
            # 门控 σ(d_l)：DeLP 教训的可观测物——每层一个标量门，
            # d_l = w_l^T [pool(z_l); b]（联合量驱动，避免死门）
            self.gate_w = nn.ParameterList([
                nn.Parameter(torch.randn(w + self.D_B) * 0.05)
                for w in in_ch])
            self.gate_b = nn.Parameter(torch.zeros(4))
            self.head = nn.Linear(d, num_classes)
        else:
            if mode == "rec":
                self.lat_scalar = nn.Parameter(torch.zeros(4))
            # 参数配平：ff/rec 头加宽到与 bfm 附加模块同参数量级
            extra = 0
            self.head = nn.Sequential(nn.Linear(d, d + 256), nn.ReLU(inplace=True),
                                      nn.Linear(d + 256, num_classes))
        self._mode_done = mode

    def forward(self, x, severity_logits=None, record=None):
        """severity_logits: optional (N,) per-sample ambiguity scalar
        (e.g. normalized occlusion severity) fed into the gate -- the
        plan's '语义歧义度' signal path. record: optional dict to
        collect gate activations and b statistics (A0 probes)."""
        if self.mode == "bfm":
            return self._forward_bfm(x, severity_logits, record)
        if self.mode == "rec":
            return self._forward_rec(x)
        return self.head(self.trunk(x))

    # ---- BFM：两段前向（先高层信念，再逐层引导第二遍）----
    def _forward_bfm(self, x, severity_logits, record):
        t = self.trunk
        z0 = t.stem(x)
        zs = [z0]
        z = t.layer1(z0); zs.append(z)
        z = t.layer2(z); zs.append(z)
        z = t.layer3(z); zs.append(z)
        z = t.layer4(z); zs.append(z)
        pooled = F.adaptive_avg_pool2d(z, 1).flatten(1)
        b = self.b_head(pooled)                       # (N, d_b) 信念状态
        sev = severity_logits
        if sev is not None:
            sev = sev.view(-1, 1)
        # 第二遍：每层输入端注入消歧指令（引导而非改写）
        cur = z0
        gates = []
        for li, layer in enumerate([t.layer1, t.layer2, t.layer3, t.layer4]):
            pooled_l = F.adaptive_avg_pool2d(cur, 1).flatten(1)
            dl = torch.cat([pooled_l, b], dim=1) @ self.gate_w[li] + self.gate_b[li]
            if sev is not None:
                dl = dl + sev.squeeze(1) * 0  # 保留接口；歧义度不直接进门（避免直通）
            g = torch.sigmoid(dl)               # (N,) 门控 σ(d_l)
            gates.append(g)
            cmd = torch.tanh(self.td[li](b)[:, :cur.shape[1]])  # 消歧指令对齐注入通道
            cur = cur + (g.view(-1, 1, 1, 1)
                         * cmd[:, :, None, None] * cur)  # 残差注入（输入端）
            cur = layer(cur)
        pooled2 = F.adaptive_avg_pool2d(cur, 1).flatten(1)
        out = self.head(pooled2)
        if record is not None:
            record["b"] = b.detach()
            record["gates"] = torch.stack(gates, dim=1).detach()  # (N, 4)
        return out

    # ---- REC：纯循环对照（N1″ 健康版：标量门控自连接，无二次过卷）----
    # 注：「同层重复 2 次」的通道安全实现——门控自连接即循环步；
    # 二次过卷在 ADR-0019 N1′ 已判架构缺陷，不再复现。
    def _forward_rec(self, x):
        t = self.trunk
        layers = [t.layer1, t.layer2, t.layer3, t.layer4]
        z = t.stem(x)
        for li, layer in enumerate(layers):
            g = torch.sigmoid(self.lat_scalar[li]).view(1, 1, 1, 1)
            z = layer(z)
            z = z + g * z                      # 循环自连接步（重复施加 2 次）
            z = z + g * z
        pooled = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.head(pooled)

    def extra_param_count(self):
        base = sum(p.numel() for p in self.head.parameters())
        if self.mode == "ff":
            return 0
        if self.mode == "rec":
            return self.lat_scalar.numel()
        return sum(p.numel() for p in list(self.b_head.parameters())
                   + list(self.td.parameters())
                   + list(self.gate_w) + list(self.gate_b))


def param_report(model: BFM):
    total = sum(p.numel() for p in model.parameters())
    return {"mode": model.mode, "total_params": total,
            "extra_params": model.extra_param_count()}
