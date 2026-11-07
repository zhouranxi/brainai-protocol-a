#!/usr/bin/env python3
"""第四格 vs 第三格判据核算（ADR-0016 附录二十一 Q1/Q2 + 附加对照）。

预注册判据：
  Q1 排序一致：逐指标 Spearman ρ ≥ 0.95
  Q2 符号稳定：radius 对 c100 保持正号
  Q3 参考锚点：官方 example 上读数 = A 变体值（已在考古中确认）
附加：ID 受控偏相关（信息量混淆检查）、官方 PR/NC1 的对照符号。
"""
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

c3 = {r["ckpt"]: r for r in json.load(
    open("/mnt/mydisk/xinguan-brainai/runs/glue_own48_results.json"))["rows"]}
c4 = {r["ckpt"]: r for r in json.load(
    open("/mnt/mydisk/xinguan-brainai/runs/glue_cell4_own48.json"))["rows"]}
common = sorted(set(c3) & set(c4))
t = np.array([c3[k]["c100_probe_acc"] for k in common])
ida = np.array([c3[k]["id_acc"] for k in common])

print(f"n={len(common)}   OOD 目标：c100_probe_acc")
print(f'{"指标":24s} {"第三格(自宿主)":>16s} {"第四格(官方宿主)":>16s} {"符号一致":>8s}')
for key, label in [("glue_dimension", "dimension"),
                   ("glue_radius", "radius"),
                   ("glue_utility", "utility")]:
    a = np.array([c3[k][key] for k in common])
    b = np.array([c4[k][key] for k in common])
    p3 = pearsonr(a, t)[0]
    p4 = pearsonr(b, t)[0]
    s3 = spearmanr(a, t)[0]
    s4 = spearmanr(b, t)[0]
    agree = "Y" if p3 * p4 > 0 else "X FLIP"
    print(f"{label:24s} r={p3:+.3f}/rho={s3:+.3f}  r={p4:+.3f}/rho={s4:+.3f}  {agree}")

print(f'{"id_acc(基线)":24s} r={pearsonr(ida, t)[0]:+.3f}/rho={spearmanr(ida, t)[0]:+.3f}')

pr4 = np.array([c4[k]["official_pr"] for k in common])
print(f'{"official_pr(4格)":24s} r={pearsonr(pr4, t)[0]:+.3f}/rho={spearmanr(pr4, t)[0]:+.3f}')
nc4 = np.array([c4[k]["official_nc1"] for k in common])
print(f'{"official_nc1(4格)":24s} r={pearsonr(nc4, t)[0]:+.3f}/rho={spearmanr(nc4, t)[0]:+.3f}')


def partial(x, y, z):
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    rxy = np.corrcoef(x, y)[0, 1]
    num = rxy - rxz * ryz
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return num / den


print(f"\nID 受控偏相关（控制 id_acc）：")
for key, label, src in [
        ("glue_dimension", "dimension(3格)", c3), ("glue_dimension", "dimension(4格)", c4),
        ("glue_radius", "radius(3格)", c3), ("glue_radius", "radius(4格)", c4),
        ("glue_utility", "utility(3格)", c3), ("glue_utility", "utility(4格)", c4)]:
    v = np.array([src[k][key] for k in common])
    print(f"  {label:20s} partial r = {partial(v, t, ida):+.3f}")

# Q1 正式判定
print(f"\nQ1 判定（阈 0.95）：")
for key in ["glue_dimension", "glue_radius", "glue_utility"]:
    a = [c3[k][key] for k in common]
    b = [c4[k][key] for k in common]
    rho = spearmanr(a, b)[0]
    ok = "PASS" if rho >= 0.95 else "FAIL"
    print(f"  {key:16s} rho={rho:.4f}  {ok}")
print("DONE")
