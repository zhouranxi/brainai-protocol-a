#!/usr/bin/env python3
"""剂量扫描种子复制批判定（ADR-0016 附录二十五预注册 R1-s/R2-s/R3-s/R4-s）。

数据布局：
  s0：dose_{lo,hi}_lam{0.03,0.3,3.0}/train_forcing_p4.json（附录二十三）
  s1/s2：dose_seed_s{1,2}_{lo,hi}_lam{0.03,0.3,3.0}/（附录二十五新 run，
         每 (种子,配方) 专属目录，单写者）
  锚点 λ=0.1/1.0：full_s{0,1,2}_whiten_lam{0.1,1.0}/（全量批已有）
  base 对照：full_base_lr{1.0,0.125}.json（含 s0/s1/s2 三种子行）

判定（跑前冻结于附录二十五，2026-11-08 19:44）：
  R1-s：hi 配方 s1/s2 幂律 b∈[0.3,0.75] 复制成功；
        跨种子中位数与 s0 差 |Δb|<0.15 强复制。
  R2-s：hi s1/s2 断崖形态——λ=0.03/0.3 带内(|Δc|≤0.022)且 λ=1.0/3.0 超带。
  R3-s：lo s1/s2 全 5 点 |Δc|≤0.022（无响应复制）。
  R4-s：全部 run id_keep ≥ −0.02；越界点剔除并单独报告。
失败如实报告，不得改带宽。
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

BASE = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p4_forcing")

RECIPES = {
    "lo": "resnet18_lr1.0_wd5e-05",
    "hi": "resnet18_lr0.125_wd0.0001",
}
DEFAULT_SEEDS = [0, 1, 2]   # 附录二十五批
ALL_SEEDS = [0, 1, 2, 3, 4] # 附录二十六批加入 s3/s4
LAMS = [0.03, 0.1, 0.3, 1.0, 3.0]
BAND = 0.022          # placebo 判定带宽（预注册口径，不得改）
ID_FLOOR = -0.02      # R4-s 功能保持下限
B_RANGE = (0.3, 0.75) # R1-s 复制成功区间
B_MEDIAN_TOL = 0.15   # R1-s 强复制容差


def load_bases():
    bases = {}
    for fp in sorted(BASE.glob("full_base_lr*.json")):
        for r in json.load(open(fp))["rows"]:
            bases[r["model"]] = r
    return bases


def load_point(recipe, seed, lam, bases):
    """返回 (Δd, Δc, id_keep) 或 None。锚点走全量批目录，其余走 dose 目录。"""
    stem = RECIPES[recipe]
    tag = f"{stem}_s{seed}"
    b = bases.get(tag)
    if b is None:
        return None
    cands = []
    if lam in (0.1, 1.0):
        cands.append(BASE / f"full_s{seed}_whiten_lam{lam}" / "train_forcing_p4.json")
    if seed == 0:
        cands.append(BASE / f"dose_{recipe}_lam{lam}" / "train_forcing_p4.json")
    else:
        cands.append(BASE / f"dose_seed_s{seed}_{recipe}_lam{lam}" / "train_forcing_p4.json")
    for fp in cands:
        if not fp.exists():
            continue
        for r in json.load(open(fp))["rows"]:
            if (r["model"] == tag and r["kind"] == "whiten"
                    and abs(r["lambda"] - lam) < 1e-9):
                return (r["d_ent_id"] - b["d_ent_id"],
                        r["c100_probe"] - b["c100_probe"],
                        r["id_probe_acc"] - b["id_probe_acc"])
    return None


def find_anomaly(pts):
    """附录二十四口径的异常点自动识别：|Δc|≤BAND（placebo 等价）
    且相邻局部斜率有负号的点。返回 {lam: (dd, dc, ik)}。"""
    lams = sorted(pts)
    anom = set()
    for i, l in enumerate(lams):
        if abs(pts[l][1]) > BAND:
            continue
        slopes = []
        if i > 0:
            d0, c0 = pts[lams[i - 1]][0], pts[lams[i - 1]][1]
            slopes.append((pts[l][1] - c0) / (pts[l][0] - d0))
        if i < len(lams) - 1:
            d1, c1 = pts[lams[i + 1]][0], pts[lams[i + 1]][1]
            slopes.append((c1 - pts[l][1]) / (d1 - pts[l][0]))
        if any(s < 0 for s in slopes):
            anom.add(l)
    return {l: p for l, p in pts.items() if l not in anom}


def fit_power(ds, cs):
    """幂律 Δc = a·Δd^b，拟合 Δc>0 且 Δd>0 的点（log-log 线性回归）。"""
    pts = [(d, c) for d, c in zip(ds, cs) if c > 0 and d > 0]
    if len(pts) < 3:
        return None, len(pts)
    lx = np.log([p[0] for p in pts])
    ly = np.log([p[1] for p in pts])
    b, loga = np.polyfit(lx, ly, 1)
    return (math.exp(loga), float(b)), len(pts)


def load_curves_from_archive(path):
    """从 build_dose_scan_seeds.py 产出的自包含 JSON 载入曲线（发布包复现用）。"""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    curves = {(r, s): {} for r in RECIPES for s in ALL_SEEDS}
    viol = []
    for p in doc["points"]:
        key = (p["recipe"], p["seed"])
        if key not in curves:
            continue
        curves[key][p["lambda"]] = (p["d_delta"], p["c_delta"], p["id_keep"])
        if p["id_keep"] < ID_FLOOR:
            viol.append((p["recipe"], p["seed"], p["lambda"], p["id_keep"]))
    if doc["meta"]["missing"]:
        print(f"WARNING: archive reports missing points: {doc['meta']['missing']}",
              file=sys.stderr)
    return curves, viol


def load_curves_from_dirs():
    """实验机原始目录布局（默认模式）。"""
    bases = load_bases()
    curves = {}
    viol = []
    for recipe in RECIPES:
        for seed in SEEDS:
            pts = {}
            for lam in LAMS:
                p = load_point(recipe, seed, lam, bases)
                if p is None:
                    continue
                pts[lam] = p
                if p[2] < ID_FLOOR:
                    viol.append((recipe, seed, lam, p[2]))
            curves[(recipe, seed)] = pts
    return curves, viol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None,
                    help="自包含归档 dose_scan_seeds.json；给定时不依赖实验机目录")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="判定所用种子集（默认随数据自适应：3 种子批 [0,1,2]；"
                         "含 s3/s4 的 v2 归档用 [0,1,2,3,4]）")
    args = ap.parse_args()

    global SEEDS
    if args.data:
        curves, func_viol = load_curves_from_archive(args.data)
        print(f"[data] archive: {args.data}")
    else:
        curves, func_viol = load_curves_from_dirs()
        print(f"[data] experiment dirs: {BASE}")
    if args.seeds:
        SEEDS = sorted(args.seeds)
    else:
        have = {s for (_r, s) in curves.keys() if curves[(_r, s)]}
        SEEDS = sorted(have) if (3 in have or 4 in have) else sorted(
            s for s in DEFAULT_SEEDS if s in have)
    for r in RECIPES:                       # guarantee keys exist
        for s in SEEDS:
            curves.setdefault((r, s), {})

    # ---- 数据总表 ----
    for recipe in RECIPES:
        print(f"\n=== {recipe}（{RECIPES[recipe]}）===")
        header = f"{'seed':>4s} " + " ".join(f"{'λ='+format(l,'g'):>16s}" for l in LAMS)
        print(header)
        for seed in SEEDS:
            pts = curves[(recipe, seed)]
            cells = []
            for lam in LAMS:
                if lam in pts:
                    dd, dc, ik = pts[lam]
                    cells.append(f"{dd:+7.1f}/{dc:+7.4f}")
                else:
                    cells.append(" " * 16)
            print(f"s{seed:<3d} " + " ".join(f"{c:>16s}" for c in cells))

    # ---- R4-s 功能保持 ----
    print(f"\n[R4-s] 功能保持（id_keep ≥ {ID_FLOOR:+.2f}）：")
    if func_viol:
        for recipe, seed, lam, ik in func_viol:
            print(f"  VIOLATION {recipe} s{seed} λ={lam} id_keep={ik:+.4f}")
        print("  → 越界点从形态拟合剔除（上方总表仍显示）")
    else:
        print("  全部通过 ✓")

    # ---- R3-s：lo 无响应复制 ----
    print(f"\n[R3-s] lo 无响应复制（全部点 |Δc|≤{BAND}）：")
    for seed in SEEDS:
        pts = curves[("lo", seed)]
        if len(pts) < 5:
            print(f"  s{seed}: 点数不足（{len(pts)}/5），无法判定")
            continue
        out_band = [(l, p[1]) for l, p in pts.items() if abs(p[1]) > BAND]
        if out_band:
            detail = ", ".join(f"λ={l}: {c:+.4f}" for l, c in sorted(out_band))
            print(f"  s{seed}: FAIL — 超带点 {detail}")
        else:
            print(f"  s{seed}: PASS（5/5 带内）")

    # ---- R2-s：hi 断崖形态 ----
    print(f"\n[R2-s] hi 断崖形态（λ=0.03/0.3 带内 且 λ=1.0/3.0 超带）：")
    for seed in SEEDS:
        pts = curves[("hi", seed)]
        need = (0.03, 0.3, 1.0, 3.0)
        if any(l not in pts for l in need):
            print(f"  s{seed}: 点数不足，无法判定")
            continue
        low_in = all(abs(pts[l][1]) <= BAND for l in (0.03, 0.3))
        high_out = all(abs(pts[l][1]) > BAND for l in (1.0, 3.0))
        verdict = "PASS" if (low_in and high_out) else "FAIL"
        detail = ", ".join(f"λ={l}: {pts[l][1]:+.4f}" for l in need)
        print(f"  s{seed}: {verdict} — {detail}")

    # ---- R1-s：幂律 b 复制 ----
    print(f"\n[R1-s] hi 幂律指数 b（附录二十四口径：自动剔除带内+负斜率异常点；"
          f"R4-s 越界点亦剔除）：")
    viol_keys = {(r, s, l) for r, s, l, _ in func_viol}
    bs = {}
    for seed in SEEDS:
        pts = curves[("hi", seed)]
        pts_f = {l: p for l, p in pts.items()
                 if ("hi", seed, l) not in viol_keys}
        clean = find_anomaly(pts_f)
        dropped = sorted(set(pts_f) - set(clean))
        if dropped:
            print(f"  s{seed}: 剔除异常点 λ={', '.join(format(d, 'g') for d in dropped)}"
                  f"（带内+负局部斜率）")
        if len(clean) < 3:
            print(f"  s{seed}: 有效点不足（{len(clean)}/5），跳过拟合")
            continue
        lams_sorted = sorted(clean)
        ds = [clean[l][0] for l in lams_sorted]
        cs = [clean[l][1] for l in lams_sorted]
        fit, npos = fit_power(ds, cs)
        if fit:
            bs[seed] = fit[1]
            in_range = B_RANGE[0] <= fit[1] <= B_RANGE[1]
            print(f"  s{seed}: b={fit[1]:.3f}（a={fit[0]:.4f}，"
                  f"n={npos}）{'∈' if in_range else '∉'}[{B_RANGE[0]},{B_RANGE[1]}]")
        else:
            print(f"  s{seed}: 可拟合点不足（{npos} 个 Δc>0）")

    if 0 in bs and 1 in bs and 2 in bs:
        med = float(np.median(list(bs.values())))
        delta = abs(med - bs[0])
        copy = all(B_RANGE[0] <= b <= B_RANGE[1] for b in bs.values())
        strong = delta < B_MEDIAN_TOL
        print(f"  跨种子: b_s0={bs[0]:.3f} 中位={med:.3f} |Δb|={delta:.3f}"
              f"（阈值 {B_MEDIAN_TOL}）")
        print(f"  判定: 复制{'成功 ✓' if copy else '失败 ✗'}，"
              f"强复制{'✓' if strong else '✗'}")
    else:
        print("  三种子 b 不齐，复制判定无法完成（如实报告）")

    # ---- 附录二十六批：R5-s/R6-s/R7-s/R8-s（s3/s4 在场时才判定）----
    S_OBS = 0.0161
    LABEL_PCT = 20.0    # % of s_obs，与 96 批预注册标签阈值同口径
    CAP_PCT = 60.0      # R6-s 上限（96 批最大臂斜率 54.5% + 容差）
    if any(s in SEEDS for s in (3, 4)):
        print(f"\n[R5-s] hi 低剂量平台复制（lowplateau(s)=λ=0.03 割线，"
              f"≥{LABEL_PCT:.0f}% s_obs 的种子数）：" )
        lp = {}
        for seed in SEEDS:
            pts = curves.get(("hi", seed), {})
            if 0.03 in pts:
                dd, dc, _ = pts[0.03]
                if dd > 0:
                    lp[seed] = dc / dd / S_OBS * 100
                    print(f"  s{seed}: {lp[seed]:.1f}% s_obs")
                else:
                    print(f"  s{seed}: Δd≤0，无割线")
            else:
                print(f"  s{seed}: 缺 λ=0.03")
        n_ge = sum(1 for v in lp.values() if v >= LABEL_PCT)
        n_lo = sum(1 for v in lp.values() if v < LABEL_PCT)
        if n_ge >= 3:
            print(f"  判定: {n_ge} 种子 ≥20% → 低剂量高平台为跨种子事实 ✓")
        elif n_lo >= 3:
            print(f"  判定: {n_lo} 种子 <20% → s1/s2 高平台为种子特异 ✗")
        else:
            print(f"  判定: 混合（{n_ge}≥20% / {n_lo}<20%）→ 不稳定，如实报告")

        print(f"\n[R6-s] 响应幅度上限（全部种子剂量点割线 < {CAP_PCT:.0f}% s_obs；"
              f"100% 为硬红线）：")
        worst = None
        for seed in SEEDS:
            for recipe in RECIPES:
                for lam, (dd, dc, _ik) in sorted(
                        curves.get((recipe, seed), {}).items()):
                    if dd > 0:
                        pct = dc / dd / S_OBS * 100
                        if worst is None or pct > worst[0]:
                            worst = (pct, recipe, seed, lam)
        if worst:
            pct, recipe, seed, lam = worst
            hard = pct >= 100.0
            ok = pct < CAP_PCT
            print(f"  全库最大割线: {pct:.1f}% s_obs"
                  f"（{recipe} s{seed} λ={lam:g}）")
            print(f"  判定: {'≥60%，追加披露并重估 far-below 表述 ✗' if not ok else '<60%，干预斜率上界在种子间成立 ✓'}"
                  f"{'；⚠️ 硬红线 ≥100% 触发，须升级处理' if hard else ''}")

        print(f"\n[R7-s] lo 无响应（s3/s4 六点 |Δc|≤{BAND}）：")
        for seed in (3, 4):
            if seed not in SEEDS:
                continue
            pts = curves.get(("lo", seed), {})
            if len(pts) < 3:
                print(f"  s{seed}: 点数不足（{len(pts)}/3）")
                continue
            out_band = [(l, p[1]) for l, p in pts.items() if abs(p[1]) > BAND]
            if out_band:
                detail = ", ".join(f"λ={l}: {c:+.4f}" for l, c in sorted(out_band))
                print(f"  s{seed}: FAIL — 超带点 {detail}")
            else:
                print(f"  s{seed}: PASS（{len(pts)}/{len(pts)} 带内）")

        print(f"\n[R8-s] 功能保持（s3/s4 十二点 id_keep ≥ {ID_FLOOR:+.2f}）：")
        nviol = 0
        for seed in (3, 4):
            if seed not in SEEDS:
                continue
            for recipe in RECIPES:
                for lam, (dd, dc, ik) in sorted(
                        curves.get((recipe, seed), {}).items()):
                    if ik < ID_FLOOR:
                        nviol += 1
                        print(f"  VIOLATION {recipe} s{seed} λ={lam} "
                              f"id_keep={ik:+.4f}")
        print(f"  判定: {'全部保持 ✓' if nviol == 0 else f'{nviol} 点越界（剔除并单列）'}")

    print("\n（判定标准冻结于附录二十五/二十六；本脚本仅忠实实现，"
          "未看过数据后调整。）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
