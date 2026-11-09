#!/usr/bin/env python3
"""把散落目录里的三种子剂量扫描收敛成一份自包含 JSON（发布/复现用）。

输入（实验机 /mnt/mydisk/xinguan-brainai/runs/v2_p4_forcing/）：
  full_base_lr{1.0,0.125}.json                  base 读数（3 种子 × 2 配方）
  full_s{0,1,2}_whiten_lam{0.1,1.0}/           锚点 λ=0.1/1.0（全量批）
  dose_{lo,hi}_lam{0.03,0.3,3.0}/              s0 剂量点
  dose_seed_s{1,2}_{lo,hi}_lam{0.03,0.3,3.0}/  s1/s2 新 run（附录二十五）

输出：BASE/dose_scan_seeds.json
  meta   生成信息、冻结常量、来源清单
  bases  每个 model tag 的 base 读数
  points 3 种子 × 2 配方 × 5 λ 的 Δd/Δc/id_keep 与原始读数、来源目录

设计原则：发布包只依赖这一份文件即可复跑判定与画图，不需要 GPU、
不需要实验机目录结构。数据本身逐字来自 train_forcing_p4.json，不做加工。
"""
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

RECIPES = {
    "lo": "resnet18_lr1.0_wd5e-05",
    "hi": "resnet18_lr0.125_wd0.0001",
}
SEEDS = [0, 1, 2]
LAMS = [0.03, 0.1, 0.3, 1.0, 3.0]
FROZEN = {
    "placebo_band": 0.022,
    "id_floor": -0.02,
    "b_replication_range": [0.3, 0.75],
    "b_median_tol": 0.15,
    "prereg": "ADR-0016 Appendix 25 (2026-11-08 19:44, frozen before launch)",
}
MEASURE_KEYS = ["d_ent_id", "c100_probe", "id_probe_acc", "test_acc", "secs"]


def read_rows(path):
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def candidate_dirs(base, recipe, seed, lam):
    """与 dose_shape_fit_seeds.py 的 load_point 完全一致的目录优先级。"""
    out = []
    if lam in (0.1, 1.0):
        out.append(base / f"full_s{seed}_whiten_lam{lam}")
    if seed == 0:
        out.append(base / f"dose_{recipe}_lam{lam}")
    else:
        out.append(base / f"dose_seed_s{seed}_{recipe}_lam{lam}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/mnt/mydisk/xinguan-brainai/runs/v2_p4_forcing",
                    help="实验机上 v2_p4_forcing 目录")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认 BASE 下）")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out) if args.out else base / "dose_scan_seeds.json"

    # ---- base 读数 ----
    bases = {}
    base_src = []
    for fp in sorted(base.glob("full_base_lr*.json")):
        base_src.append(fp.name + "/")
        for r in read_rows(fp):
            bases[r["model"]] = {
                "model": r["model"],
                **{k: r.get(k) for k in MEASURE_KEYS if k in r},
                "src": fp.name,
            }

    # ---- 30 个剂量点 ----
    points = []
    missing = []
    dirs_used = set()
    for recipe, stem in RECIPES.items():
        for seed in SEEDS:
            tag = f"{stem}_s{seed}"
            b = bases.get(tag)
            if b is None:
                missing.append(f"base {tag}")
                continue
            for lam in LAMS:
                found = None
                for d in candidate_dirs(base, recipe, seed, lam):
                    for r in read_rows(d / "train_forcing_p4.json"):
                        if (r["model"] == tag and r["kind"] == "whiten"
                                and abs(r["lambda"] - lam) < 1e-9):
                            found = (r, d.name)
                            break
                    if found:
                        break
                if found is None:
                    missing.append(f"{recipe} s{seed} λ={lam}")
                    continue
                r, dname = found
                dirs_used.add(dname + "/")
                dd = r["d_ent_id"] - b["d_ent_id"]
                dc = r["c100_probe"] - b["c100_probe"]
                ik = r["id_probe_acc"] - b["id_probe_acc"]
                points.append({
                    "recipe": recipe, "seed": seed, "lambda": lam,
                    "model": tag,
                    "d_delta": round(dd, 6),
                    "c_delta": round(dc, 6),
                    "id_keep": round(ik, 6),
                    "raw": {k: r.get(k) for k in MEASURE_KEYS if k in r},
                    "base": {k: b[k] for k in MEASURE_KEYS if k in b},
                    "src": dname + "/train_forcing_p4.json",
                })

    try:
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%h"], cwd=base.parent.parent,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = None

    doc = {
        "meta": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "generator": "code/build_dose_scan_seeds.py",
            "experiment_commit": commit,
            "frozen_constants": FROZEN,
            "n_bases": len(bases), "n_points": len(points),
            "expected_points": len(RECIPES) * len(SEEDS) * len(LAMS),
            "missing": missing,
            "source_dirs": sorted(dirs_used) + sorted(set(base_src)),
            "note": ("Point values are verbatim differences of raw readings; "
                     "no filtering, no band changes. Verdict is implemented in "
                     "code/dose_shape_fit_seeds.py (criteria frozen before launch)."),
        },
        "bases": bases,
        "points": points,
    }
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}: {len(bases)} bases, {len(points)} points, "
          f"missing={missing}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
