"""P4 全量批合并分析：base 读数 + forcing 结果并表，算逐配方斜率与判决统计。

输入：
  runs/v2_p4_forcing/full_base_lr*.json     —— base 读数（p4_base arm）
  runs/v2_p4_forcing/full_s*_*/train_forcing_p4.json —— 全量 forcing
  runs/v2_p4_forcing/{pilot,pilot_b2_*}/train_forcing_p4.json —— 已覆盖读数

输出：
  runs/v2_p4_forcing/full_merged.json       —— 合并表
  stdout：逐（条件,配方）斜率统计 + P4-2 判决核算
"""
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p4_forcing")
OUT = BASE_DIR / "full_merged.json"

S_OBS_D_ENT = 0.0161      # 观察斜率（d_ent 口径，E1 锚点）
LABEL_THR = 0.2 * S_OBS_D_ENT
KNOB_THR = 0.5 * S_OBS_D_ENT

# placebo 带宽：|Δc100| ≤ PLACEBO_BAND 视为 placebo 可解释。
# ⚠️ 论文判定口径（ADR 附录二十/二十二，P4-2 的 39 臂）用的是 ±0.022
# ——全量批 placebo 26 读数的 SD（+0.0008 ± 0.0220，附录二十 P4-4）。
# 0.033 是附录十九 pilot 时代的带宽（2 条 pilot placebo 读数的最大值），
# 保留为历史对照口径：BAND=0.033 会给出 34 臂（33 whiten + 1 disperse），
# 与论文数字 39/39、34/39 不一致——勿用本值复算论文判定。
PLACEBO_BAND = 0.033
PLACEBO_BAND_PAPER = 0.022  # 论文判定口径（附录二十）


def load_rows():
    rows = []
    for fp in sorted(BASE_DIR.glob("full_base_lr*.json")):
        rows.extend(json.load(open(fp))["rows"])
    for pat in ["pilot/train_forcing_p4.json", "pilot_b2_*/train_forcing_p4.json",
                "full_s*/train_forcing_p4.json"]:
        for fp in sorted(BASE_DIR.glob(pat)):
            # ⚠️ glob 的 * 会匹配 *.poisoned_v1 目录，必须显式排除作废读数
            if "poisoned" in str(fp):
                continue
            for r in json.load(open(fp))["rows"]:
                if r["arm"] != "p4_base":
                    r["_src"] = fp.parent.name
                    rows.append(r)
    return rows


def is_valid(r):
    """读数是否可用于判决核算。

    排除两类（附录十八/十九已定）：
      ① d_ent 为 NaN（v1 NaN 毒化的历史读数，或 forcing 后谱近似失败）；
      ② 功能破坏臂（id_keep < -0.05）：训练塌陷的 run 的 c100 无解释力。
    """
    de = r.get("d_ent_id")
    if de is None or not math.isfinite(de):
        return False
    ik = r.get("id_keep")
    if ik is not None and math.isfinite(ik) and ik < -0.05:
        return False
    return True


def main():
    rows = load_rows()
    base_by_tag = {r["model"]: r for r in rows if r["arm"] == "p4_base"}
    print(f"base 读数 {len(base_by_tag)} 个模型；forcing 读数 "
          f"{sum(1 for r in rows if r['arm'] != 'p4_base')} 条")

    merged = []
    for r in rows:
        if r["arm"] == "p4_base":
            continue
        b = base_by_tag.get(r["model"])
        m = dict(r)
        m["base_d_ent"] = b["d_ent_id"] if b else None
        m["base_c100"] = b["c100_probe"] if b else None
        m["base_glue_d"] = (b or {}).get("glue_d")
        m["d_delta"] = (r["d_ent_id"] - b["d_ent_id"]
                        if b and isinstance(r["d_ent_id"], float)
                        and isinstance(b["d_ent_id"], float) else None)
        m["c_delta"] = (r["c100_probe"] - b["c100_probe"] if b else None)
        m["slope"] = (m["c_delta"] / m["d_delta"]
                      if m["d_delta"] not in (None, 0) and m["c_delta"] is not None
                      else None)
        m["id_keep"] = (r["id_probe_acc"] - b["id_probe_acc"] if b else None)
        merged.append(m)

    json.dump({"rows": merged}, open(OUT, "w"), indent=1)
    print(f"merged -> {OUT}")

    # —— 逐条件汇总 ——
    by_cond = defaultdict(list)
    n_drop = 0
    for m in merged:
        if m["slope"] is None or not math.isfinite(m["slope"]):
            n_drop += 1
            continue
        if not is_valid(m):
            n_drop += 1
            continue
        by_cond[(m["kind"], m["lambda"])].append(m)
    print(f"\n剔除无效读数 {n_drop} 条（NaN d_ent / 功能破坏臂 / 无斜率）")

    print(f"\n{'条件':22s} {'n':>3s} {'同号':>4s} {'斜率中位':>9s} "
          f"{'%s_obs':>7s} {'in_band':>8s} {'id保持':>7s}")
    for (kind, lam), ms in sorted(by_cond.items()):
        sl = sorted(m["slope"] for m in ms)
        med = sl[len(sl) // 2]
        sgn = sum(1 for m in ms if m["slope"] * S_OBS_D_ENT > 0)
        inb = sum(1 for m in ms if abs(m["c_delta"]) <= PLACEBO_BAND)
        idmin = min(m["id_keep"] for m in ms)
        print(f"{kind}@{lam:<15g} {len(ms):3d} {sgn:3d}/{len(ms)} "
              f"{med:+9.5f} {100*med/S_OBS_D_ENT:6.1f}% "
              f"{inb:4d}/{len(ms):<4d} {idmin:+7.3f}")

    # P4-2 全量判决核算（D↑ 臂，排除 placebo 带内）
    # 双口径并行输出：PAPER 口径 ±0.022（附录二十，39 臂）与 pilot 旧口径 0.033（34 臂）
    for band, tag in ((PLACEBO_BAND_PAPER, "论文口径 ±0.022"),
                      (PLACEBO_BAND, "pilot 旧口径 ±0.033")):
        up = [m for (k, l), ms in by_cond.items() if k in ("whiten", "disperse")
              for m in ms if abs(m["c_delta"]) > band]
        if not up:
            continue
        sl = [m["slope"] for m in up]
        sgn = sum(1 for s in sl if s > 0)
        print(f"\nP4-2 核算（D↑ 臂 n={len(up)}，{tag}）：同号 {sgn}/{len(up)}，"
              f"斜率范围 [{min(sl):+.5f}, {max(sl):+.5f}]，"
              f"label 阈 {LABEL_THR:.5f}，knob 阈 {KNOB_THR:.5f}")
        n_label = sum(1 for s in sl if s <= LABEL_THR)
        n_knob = sum(1 for s in sl if s >= KNOB_THR)
        print(f"  ≤label 阈 {n_label}/{len(up)}，≥knob 阈 {n_knob}/{len(up)}")


if __name__ == "__main__":
    main()
