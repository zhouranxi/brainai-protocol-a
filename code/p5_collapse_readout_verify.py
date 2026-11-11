# -*- coding: utf-8 -*-
"""P5 收判独立复算（附录三十三冻结判读）。
口径：
- base 表：results/p4_full/full_base_lr0.125.json（lr0.125 wd1e-4 五种子）
- 塌陷判定：dc = c100_probe(run) - c100_probe(base) <= 0.03
- id_keep 闸：id_probe_acc(run) - id_probe_acc(base) >= -0.02
- secant_pct（仅 C1）：round(100 * (dc/dd) / 0.0161, 1)，s_obs=0.0161 冻结常数
- C1 判读（冻结）：>=6/8 seed-conditioned；<=2/8 transmission noise；3-5/8 mixed
与 results/p5_collapse/p5_readout.json（中断前脚本输出）逐项对照。
"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"c:\Users\Lance\WorkBuddy\2026-10-25-21-15-17\github-repo"
S_OBS = 0.0161
LINE = 0.03
ID_FLOOR = -0.02

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

base_tbl = load(ROOT + r"\results\p4_full\full_base_lr0.125.json")
bases = {r["model"]: r for r in base_tbl["rows"] if r["model"].startswith("resnet18_lr0.125_wd0.0001_")}

prior = load(ROOT + r"\results\p5_collapse\p5_readout.json")
prior_rows = {(r["arm"], r["run"]): r for r in prior["rows"]}

out, mismatch = [], 0
def one(arm, run, path, model):
    global mismatch
    row = load(ROOT + path)["rows"][0]
    b = bases[model]
    dd = round(row["d_ent_id"] - b["d_ent_id"], 3)
    dc = round(row["c100_probe"] - b["c100_probe"], 4)
    ik = round(row["id_probe_acc"] - b["id_probe_acc"], 4)
    col = dc <= LINE
    sc = round(100 * (dc / dd) / S_OBS, 1) if arm == "C1" else None
    p = prior_rows.get((arm, run))
    ok = p and abs(p["d_delta"]-dd) < 5e-4 and abs(p["c_delta"]-dc) < 5e-5 \
         and abs(p["id_keep"]-ik) < 5e-5 and p["collapse"] == col \
         and ((sc is None and p["secant_pct"] is None) or abs(p["secant_pct"]-sc) < 0.05)
    if not ok:
        mismatch += 1
    out.append(dict(arm=arm, run=run, d_delta=dd, c_delta=dc, collapse=col,
                    id_keep=ik, secant_pct=sc, match_prior=bool(ok)))

for r in range(1, 9):
    one("C1", f"rerun_seed{r}", rf"\results\p5_collapse\collapse_c1_seed{r}.json",
        f"resnet18_lr0.125_wd0.0001_s0")
for s in range(5):
    one("C2", f"s{s}_lam0.2", rf"\results\p5_collapse\collapse_c2_s{s}_lam0.2.json",
        f"resnet18_lr0.125_wd0.0001_s{s}")

c1 = [r for r in out if r["arm"] == "C1"]
c2 = [r for r in out if r["arm"] == "C2"]
n1 = sum(r["collapse"] for r in c1)
n2 = sum(r["collapse"] for r in c2)
id_min = min(r["id_keep"] for r in out)
verdict = ("seed_conditioned_fragility" if n1 >= 6
           else "transmission_noise" if n1 <= 2 else "mixed_as_prereg")

print(f"{'arm':4} {'run':12} {'dd':>8} {'dc':>7} {'collapse':8} {'id_keep':>8} {'sec%':>5} match")
for r in out:
    print(f"{r['arm']:4} {r['run']:12} {r['d_delta']:8.3f} {r['c_delta']:7.4f} "
          f"{str(r['collapse']):8} {r['id_keep']:8.4f} "
          f"{('' if r['secant_pct'] is None else format(r['secant_pct'],'.1f')):>5} {r['match_prior']}")
print(f"\nC1 collapses: {n1}/8 | C2 collapses: {n2}/5 | id_keep min: {id_min:.4f} (floor {ID_FLOOR})")
print(f"C1 frozen verdict: {verdict}")
print(f"prior-file mismatch count: {mismatch}")
json.dump(dict(rows=out, c1_n_collapse=n1, c2_n_collapse=n2, id_keep_min=id_min,
               verdict=verdict, s_obs=S_OBS, collapse_line=LINE, id_floor=ID_FLOOR),
          open(ROOT + r"\results\p5_collapse\p5_readout_verify.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
