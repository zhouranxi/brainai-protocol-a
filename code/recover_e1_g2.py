#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 e1_g2.log 恢复被 g3 覆盖的前 4 模型 76 行数据。

事故：g2/g3 两进程共用 runs/v2/e1/intervene_v2.json，各自内存缓存 rows，
g3 后写盘覆盖了 g2 的行。日志逐条件打印了全部字段，可无损恢复
（glue_r/glue_u/c10c 除外——日志没打，标 null）。
"""
import json
import re
import sys
from pathlib import Path

LOG = Path("/mnt/mydisk/xinguan-brainai/runs/v2/e1_g2.log")
OUT = Path("/mnt/mydisk/xinguan-brainai/runs/v2/e1/intervene_v2.json")

MODELS = [
    "resnet18_lr0.125_wd2.5e-05_s1",
    "resnet18_lr1.0_wd2.5e-05_s0",
    "resnet18_lr1.0_wd5e-05_s0",
    "resnet18_lr0.125_wd0.0001_s0",
]
# 与 CONDITIONS 顺序一致
CONDS = ([("base", None)]
         + [("gauss_iso", a) for a in (0.5, 1.0, 2.0)]
         + [("gauss_aniso", a) for a in (0.5, 1.0, 2.0)]
         + [("rotate", None)]
         + [("scale", s) for s in (0.5, 2.0)]
         + [("recolor", g) for g in (1.5, 0.75, 0.5, 0.25, 0.0)]
         + [("tot_recolor", g) for g in (0.5, 0.0)]
         + [("placebo", a) for a in (1.0, 2.0)])

pat = re.compile(
    r"^\s+(\w+)(?:=([\d.]+|None))?: D=([\d.]+|None) idp=([\d.]+) "
    r"c100=([\d.]+) to=(\w+) \(", )


def parse():
    rows = []
    mi = -1
    ci = 0
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\[(\d)/4\] (\S+) arch=(\S+) feats", line)
        if m:
            mi = int(m.group(1)) - 1
            ci = 0
            continue
        m = pat.match(line)
        if m and 0 <= mi < 4:
            kind, param, D, idp, c100, to = m.groups()
            kind_e, param_e = CONDS[ci]
            assert kind_e == kind, f"cond mismatch {ci}: {kind} vs {kind_e}"
            rows.append({
                "arm": "eval", "model": MODELS[mi], "arch": "resnet18",
                "kind": kind,
                "param": (float(param) if param not in (None, "None")
                          else None),
                "glue_d": (float(D) if D != "None" else None),
                "glue_r": None, "glue_u": None,   # 日志未打印
                "glue_timeout": to == "True",
                "id_fc_acc": None,                # 日志未打印
                "id_probe_acc": float(idp),
                "c100_probe": float(c100),
                "c10c_acc": None,                 # 日志未打印
                "recovered_from": "e1_g2.log",
            })
            ci += 1
    return rows


def main():
    rec = parse()
    print(f"recovered {len(rec)} rows")
    if len(rec) != 76:
        print("FATAL: expected 76 rows", file=sys.stderr)
        return 1
    d = json.load(open(OUT))
    have = {(r["model"], r["kind"], repr(r.get("param"))) for r in d["rows"]}
    added = 0
    for r in rec:
        k = (r["model"], r["kind"], repr(r["param"]))
        if k not in have:
            d["rows"].append(r)
            added += 1
    d["rows"].sort(key=lambda r: r["model"])
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"merged: +{added} rows, total {len(d['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
