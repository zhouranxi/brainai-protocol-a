#!/usr/bin/env python3
"""P19 dose axis (F-b) on V100 box: 8 forcing ckpts x 3 smoothness metrics.

Runs under BRAINAI_DATA_ROOT=/home/lance/p19/data, checkpoints under
/home/lance/p19/runs/v2_p4_forcing/{p8_s0_seed2,p10_s0_seed3}_lam{0.03,0.1,1.0,3.0}.
Resume-safe: skips tags already in p19_dose.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import p19_smoothness as p19  # noqa: E402

RUNS = Path("/home/lance/p19/runs/v2_p4_forcing")
OUT = Path("/home/lance/p19/runs/v2_p19/p19_dose.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Resume-safe: load existing rows
rows = []
done = set()
if OUT.exists():
    rows = json.loads(OUT.read_text())["rows"]
    done = {r["tag"] for r in rows}

missing = []
for prefix in ("p8_s0_seed2", "p10_s0_seed3"):
    for lam in ("0.03", "0.1", "1.0", "3.0"):
        d = RUNS / f"{prefix}_lam{lam}"
        cks = sorted(d.glob("*whiten*.pt"))
        tag = f"{prefix}_lam{lam}"
        if tag in done:
            print(f"skip {tag} (already done)", flush=True)
            continue
        if not cks:
            missing.append(tag)
            print(f"MISSING {tag}", flush=True)
            continue
        m = p19.load_model(cks[-1])
        F, Y = p19.extract_features(m)
        r = p19.smoothness_metrics(F, Y)
        rows.append({"tag": tag, "family": "dose", **r})
        del m
        torch.cuda.empty_cache()
        OUT.write_text(json.dumps({"rows": rows}, indent=1))
        print(f"{tag} {r}", flush=True)

OUT.write_text(json.dumps({"rows": rows}, indent=1))
print(f"written: {OUT} ({len(rows)} rows, missing={missing})", flush=True)
