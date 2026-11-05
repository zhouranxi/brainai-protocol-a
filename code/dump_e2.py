import json
from collections import Counter

for a in ("e2_r34", "e2_r50", "e2_sf"):
    try:
        rows = json.load(open(f"/mnt/mydisk/xinguan-brainai/runs/v2/{a}/intervene_v2.json"))["rows"]
        print(a, dict(Counter(r["model"] for r in rows)))
    except Exception:
        print(a, "no data")
