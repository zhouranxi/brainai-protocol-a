#!/usr/bin/env python3
"""P18 batch-C chunk runner (steps>=7 pure bucket): budget seconds then exit."""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p18_hier_prompt import CACHE, CONDS, build_prompt, run_ollama, pred_num, gold_num

OUT = Path(__file__).resolve().parent.parent / "results" / "p18_llm_hier" / "p18C_results.jsonl"
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 480
BATCH_B = json.load(open(r"C:\Users\Lance\.cache\p18_batchB_ids.json"))
_all = [json.loads(l) for l in open(CACHE, encoding="utf-8")]
lines = [_all[i] for i in BATCH_B]

t_end = time.time() + BUDGET
done = set()
if OUT.exists():
    for line in open(OUT, encoding="utf-8"):
        r = json.loads(line)
        done.add((r["pid"], r["cond"]))
n = 0
for pid, item in enumerate(lines):
    gold = gold_num(item["answer"])
    for cond in CONDS:
        if (pid, cond) in done:
            continue
        if time.time() > t_end:
            print(f"[chunkC] budget out, wrote {n} rows", flush=True)
            sys.exit(0)
        try:
            text, ntok, secs = run_ollama(build_prompt(cond, item["question"]), timeout=240)
        except Exception as e:
            text, ntok, secs = f"__ERROR__ {type(e).__name__}", 0, 0.0
        row = {"pid": pid, "cond": cond, "gold": gold,
               "pred": pred_num(text), "correct": pred_num(text) == gold,
               "ntok": ntok, "secs": secs}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
print(f"[chunkC] ALL DONE, wrote {n}", flush=True)
