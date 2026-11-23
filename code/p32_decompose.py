#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P32 (ADR-0032): R1 decomposition 2x2 probe.

Fills the two missing cells of the history x format grid:
  b3 = stateless + full format (M-c first-turn prompt, used standalone at each grade)
  b4 = stateless + single-answer (M-b first-turn prompt, standalone at each grade)
Existing arms M-c / M-b come from the P31 batch (same 60 puzzles, same
temperature 0.7 x 3 resampling, same v4 scorer).

Design frozen in paper/adr_0032.md.
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://localhost:8543/v1/chat/completions"
PUZZLES_PATH = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p31/p31_puzzles.json")
OUT = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p32/p32_decompose.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def ask(content, max_tokens=1200):
    payload = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE, data=payload,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"].get("content") or ""
        except Exception as e:
            if attempt == 2:
                return f"[ERROR {type(e).__name__}]"
            time.sleep(5)


# ---- v4 scorer: confidence-anchored gold rank + dup counter (same as P25v2/P31) ----
ITEM_RE = re.compile(r"^\s*(?:\d+[\.\、\)]|[-•*]|\(\d+\)|（\d+）)")
NUM_RE = re.compile(r"^\s*(\d+)[\.\、\)]")
PCT_RE = re.compile(r"\d+(\.\d+)?\s*%")


def _pct_near(lines, i):
    for j in (i, i + 1, i + 2):
        if j < len(lines) and PCT_RE.search(lines[j]):
            return True
    return False


def parse_items(text):
    """Return list of (line_idx, item_no, pct_value) for confidence-anchored items."""
    lines = text.splitlines()
    items = []
    no = 0
    for i, ln in enumerate(lines):
        if ITEM_RE.match(ln):
            m = NUM_RE.match(ln)
            no = int(m.group(1)) if m else no + 1
            if _pct_near(lines, i):
                pm = PCT_RE.search(ln) or PCT_RE.search(lines[i + 1] if i + 1 < len(lines) else "")
                pct = -1.0
                for j in (i, i + 1, i + 2):
                    if j < len(lines):
                        mm = PCT_RE.search(lines[j])
                        if mm:
                            try:
                                pct = float(mm.group(0).replace("%", "").strip())
                            except ValueError:
                                pass
                            break
                items.append((i, no, pct))
    return items


def gold_rank(text, gold):
    """Rank of the highest-confidence item mentioning gold. 0=absent, -1=error."""
    if text.startswith("[ERROR"):
        return -1
    items = parse_items(text)
    cand = [(pct, no) for (i, no, pct) in items if gold in text.splitlines()[i]]
    if cand:
        best = max(cand, key=lambda x: x[0])
        return min(max(best[1], 1), 3)
    return 0 if gold not in text else 1


def gold_dup(text, gold):
    """Count confidence-anchored items mentioning gold."""
    if text.startswith("[ERROR"):
        return 0
    lines = text.splitlines()
    items = parse_items(text)
    return sum(1 for (i, _, _) in items if gold in lines[i])


# ---- the two new arms: exact first-turn prompts of M-c / M-b, standalone ----

def run_arm(arm, puzzle, line):
    if arm == "b3":
        return ask(f"谜题：{puzzle['q']}\n{line}\n"
                   "列出至少三个候选假设，按把握排序，每个附把握百分比。"
                   "证据不足时禁止收敛到单一答案。")
    if arm == "b4":
        return ask(f"谜题：{puzzle['q']}\n{line}\n"
                   "直接给出你最有把握的一个答案，并简述理由。")
    raise ValueError(arm)


def main():
    puzzles = json.loads(PUZZLES_PATH.read_text(encoding="utf-8"))
    rows = []
    done = set()
    if OUT.exists():
        rows = json.loads(OUT.read_text(encoding="utf-8"))["rows"]
        done = {(r["puzzle"], r["arm"], r["grade"], r["rep"]) for r in rows}
    for pz in puzzles:
        for arm in ("b3", "b4"):
            for grade in (1, 2, 3):
                line = [pz["g1"], pz["g2"], pz["g3"]][grade - 1]
                for rep in (0, 1, 2):
                    key = (pz["id"], arm, grade, rep)
                    if key in done:
                        continue
                    t0 = time.time()
                    text = run_arm(arm, pz, line)
                    rank = gold_rank(text, pz["gold"])
                    dup = gold_dup(text, pz["gold"])
                    row = {"puzzle": pz["id"], "arm": arm, "grade": grade,
                           "rep": rep, "rank": rank, "dup": dup,
                           "secs": round(time.time() - t0, 1), "text": text}
                    rows.append(row)
                    OUT.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
                    print(f"[{pz['id']} {arm} G{grade} r{rep}] rank={rank} "
                          f"dup={dup} ({row['secs']}s)", flush=True)
    Path("/tmp/p32_done").write_text("done")
    print(f"written: {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
