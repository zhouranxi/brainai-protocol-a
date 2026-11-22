#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P25 v2 (ADR-0030): fixes two measurement artifacts in v1.
1) max_tokens 400->1200: M-c/M-d long-analysis answers were truncated,
   final candidate list often lost.
2) gold_rank rewritten: score position in the FINAL candidate list
   (last list-item occurrence), not first textual mention.
3) store full text (was 120 chars) for auditability.
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/code")
OUT = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p25/p25_race_v2.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8543/v1/chat/completions"
PUZZLES_PATH = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p25/p25_puzzles.json")


def ask(messages, max_tokens=1200):
    payload = json.dumps({
        "model": "qwen",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
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


ITEM_RE = re.compile(r"^\s*(?:\d+[\.、)]|[-•*]|\(\d+\)|（\d+）)")
NUM_RE = re.compile(r"^\s*(\d+)[\.、)]")
PCT_RE = re.compile(r"\d+(\.\d+)?\s*%")


def _parse_items(lines):
    """Candidate entries = list-item lines whose own text or the next two
    lines carry a confidence percentage. Returns [(line_idx, item_no, pct)]."""
    items = []
    last_no = 0
    for i, ln in enumerate(lines):
        if not ITEM_RE.match(ln):
            continue
        m = NUM_RE.match(ln)
        last_no = int(m.group(1)) if m else last_no + 1
        pct = None
        for j in (i, i + 1, i + 2):
            if j < len(lines):
                mm = PCT_RE.search(lines[j])
                if mm:
                    pct = float(mm.group(0).rstrip("%"))
                    break
        if pct is not None:
            items.append((i, last_no, pct))
    return items


def gold_rank_v2(text, gold):
    """Rank of gold = item_no of the HIGHEST-confidence candidate entry
    mentioning gold. Degenerate cases: absent->0, error->-1,
    mentioned without any confidence-anchored list->1."""
    if text.startswith("[ERROR"):
        return -1
    lines = text.splitlines()
    items = _parse_items(lines)
    cand = [(pct, no) for (i, no, pct) in items if gold in lines[i]]
    if cand:
        best = max(cand)  # highest confidence entry mentioning gold
        return min(best[1], 3)
    return 0 if gold not in text else 1


def gold_dup_count(text, gold):
    """Number of candidate entries mentioning gold (placeholder-padding
    detector: gold repeated across real + filler slots)."""
    if text.startswith("[ERROR"):
        return 0
    lines = text.splitlines()
    return sum(1 for (i, _, _) in _parse_items(lines) if gold in lines[i])


def run_mechanism(mech, puzzle, grade_lines):
    if mech == "a":  # stateless: current grade only
        return ask([{"role": "user", "content":
                     f"谜题：{puzzle['q']}\n{grade_lines[-1]}\n"
                     "列出前三个候选假设，按可能性排序。"}])
    if mech == "b":  # greedy lock: commit at g1, only fine-tune later
        if len(grade_lines) == 1:
            return ask([{"role": "user", "content":
                         f"谜题：{puzzle['q']}\n{grade_lines[0]}\n"
                         "直接给出你最有把握的一个答案，并简述理由。"}])
        return ask([
            {"role": "user", "content":
             f"谜题：{puzzle['q']}\n{grade_lines[0]}\n直接给出你最有把握的一个答案。"},
            {"role": "assistant", "content": "（已给出答案）"},
            {"role": "user", "content": grade_lines[-1] +
             "\n新线索出现了。你可以在原答案基础上微调（或维持）。"}])
    if mech == "c":  # maintain-update
        msgs = [{"role": "user", "content":
                 f"谜题：{puzzle['q']}\n{grade_lines[0]}\n"
                 "列出至少三个候选假设，按把握排序，每个附把握百分比。"
                 "证据不足时禁止收敛到单一答案。"}]
        for g, line in enumerate(grade_lines[1:], 2):
            msgs.append({"role": "assistant", "content": "（候选已更新）"})
            msgs.append({"role": "user", "content":
                         f"第{g}级线索：{line}\n更新你的候选列表"
                         "（≥3 个，把握%），并说明哪条证据改变了排序。"})
        return ask(msgs)
    if mech == "d":  # self-critique on top of c
        base = run_mechanism("c", puzzle, grade_lines)
        return ask([{"role": "user", "content":
                     f"谜题：{puzzle['q']}\n" + "\n".join(grade_lines) +
                     "\n以下是你此前的候选推理：\n" + base +
                     "\n请批判性审查：候选集是否忽略了什么解释？"
                     "若有，修订你的最终前三候选（含把握%）。"}])
    raise ValueError(mech)


def main():
    puzzles = json.loads(PUZZLES_PATH.read_text(encoding="utf-8"))
    rows = []
    done = set()
    if OUT.exists():
        rows = json.loads(OUT.read_text())["rows"]
        done = {(r["puzzle"], r["mech"], r["grade"]) for r in rows}
    for pz in puzzles:
        for mech in "abcd":
            for grade in (1, 2, 3):
                key = (pz["id"], mech, grade)
                if key in done:
                    continue
                lines = [pz["g1"], pz["g2"], pz["g3"]][:grade]
                t0 = time.time()
                text = run_mechanism(mech, pz, lines)
                rank = gold_rank_v2(text, pz["gold"])
                row = {"puzzle": pz["id"], "mech": mech, "grade": grade,
                       "rank": rank,
                       "dup": gold_dup_count(text, pz["gold"]),
                       "secs": round(time.time() - t0, 1),
                       "text": text}
                rows.append(row)
                OUT.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
                print(f"[{pz['id']} M-{mech} G{grade}] rank={rank} "
                      f"({row['secs']}s)", flush=True)
    Path("/tmp/p25v2_done").write_text("done")
    print(f"written: {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
