#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P25 (ADR-0030): graded-reveal mechanism race on Qwen3.8-27B.

Four inference-time mechanisms x 30 locked puzzles x 3 grades.
Measures top-3 recall + rank of gold per grade, per mechanism.
Resume-safe (per puzzle x mechanism x grade). Zero training.
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/code")
OUT = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p25/p25_race.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8543/v1/chat/completions"

PUZZLES_PATH = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p25/p25_puzzles.json")


def ask(messages, max_tokens=400):
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
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            msg = d["choices"][0]["message"]
            return msg.get("content") or ""
        except Exception as e:
            if attempt == 2:
                return f"[ERROR {type(e).__name__}]"
            time.sleep(5)


def gold_rank(text, gold):
    """Rank of gold in the candidate list: 1 if first-mentioned in top list,
    2/3 if mentioned later, 0 if absent, -1 if text error."""
    if text.startswith("[ERROR"):
        return -1
    # strip to the candidate enumeration (first ~5 lines)
    head = text[:800]
    if gold in head:
        # position among list items: count bullet/number markers before gold
        before = head[:head.index(gold)]
        items = re.findall(r"^\s*(?:\d+[\.、)]|[-•*])", before, re.M)
        return min(len(items) + 1, 3)
    return 0


def run_mechanism(mech, puzzle, grade_lines):
    """Build messages per mechanism up to current grade, ask once, return text."""
    if mech == "a":  # stateless: only the current grade's clue
        return ask([{"role": "user", "content":
                     f"谜题：{puzzle['q']}\n{grade_lines[-1]}\n"
                     "列出前三个候选假设，按可能性排序。"}])
    if mech == "b":  # greedy lock: commit to single answer at g1, adjust after
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
                rank = gold_rank(text, pz["gold"])
                row = {"puzzle": pz["id"], "mech": mech, "grade": grade,
                       "rank": rank, "secs": round(time.time() - t0, 1),
                       "text_head": text[:120]}
                rows.append(row)
                OUT.write_text(json.dumps({"rows": rows}, indent=1))
                print(f"[{pz['id']} M-{mech} G{grade}] rank={rank} "
                      f"({row['secs']}s)", flush=True)
    print(f"written: {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
