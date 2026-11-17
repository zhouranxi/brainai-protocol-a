#!/usr/bin/env python3
"""P18 (ADR-0023): hierarchical-prior carrier adjudication on LLMs.

100 GSM8K problems x 8 prompt conditions (C0 flat .. C7 verbose),
temperature 0, local Ollama model. Conditions:
  C0 flat           direct question
  C1 hier-decompose hierarchical task-decomposition, upfront
  C2 hier-stepwise  same tree, revealed per step (multi-turn)
  C3 flat-decompose non-hierarchical step list (structure control)
  C4 concept-hier   concept-subsumption tree only
  C5 reversed       C1 with shuffled node order
  C6 hier-post      decomposition given AFTER answering (placebo)
  C7 verbose        length-matched unstructured filler
L1-L5 frozen in ADR-0023. Resume-safe per (problem, condition).
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

CACHE = os.path.expanduser("~/.cache/p18_gsm8k_test.jsonl")
OUT = Path(__file__).resolve().parent.parent / "results" / "p18_llm_hier" / "p18_results.jsonl"
MODEL = os.environ.get("P18_MODEL", "qwen3:8b")
N = 100
OLLAMA = "http://localhost:11434/api/chat"
CONDS = ("C0_flat", "C1_hier", "C2_step", "C3_flatdec",
         "C4_concept", "C5_shuf", "C6_post", "C7_verb")


def gold_num(ans: str) -> str:
    m = re.findall(r"####\s*([\d,.\-]+)", ans)
    return m[-1].replace(",", "").strip() if m else None


def pred_num(text: str) -> str:
    m = re.findall(r"(?:final answer|answer)\D{0,20}([\d,.\-]+)", text, re.I)
    if not m:
        m = re.findall(r"####\s*([\d,.\-]+)", text)
    if not m:
        nums = re.findall(r"([\d,.\-]+)", text[-300:])
        m = [nums[-1]] if nums else [""]
    return m[-1].replace(",", "").strip().rstrip(".")


def decompose(q: str) -> list[str]:
    """Deterministic pseudo-decomposition of a GSM8K question."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", q) if s.strip()]
    steps = []
    for i, s in enumerate(sents[:-1], 1):
        steps.append(f"Step {i}: extract and record the quantity in: \"{s[:90]}...\"")
    steps.append(f"Final step: combine the recorded quantities to answer the question.")
    return steps


def build_prompt(cond: str, q: str) -> list[dict]:
    steps = decompose(q)
    sys_common = "You are a careful math solver. Give your reasoning, then end with 'Final answer: <number>'."
    if cond == "C0_flat":
        return [{"role": "system", "content": sys_common},
                {"role": "user", "content": q}]
    if cond == "C1_hier":
        tree = "Goal: answer the question.\n" + "\n".join(
            f"  {'└─ ' if i == len(steps) - 1 else '├─ '}{s}"
            for i, s in enumerate(steps))
        return [{"role": "system", "content": sys_common},
                {"role": "user", "content": f"Decomposition plan (hierarchical):\n{tree}\n\nQuestion: {q}"}]
    if cond == "C2_step":
        msgs = [{"role": "system", "content": sys_common}]
        msgs.append({"role": "user", "content":
                     f"Question: {q}\nI will guide you step by step."})
        msgs.append({"role": "assistant", "content":
                     "Understood. Please give me the first sub-task."})
        for s in steps:
            msgs.append({"role": "user", "content": f"{s}\n(continue; more steps may follow)"})
        msgs.append({"role": "user", "content": "All steps given. Now solve."})
        return msgs
    if cond == "C3_flatdec":
        flat = "Steps:\n" + "\n".join(f"- {s}" for s in steps)
        return [{"role": "system", "content": sys_common},
                {"role": "user", "content": f"{flat}\n\nQuestion: {q}"}]
    if cond == "C4_concept":
        concepts = sorted({w.lower().strip('.,') for w in re.findall(r"[A-Za-z]{4,}", q)})[:8]
        tree = "Relevant concepts (taxonomy):\n" + "\n".join(
            f"  ├─ quantity: {c}" for c in concepts[:-1]) + f"\n  └─ quantity: {concepts[-1]}"
        return [{"role": "system", "content": sys_common},
                {"role": "user", "content": f"{tree}\n\nQuestion: {q}"}]
    if cond == "C5_shuf":
        rng = random.Random(42)
        sh = steps[:]
        rng.shuffle(sh)
        tree = "Goal: answer the question.\n" + "\n".join(f"  - {s}" for s in sh)
        return [{"role": "system", "content": sys_common},
                {"role": "user", "content": f"Decomposition plan (hierarchical):\n{tree}\n\nQuestion: {q}"}]
    if cond == "C6_post":
        tree = "\n".join(f"  - {s}" for s in steps)
        return [{"role": "system", "content": sys_common},
                {"role": "user", "content":
                 f"{q}\n\n(First solve it. After your final answer, here is a decomposition you may ignore:\n{tree})"}]
    if cond == "C7_verb":
        filler = ("Please read the question attentively and consider all the "
                  "mentioned quantities carefully. Take your time, think "
                  "through the arithmetic step by step with precision, and "
                  "double-check units before concluding.")
        return [{"role": "system", "content": sys_common},
                {"role": "user", "content": f"{filler}\n\nQuestion: {q}"}]


def run_ollama(msgs):
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(
            {"model": MODEL, "messages": msgs, "stream": False,
             "options": {"temperature": 0}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d["message"]["content"], d.get("eval_count", 0), round(time.time() - t0, 1)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["pid"], r["cond"]))
        print(f"[resume] {len(done)} done", flush=True)
    lines = [json.loads(l) for l in open(CACHE, encoding="utf-8")][:N]
    total = 0
    for pid, item in enumerate(lines):
        gold = gold_num(item["answer"])
        for cond in CONDS:
            if (pid, cond) in done:
                continue
            msgs = build_prompt(cond, item["question"])
            try:
                text, ntok, secs = run_ollama(msgs)
            except Exception as e:
                text, ntok, secs = f"__ERROR__ {e}", 0, 0.0
            row = {"pid": pid, "cond": cond, "gold": gold,
                   "pred": pred_num(text), "correct": pred_num(text) == gold,
                   "ntok": ntok, "secs": secs}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            if total % 20 == 0:
                print(f"[{total}] last: {pid}/{cond} ok={row['correct']}",
                      flush=True)
    print(f"ALL DONE, wrote {total} new rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
