#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P35 (ADR-0034): P31 confirmation batch re-run with full texts, plus an
M-d arm with the token cap lifted.

Same 60 puzzles, same temperature (0.7 x 3), same prompts as p31_race.py.
Differences:
  * stores `text` (full), `base` (M-d's pre-critique candidate output),
    `finish_reason` and `base_finish_reason` from the API (exact truncation flag)
  * `--max-tokens` parameterised (1200 = faithful re-run, 2400 = untruncated arm)
  * `--mechs` to restrict arms
Resume-safe. Reuses p25_race_v2's scorer via import.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/code")
from p25_race_v2 import gold_rank_v2, gold_dup_count  # noqa: E402

BASE_URL = "http://localhost:8543/v1/chat/completions"
PUZZLES_PATH = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p31/p31_puzzles.json")
OUT_DIR = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p35")
TEMPS = [0.7, 0.7, 0.7]


def ask_t(messages, temp, max_tokens):
    """Return (content, finish_reason). finish_reason=='length' == truncated."""
    payload = json.dumps({
        "model": "qwen",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        BASE_URL, data=payload, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            ch = d["choices"][0]
            return (ch["message"].get("content") or ""), ch.get("finish_reason", "?")
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return f"[ERROR {type(e).__name__}]", "error"
            time.sleep(5)


def run_mech(mech, puzzle, grade_lines, temp, mt):
    """Return (final_text, finish_reason, base_text_or_None, base_finish_or_None).
    Prompts are byte-identical to p31_race.run_mech_t."""
    base = base_fin = None
    if mech == "a":
        text, fin = ask_t([{"role": "user", "content":
                            f"谜题：{puzzle['q']}\n{grade_lines[-1]}\n"
                            "列出前三个候选假设，按可能性排序。"}], temp, mt)
    elif mech == "b":
        if len(grade_lines) == 1:
            text, fin = ask_t([{"role": "user", "content":
                                f"谜题：{puzzle['q']}\n{grade_lines[0]}\n"
                                "直接给出你最有把握的一个答案，并简述理由。"}], temp, mt)
        else:
            text, fin = ask_t([
                {"role": "user", "content":
                 f"谜题：{puzzle['q']}\n{grade_lines[0]}\n直接给出你最有把握的一个答案。"},
                {"role": "assistant", "content": "（已给出答案）"},
                {"role": "user", "content": grade_lines[-1] +
                 "\n新线索出现了。你可以在原答案基础上微调（或维持）。"}], temp, mt)
    elif mech == "c":
        msgs = [{"role": "user", "content":
                 f"谜题：{puzzle['q']}\n{grade_lines[0]}\n"
                 "列出至少三个候选假设，按把握排序，每个附把握百分比。"
                 "证据不足时禁止收敛到单一答案。"}]
        for g, line in enumerate(grade_lines[1:], 2):
            msgs.append({"role": "assistant", "content": "（候选已更新）"})
            msgs.append({"role": "user", "content":
                         f"第{g}级线索：{line}\n更新你的候选列表"
                         "（≥3 个，把握%），并说明哪条证据改变了排序。"})
        text, fin = ask_t(msgs, temp, mt)
    elif mech == "d":
        base, base_fin, _, _ = run_mech("c", puzzle, grade_lines, temp, mt)
        text, fin = ask_t([{"role": "user", "content":
                            f"谜题：{puzzle['q']}\n" + "\n".join(grade_lines) +
                            "\n以下是你此前的候选推理：\n" + base +
                            "\n请批判性审查：候选集是否忽略了什么解释？"
                            "若有，修订你的最终前三候选（含把握%）。"}], temp, mt)
    else:
        raise ValueError(mech)
    return text, fin, base, base_fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--mechs", default="abcd")
    args = ap.parse_args()
    mt = args.max_tokens
    mechs = [m for m in args.mechs if m in "abcd"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"p35_race_{mt}.json"
    done_flag = Path(f"/tmp/p35_{mt}_done")
    done_flag.unlink(missing_ok=True)

    puzzles = json.loads(PUZZLES_PATH.read_text(encoding="utf-8"))
    rows, done = [], set()
    if out.exists():
        rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
        done = {(r["puzzle"], r["mech"], r["grade"], r["rep"]) for r in rows}
    n_total = len(puzzles) * len(mechs) * 3 * len(TEMPS)
    print(f"[mt={mt} mechs={''.join(mechs)}] resume: {len(done)}/{n_total}", flush=True)

    for pz in puzzles:
        for mech in mechs:
            for grade in (1, 2, 3):
                for rep, temp in enumerate(TEMPS):
                    if (pz["id"], mech, grade, rep) in done:
                        continue
                    lines = [pz["g1"], pz["g2"], pz["g3"]][:grade]
                    t0 = time.time()
                    text, fin, base, base_fin = run_mech(mech, pz, lines, temp, mt)
                    rank = gold_rank_v2(text, pz["gold"])
                    row = {"puzzle": pz["id"], "mech": mech, "grade": grade,
                           "rep": rep, "rank": rank,
                           "dup": gold_dup_count(text, pz["gold"]),
                           "secs": round(time.time() - t0, 1),
                           "difficulty": pz["difficulty"],
                           "finish_reason": fin, "text": text}
                    if mech == "d":
                        row["base_finish_reason"] = base_fin
                        row["base"] = base
                    rows.append(row)
                    out.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
                    if len(rows) % 20 == 0:
                        print(f"[{len(rows)}/{n_total}] {pz['id']} M-{mech} G{grade} "
                              f"r{rep} rank={rank} fin={fin}", flush=True)
    done_flag.write_text("done")
    print(f"written: {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
