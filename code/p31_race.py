#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADR-0031 R2 confirmation batch: 60 new puzzles (P31-P90) x 4 mechanisms
x 3 grades x 3 temperature resamples (0.7). Median rank per (mech,grade).
Only M-c vs M-d vs M-a vs M-b, frozen protocol from ADR-0030.
Resume-safe. Reuses p25_race_v2's ask/gold_rank_v2 via import.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/code")
from p25_race_v2 import ask, gold_rank_v2, gold_dup_count, run_mechanism  # noqa: E402

OUT = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p31/p31_race.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
PUZZLES_PATH = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p31/p31_puzzles.json")
TEMPS = [0.7, 0.7, 0.7]  # 3 resamples (temp=0 replaced by 0.7 per ADR-0031)


def ask_t(messages, temp, max_tokens=1200):
    import json as _json
    import urllib.request
    payload = _json.dumps({
        "model": "qwen",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8543/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    import time as _time
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = _json.loads(r.read())
            return d["choices"][0]["message"].get("content") or ""
        except Exception as e:
            if attempt == 2:
                return f"[ERROR {type(e).__name__}]"
            _time.sleep(5)


def run_mech_t(mech, puzzle, grade_lines, temp):
    """Same as p25_race_v2.run_mechanism but with temperature."""
    if mech == "a":
        return ask_t([{"role": "user", "content":
                       f"谜题：{puzzle['q']}\n{grade_lines[-1]}\n"
                       "列出前三个候选假设，按可能性排序。"}], temp)
    if mech == "b":
        if len(grade_lines) == 1:
            return ask_t([{"role": "user", "content":
                           f"谜题：{puzzle['q']}\n{grade_lines[0]}\n"
                           "直接给出你最有把握的一个答案，并简述理由。"}], temp)
        return ask_t([
            {"role": "user", "content":
             f"谜题：{puzzle['q']}\n{grade_lines[0]}\n直接给出你最有把握的一个答案。"},
            {"role": "assistant", "content": "（已给出答案）"},
            {"role": "user", "content": grade_lines[-1] +
             "\n新线索出现了。你可以在原答案基础上微调（或维持）。"}], temp)
    if mech == "c":
        msgs = [{"role": "user", "content":
                 f"谜题：{puzzle['q']}\n{grade_lines[0]}\n"
                 "列出至少三个候选假设，按把握排序，每个附把握百分比。"
                 "证据不足时禁止收敛到单一答案。"}]
        for g, line in enumerate(grade_lines[1:], 2):
            msgs.append({"role": "assistant", "content": "（候选已更新）"})
            msgs.append({"role": "user", "content":
                         f"第{g}级线索：{line}\n更新你的候选列表"
                         "（≥3 个，把握%），并说明哪条证据改变了排序。"})
        return ask_t(msgs, temp)
    if mech == "d":
        base = run_mech_t("c", puzzle, grade_lines, temp)
        return ask_t([{"role": "user", "content":
                       f"谜题：{puzzle['q']}\n" + "\n".join(grade_lines) +
                       "\n以下是你此前的候选推理：\n" + base +
                       "\n请批判性审查：候选集是否忽略了什么解释？"
                       "若有，修订你的最终前三候选（含把握%）。"}], temp)
    raise ValueError(mech)


def main():
    puzzles = json.loads(PUZZLES_PATH.read_text(encoding="utf-8"))
    rows = []
    done = set()
    if OUT.exists():
        rows = json.loads(OUT.read_text())["rows"]
        done = {(r["puzzle"], r["mech"], r["grade"], r["rep"]) for r in rows}
    n_total = len(puzzles) * 4 * 3 * 3
    n_done = len(done)
    print(f"resume: {n_done}/{n_total} done", flush=True)
    for pz in puzzles:
        for mech in "abcd":
            for grade in (1, 2, 3):
                for rep, temp in enumerate(TEMPS):
                    key = (pz["id"], mech, grade, rep)
                    if key in done:
                        continue
                    lines = [pz["g1"], pz["g2"], pz["g3"]][:grade]
                    t0 = time.time()
                    text = run_mech_t(mech, pz, lines, temp)
                    rank = gold_rank_v2(text, pz["gold"])
                    row = {"puzzle": pz["id"], "mech": mech, "grade": grade,
                           "rep": rep, "rank": rank,
                           "dup": gold_dup_count(text, pz["gold"]),
                           "secs": round(time.time() - t0, 1),
                           "difficulty": pz["difficulty"]}
                    rows.append(row)
                    OUT.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
                    n_done += 1
                    if n_done % 20 == 0:
                        print(f"[{n_done}/{n_total}] {pz['id']} M-{mech} "
                              f"G{grade} r{rep} rank={rank}", flush=True)
    Path("/tmp/p31_done").write_text("done")
    print(f"written: {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
