#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P38 (ADR-0037): stateless mechanism (M-a) at temperature 0 and extra temp-0.7 resamples.

Arms (same 60 P31 puzzles, 3 grades, prompt byte-for-byte from p31_race.run_mech_t, mech "a"):
  --temp 0.0 --mech a --reps 3 --rep-start 0    (arm A: temp-0 baseline + determinism gate)
  --temp 0.7 --mech a --reps 4 --rep-start 3    (arm B: resamples 3..6; pooled with P31 (0..2)
                                                 and R1200 (re-indexed 3..5) this gives 10 draws per cell)
max_tokens 1200 as in P31. Stores text + finish_reason + rank + top3 + dup. Resume-safe.
Reuses the archived scorer from p25_race_v2. Derived from p37_race_stability.py.
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/code")
from p25_race_v2 import gold_rank_v2, gold_dup_count  # noqa: E402

BASE = "http://localhost:8543/v1/chat/completions"
PUZZLES = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p31/p31_puzzles.json")
OUT_DIR = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p38")


def ask(messages, temp, max_tokens=1200):
    payload = json.dumps({"model": "qwen", "messages": messages, "max_tokens": max_tokens,
                          "temperature": temp,
                          "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE, data=payload, headers={"Content-Type": "application/json"})
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


def run(mech, puzzle, lines, temp):
    """Prompts copied verbatim from p31_race.run_mech_t."""
    if mech == "a":
        # verbatim from p31_race.run_mech_t, branch mech == "a"
        return ask([{"role": "user", "content":
                     f"谜题：{puzzle['q']}\n{lines[-1]}\n"
                     "列出前三个候选假设，按可能性排序。"}], temp)
    if mech == "c":
        msgs = [{"role": "user", "content":
                 f"谜题：{puzzle['q']}\n{lines[0]}\n"
                 "列出至少三个候选假设，按把握排序，每个附把握百分比。"
                 "证据不足时禁止收敛到单一答案。"}]
        for g, line in enumerate(lines[1:], 2):
            msgs.append({"role": "assistant", "content": "（候选已更新）"})
            msgs.append({"role": "user", "content":
                         f"第{g}级线索：{line}\n更新你的候选列表"
                         "（≥3 个，把握%），并说明哪条证据改变了排序。"})
        return ask(msgs, temp)
    if mech == "b":
        if len(lines) == 1:
            return ask([{"role": "user", "content":
                         f"谜题：{puzzle['q']}\n{lines[0]}\n"
                         "直接给出你最有把握的一个答案，并简述理由。"}], temp)
        return ask([
            {"role": "user", "content":
             f"谜题：{puzzle['q']}\n{lines[0]}\n直接给出你最有把握的一个答案。"},
            {"role": "assistant", "content": "（已给出答案）"},
            {"role": "user", "content": lines[-1] +
             "\n新线索出现了。你可以在原答案基础上微调（或维持）。"}], temp)
    raise ValueError(mech)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--mech", choices=["a", "b", "c"], required=True)
    ap.add_argument("--rep-start", type=int, default=0, help="first rep index to fill (arm B uses 3)")
    ap.add_argument("--reps", type=int, required=True)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"p38_{a.mech}_t{str(a.temp).replace('.', '')}_s{a.rep_start}r{a.reps}"
    out = OUT_DIR / f"{tag}.json"
    Path(f"/tmp/p38_{tag}_done").unlink(missing_ok=True)
    puzzles = json.loads(PUZZLES.read_text(encoding="utf-8"))
    rows, done = [], set()
    if out.exists():
        rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
        done = {(r["puzzle"], r["grade"], r["rep"]) for r in rows}
    total = len(puzzles) * 3 * a.reps
    print(f"[{tag}] resume {len(done)}/{total}", flush=True)
    for pz in puzzles:
        for grade in (1, 2, 3):
            for rep in range(a.rep_start, a.rep_start + a.reps):
                if (pz["id"], grade, rep) in done:
                    continue
                t0 = time.time()
                text, fin = run(a.mech, pz, [pz["g1"], pz["g2"], pz["g3"]][:grade], a.temp)
                rows.append({"puzzle": pz["id"], "mech": a.mech, "grade": grade, "rep": rep,
                             "temp": a.temp, "rank": gold_rank_v2(text, pz["gold"]),
                             "top3": int(gold_rank_v2(text, pz["gold"]) in (1, 2, 3)),
                             "dup": gold_dup_count(text, pz["gold"]),
                             "difficulty": pz["difficulty"],
                             "secs": round(time.time() - t0, 1), "finish_reason": fin, "text": text})
                out.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
                if len(rows) % 25 == 0:
                    print(f"[{len(rows)}/{total}] {pz['id']} G{grade} r{rep} rank={rows[-1]['rank']} fin={fin} {rows[-1]['secs']}s", flush=True)
    Path(f"/tmp/p38_{tag}_done").write_text("done")
    print(f"written: {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
