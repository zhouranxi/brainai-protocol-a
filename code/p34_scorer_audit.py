#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P34 audit: scorer replay + truncation census on the P25 v2 batch + P32 cross-check.

Four checks on results/p25_reveal/p25_race_v2.json (360 rows, full texts)
and results/p32_decompose/ (1080 rows, arms b3/b4):

  1. REPLAY   -- re-run the archived v2 scorer (p25_race_v2.gold_rank_v2, copied
                 verbatim below) on every row and compare with the recorded rank.
  2. TRUNC    -- census of outputs cut by max_tokens=1200. A row is truncated if
                 it is long (>800 chars) AND does not end on sentence/list
                 terminator. Short bare answers (mech b: '连篇累牍') are complete,
                 not truncated, hence the length gate.
  3. HAZARD   -- the v2 scorer lets a candidate borrow a confidence percentage from
                 the next two lines. We count cases where the borrowed pct comes from
                 a *different* enumeration (numbering restarts) and the borrowing
                 line mentions gold, and whether requiring pct on the candidate's
                 own line would change any recorded rank.

Finding (2026-11-24): replay 0/360 mismatches; truncation is mech-specific
(d 57.8%, c 7.8%, a 4.4%, b 0%) and does not shift top-1 within mech;
cross-list borrowing touches 5 rows without changing any rank.
"""
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # github-repo/
ROWS = REPO / "results" / "p25_reveal" / "p25_race_v2.json"
PUZZLES = REPO / "code" / "p25_puzzles.json"

# ---- v2 scorer, verbatim from github-repo/code/p25_race_v2.py:46-85 ----
ITEM_RE = re.compile(r"^\s*(?:\d+[\.、)]|[-•*]|\(\d+\)|（\d+）)")
NUM_RE = re.compile(r"^\s*(\d+)[\.、)]")
PCT_RE = re.compile(r"\d+(\.\d+)?\s*%")


def _parse_items(lines):
    items, last_no = [], 0
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
    if text.startswith("[ERROR"):
        return -1
    lines = text.splitlines()
    items = _parse_items(lines)
    cand = [(pct, no) for (i, no, pct) in items if gold in lines[i]]
    if cand:
        best = max(cand)
        return min(best[1], 3)
    return 0 if gold not in text else 1


def _strict_rank(text, gold):
    """Variant: pct must be on the candidate's own line."""
    lines = text.splitlines()
    cand = [(float(PCT_RE.search(lines[i]).group(0).rstrip("%")), no)
            for (i, no, _) in _parse_items(lines)
            if gold in lines[i] and PCT_RE.search(lines[i])]
    return min(max(cand)[1], 3) if cand else None


TERMINATORS = set("。！？.!)）**\"'`")


def truncated(text):
    t = text.rstrip()
    return len(t) > 800 and t[-1] not in TERMINATORS


def main():
    gold_of = {p["id"]: p["gold"] for p in json.loads(PUZZLES.read_text(encoding="utf-8"))}
    rows = json.loads(ROWS.read_text(encoding="utf-8"))["rows"]
    print(f"rows: {len(rows)}")

    # 1. replay
    mism = [(r["puzzle"], r["mech"], r["grade"], r["rank"], gold_rank_v2(r["text"], gold_of[r["puzzle"]]))
            for r in rows if gold_rank_v2(r["text"], gold_of[r["puzzle"]]) != r["rank"]]
    print(f"\n[1] replay mismatches: {len(mism)}/{len(rows)}")
    for m in mism[:10]:
        print("    ", m)

    # 2. truncation
    print("\n[2] truncation census (len>800 and unterminated)")
    print(f"    {'mech':<5}{'trunc':>7}{'n':>6}{'rate':>8}{'max_len':>9}{'median':>8}")
    for m in "acdb":
        sub = [r for r in rows if r["mech"] == m]
        tr = [r for r in sub if truncated(r["text"])]
        L = [len(r["text"]) for r in sub]
        print(f"    {m:<5}{len(tr):>7}{len(sub):>6}{len(tr)/len(sub):>8.1%}{max(L):>9}{int(st.median(L)):>8}")
    for m in "cd":
        sub = [r for r in rows if r["mech"] == m]
        ct = Counter(r["rank"] for r in sub if truncated(r["text"]))
        cf = Counter(r["rank"] for r in sub if not truncated(r["text"]))
        nt, nf = sum(ct.values()), sum(cf.values())
        print(f"    mech {m} top1: truncated {ct[1]}/{nt} = {ct[1]/nt:.0%} | complete {cf[1]}/{nf} = {cf[1]/nf:.0%}")

    # 3. cross-list borrowing
    haz = []
    for r in rows:
        gold = gold_of[r["puzzle"]]
        lines = r["text"].splitlines()
        for (i, no, pct) in _parse_items(lines):
            if PCT_RE.search(lines[i]) or gold not in lines[i]:
                continue
            src = next((j for j in (i + 1, i + 2) if j < len(lines) and PCT_RE.search(lines[j])), None)
            if src is None:
                continue
            mm = NUM_RE.match(lines[src])
            if mm and int(mm.group(1)) <= no:
                haz.append((r["puzzle"], r["mech"], r["grade"], r["rank"]))
    changed = [(r["puzzle"], r["mech"], r["grade"], r["rank"], s)
               for r in rows
               for s in [_strict_rank(r["text"], gold_of[r["puzzle"]])]
               if s is not None and s != r["rank"]]
    print(f"\n[3] cross-list pct borrowing touching gold: {len(haz)} rows {haz}")
    print(f"    rank differs under own-line-only scorer: {len(changed)} rows, by mech {dict(Counter(c[1] for c in changed))}")

    # hard assertions the paper can lean on
    assert len(mism) == 0, "scorer replay must be exact"
    td = sum(1 for r in rows if r["mech"] == "d" and truncated(r["text"]))
    assert td >= 45, f"M-d truncation expected >50%, got {td}/90"

    # 4. cross-check on the P32 probe (full texts archived, arms b3/b4 only):
    #    the format-only arm b3 should NOT show the M-d-level truncation.
    p32 = REPO / "results" / "p32_decompose" / "p32_decompose.json"
    if p32.exists():
        rows32 = json.loads(p32.read_text(encoding="utf-8"))["rows"]
        print("\n[4] P32 probe truncation (cross-check, format-only arms)")
        for arm in sorted({r["arm"] for r in rows32}):
            sub = [r for r in rows32 if r["arm"] == arm]
            tr = sum(1 for r in sub if truncated(r["text"]))
            print(f"    {arm}: {tr}/{len(sub)} = {tr/len(sub):.1%}")
            assert tr / len(sub) < 0.10, f"{arm} truncation unexpectedly high"
    print("\nassertions OK")


if __name__ == "__main__":
    main()
