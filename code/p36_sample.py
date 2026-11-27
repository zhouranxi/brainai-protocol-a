#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P36 (ADR-0035): blinded sampling of M-d truncated rows for manual commitment judgement.

Writes:
  results/p36_commit/blind_{1200,2400}.txt  -- (id, text) ONLY. No rank, no grade, no puzzle id.
  results/p36_commit/key.json               -- the mapping, merged back only after judging.
Sampling: seed 20261127, without replacement, fixed sizes (60 / 30). Frozen before reading.
"""
import json, random
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "p36_commit"
SEED, N12, N24 = 20261127, 60, 30

def main():
    key = {}
    for mt, n in ((1200, N12), (2400, N24)):
        rows = json.loads((REPO / "results" / "p35_fulltext" / f"p35_race_{mt}.json").read_text(encoding="utf-8"))["rows"]
        pool = [r for r in rows if r["mech"] == "d" and r["finish_reason"] == "length"]
        random.seed(SEED + mt)
        samp = random.sample(pool, n)
        # shuffle the ids too so the file order carries no information
        lines = []
        for k, r in enumerate(samp, 1):
            sid = f"R{mt}-{k:03d}"
            key[sid] = {"arm": mt, "puzzle": r["puzzle"], "grade": r["grade"], "rep": r["rep"], "rank": r["rank"],
                        "len": len(r["text"])}
            lines.append(f"==================== {sid} ====================\n{r['text']}\n")
        (OUT / f"blind_{mt}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"arm {mt}: pool {len(pool)} truncated M-d rows -> sampled {n}; blind file written")
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    # blinding self-check: the blind files must not contain the recorded rank anywhere
    key_words = ("rank", "grade", "difficulty", "puzzle_id", "recorded")
    for mt in (1200, 2400):
        t = (OUT / f"blind_{mt}.txt").read_text(encoding="utf-8")
        # only the separator lines are structural; every separator must be exactly an id line
        seps = [ln for ln in t.splitlines() if ln.startswith("====")]
        assert len(seps) == (N12 if mt == 1200 else N24)
        assert all(__import__("re").fullmatch(r"=+ R\d+-\d{3} =+", ln) for ln in seps), "a separator carries extra text"
        # the model's own text may legitimately contain the word 'rank' (e.g. English); check only the structural part
        structure = "\n".join(seps)
        assert not any(w in structure.lower() for w in key_words)
    print("blinding check OK (no rank field in blind files)")

if __name__ == "__main__":
    main()
