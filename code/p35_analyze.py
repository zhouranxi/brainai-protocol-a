#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P35 (ADR-0034) read-out: truncation census (Q1) + untruncated-arm comparison (Q2).

Usage:
  python code/p35_analyze.py 1200            # phase A: Q1 + consistency check vs P31
  python code/p35_analyze.py 1200 2400       # phase B: phase A + Q2 gate R-T1 / R-T0

Conventions (frozen in ADR-0034 and paper/p35_judge_prompt.md):
  * truncation flag  : API-native finish_reason == "length" (primary);
                       P34 heuristic len>800 & unterminated (secondary, to dock with batch-1)
  * top-1, caliber (a): per puzzle, median rank over the 3 resamples, then share with median == 1
  * top-1, caliber (b): share of rows with rank == 1
  * sign test         : paired per puzzle on the M-d top-1 indicator, R2400 vs R1200,
                       two-sided exact binomial on the discordant pairs (ties dropped)
  * consistency check : R1200 per-mechanism G3 top-1 vs original P31 (results/p31_confirm)

Nothing here writes to the paper; it only prints.
"""
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
MECHS = "abcd"
BATCH1 = {"d": 57.8, "c": 7.8, "a": 4.4, "b": 0.0}  # P34 heuristic census, results/p25_reveal

TERMINATORS = set("。！？.!)）**\"'`")


def truncated_heur(text):
    """P34 heuristic, verbatim (code/p34_scorer_audit.py:84-86)."""
    t = text.rstrip()
    return len(t) > 800 and t[-1] not in TERMINATORS


def median(xs):
    """Upper median of three, the paper's convention (code/p31_fig_main.py:28-30)."""
    s = sorted(xs)
    return s[len(s) // 2]


def load(mt):
    p = RES / "p35_fulltext" / f"p35_race_{mt}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["rows"]


def med_rank(rows):
    """(mech, grade, puzzle) -> median rank over reps."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["mech"], r["grade"], r["puzzle"])].append(r["rank"])
    return {k: median(v) for k, v in acc.items()}


def top1_medians(rows, mechs=MECHS):
    """Caliber (a): {mech: {grade: share of puzzles with median rank == 1}}."""
    med = med_rank(rows)
    out = {}
    for m in mechs:
        out[m] = {}
        for g in (1, 2, 3):
            rs = [v for (mm, gg, _), v in med.items() if mm == m and gg == g]
            out[m][g] = (sum(1 for x in rs if x == 1) / len(rs), len(rs)) if rs else (float("nan"), 0)
    return out


def top1_rows(rows, mechs=MECHS):
    """Caliber (b): {mech: {grade: share of rows with rank == 1}}."""
    out = {}
    for m in mechs:
        out[m] = {}
        for g in (1, 2, 3):
            rs = [r["rank"] for r in rows if r["mech"] == m and r["grade"] == g]
            out[m][g] = (sum(1 for x in rs if x == 1) / len(rs), len(rs)) if rs else (float("nan"), 0)
    return out


def dup_medians(rows, mechs=MECHS):
    """{mech: {grade: mean duplicate-gold count}}, row-level."""
    out = {}
    for m in mechs:
        out[m] = {}
        for g in (1, 2, 3):
            rs = [r["dup"] for r in rows if r["mech"] == m and r["grade"] == g]
            out[m][g] = (sum(rs) / len(rs), len(rs)) if rs else (float("nan"), 0)
    return out


def sign_test_p(wins, losses):
    """Two-sided exact sign test on discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def integrity(rows, mt):
    """Health checks: full factorial, unique keys, no API errors, full texts present."""
    keys = [(r["puzzle"], r["mech"], r["grade"], r["rep"]) for r in rows]
    mechs = sorted({r["mech"] for r in rows})
    n_pz = len({r["puzzle"] for r in rows})
    want = n_pz * len(mechs) * 3 * 3
    err = [r for r in rows if r.get("finish_reason") == "error" or r["text"].startswith("[ERROR")]
    rep_ids = sorted({r["rep"] for r in rows})
    ok = (len(keys) == len(set(keys)) and len(rows) == want
          and rep_ids == [0, 1, 2] and not err
          and all("text" in r for r in rows))
    print(f"\n[integrity mt={mt}] rows={len(rows)} expected={want} unique_keys="
          f"{len(set(keys))} puzzles={n_pz} mechs={''.join(mechs)} reps={rep_ids} "
          f"api_errors={len(err)} all_have_text={all('text' in r for r in rows)} -> {'OK' if ok else 'FAIL'}")
    return ok


def q1(rows, label):
    print(f"\n===== Q1 truncation census ({label}; flag = finish_reason == 'length') =====")
    print(f"  {'mech':<5}{'cut/n':>10}{'API rate':>10}{'P34 heur.':>11}{'batch-1':>9}"
          f"{'median chars':>14}{'p90':>7}")
    for m in "dcab":
        sub = [r for r in rows if r["mech"] == m]
        if not sub:
            continue
        cut = sum(1 for r in sub if r["finish_reason"] == "length")
        heur = sum(1 for r in sub if truncated_heur(r["text"]))
        L = sorted(len(r["text"]) for r in sub)
        b1 = f"{BATCH1[m]:.1f}%" if label == "R1200" else "-"
        print(f"  {m:<5}{cut:>4}/{len(sub):<5}{cut / len(sub):>10.1%}{heur / len(sub):>11.1%}"
              f"{b1:>9}{L[len(L) // 2]:>14}{L[int(0.9 * (len(L) - 1))]:>7}")
    # cross-tab of the two flags
    for m in "dcab":
        sub = [r for r in rows if r["mech"] == m]
        if not sub:
            continue
        api = Counter((r["finish_reason"] == "length", truncated_heur(r["text"])) for r in sub)
        print(f"    flag x-flag M-{m}: API&heur={api[(True, True)]} API-only={api[(True, False)]} "
              f"heur-only={api[(False, True)]} neither={api[(False, False)]}")
    d = [r for r in rows if r["mech"] == "d"]
    if d:
        print("\n  M-d base (pre-critique candidate step) finish_reason:")
        for name, sel in (("all", d), ("G1", [r for r in d if r["grade"] == 1]),
                          ("G2", [r for r in d if r["grade"] == 2]),
                          ("G3", [r for r in d if r["grade"] == 3])):
            c = Counter(r.get("base_finish_reason") for r in sel)
            print(f"    {name:<4} " + "  ".join(f"{k}={c[k]} ({c[k] / len(sel):.1%})" for k in ("stop", "length") if k in c)
                  + f"   n={len(sel)}")
        cut = sum(1 for r in d if r["finish_reason"] == "length")
        both = sum(1 for r in d if r["finish_reason"] == "length" and r.get("base_finish_reason") == "length")
        print(f"    critique-step cut & base-step cut: {both}/{cut} of the cut rows "
              f"({both / cut:.1%}) -> the cut falls on the critique pass in {1 - both / cut:.1%}")
        # truncation by grade for M-d
        print("    M-d cut rate by grade: " + "  ".join(
            f"G{g}={sum(1 for r in d if r['grade'] == g and r['finish_reason'] == 'length')}/"
            f"{sum(1 for r in d if r['grade'] == g)}"
            for g in (1, 2, 3)))
    print("\n  top-1 by truncation status (rows, caliber b):")
    for m in "dcab":
        sub = [r for r in rows if r["mech"] == m]
        if not sub:
            continue
        for g in (1, 2, 3):
            s = [r for r in sub if r["grade"] == g]
            t = [r for r in s if r["finish_reason"] == "length"]
            c = [r for r in s if r["finish_reason"] != "length"]
            ft = f"{sum(1 for r in t if r['rank'] == 1)}/{len(t)}" if t else "0/0"
            fc = f"{sum(1 for r in c if r['rank'] == 1)}/{len(c)}"
            print(f"    M-{m} G{g}: cut {ft:>8} | complete {fc:>8}")


def show_top1(title, t):
    print(f"\n{title}")
    print(f"  {'mech':<5}" + "".join(f"{'G' + str(g):>14}" for g in (1, 2, 3)))
    for m in "abcd":
        if m in t and t[m]:
            print(f"  {m:<5}" + "".join(
                f"{100 * t[m][g][0]:>8.1f}% ({t[m][g][1]:>3})" for g in (1, 2, 3)))


def consistency(rows1200):
    p31 = RES / "p31_confirm" / "p31_race.json"
    if not p31.exists():
        print("\n[consistency] original P31 file missing; check skipped")
        return
    old = json.loads(p31.read_text(encoding="utf-8"))["rows"]
    print("\n===== consistency check: R1200 re-run vs original P31 (same 60 puzzles, temp 0.7 x 3) =====")
    print("  key sets identical:", {(r["puzzle"], r["mech"], r["grade"], r["rep"]) for r in old}
          == {(r["puzzle"], r["mech"], r["grade"], r["rep"]) for r in rows1200})
    print("  puzzle universe:", len({r["puzzle"] for r in old}), "old /", len({r["puzzle"] for r in rows1200}), "new")
    ok_all = True
    print(f"\n  {'mech':<5}{'grade':<7}{'P31':>10}{'R1200':>10}{'diff pp':>10}{'flag':>7}"
          f"   | rows: P31 {'':<6} R1200 {'':<6} diff")
    for m in "abcd":
        for g in (1, 2, 3):
            o_n = sum(1 for (mm, gg, _), v in med_rank(old).items() if mm == m and gg == g)
            o = [v for (mm, gg, _), v in med_rank(old).items() if mm == m and gg == g]
            n = [v for (mm, gg, _), v in med_rank(rows1200).items() if mm == m and gg == g]
            o_t1 = sum(1 for x in o if x == 1) / len(o)
            n_t1 = sum(1 for x in n if x == 1) / len(n)
            d = 100 * (n_t1 - o_t1)
            flag = "RED" if abs(d) >= 10 else ""
            ok_all &= not flag
            orr = [r["rank"] for r in old if r["mech"] == m and r["grade"] == g]
            nrr = [r["rank"] for r in rows1200 if r["mech"] == m and r["grade"] == g]
            o_r = sum(1 for x in orr if x == 1) / len(orr)
            n_r = sum(1 for x in nrr if x == 1) / len(nrr)
            print(f"  {m:<5}G{g:<6}{o_t1:>10.1%}{n_t1:>10.1%}{d:>+10.1f}{flag:>7}"
                  f"   |      {o_r:>6.1%} {n_r:>6.1%} {100 * (n_r - o_r):>+6.1f}")
        g = 3
        o = [v for (mm, gg, _), v in med_rank(old).items() if mm == m and gg == g]
        n = [v for (mm, gg, _), v in med_rank(rows1200).items() if mm == m and gg == g]
        top3o = sum(1 for x in o if x in (1, 2, 3)) / len(o)
        top3n = sum(1 for x in n if x in (1, 2, 3)) / len(n)
        print(f"  {m:<5}  G3 top-3 recall  P31 {top3o:.1%}  R1200 {top3n:.1%}  ({100 * (top3n - top3o):+.1f} pp)")
    print(f"\n  consistency (< 10pp on every mechanism-grade cell): {'PASS' if ok_all else 'FAIL'}")
    # P31's headline reproduced on the re-run: M-d vs M-c endgame delta, paired sign test
    print("\n  P31 headline re-derived on R1200 (60 confirmation puzzles, median rank):")
    for label, rw in (("P31 ", old), ("R1200", rows1200)):
        med = med_rank(rw)
        pz = sorted(p for (m, g, p) in med if m == "d" and g == 3)
        for g in (1, 2, 3):
            w = sum(1 for p in pz if med[("d", g, p)] == 1 and med[("c", g, p)] != 1)
            l = sum(1 for p in pz if med[("c", g, p)] == 1 and med[("d", g, p)] != 1)
            print(f"    {label} M-d vs M-c G{g}: wins {w} / losses {l} / ties {len(pz) - w - l}, "
                  f"sign p = {sign_test_p(w, l):.3g}")
    # per-puzzle agreement
    mo, mn = med_rank(old), med_rank(rows1200)
    for m in "abcd":
        ks = [k for k in mo if k[0] == m]
        agree = sum(1 for k in ks if mo[k] == mn[k]) / len(ks)
        t1o = sum(1 for k in ks if mo[k] == 1) / len(ks)
        t1n = sum(1 for k in ks if mn[k] == 1) / len(ks)
        flip = sum(1 for k in ks if (mo[k] == 1) != (mn[k] == 1))
        print(f"    M-{m}: per-puzzle median-rank agreement {agree:.1%}; top-1 indicator flips {flip}/{len(ks)} "
              f"(top-1 {t1o:.1%} -> {t1n:.1%})")


def q2(rows1200, rows2400):
    print("\n===== Q2: M-d, R2400 (cap lifted) vs R1200 =====")
    m12, m24 = med_rank([r for r in rows1200 if r["mech"] == "d"]), med_rank(rows2400)
    a12, a24 = top1_medians(rows1200, "d")["d"], top1_medians(rows2400, "d")["d"]
    b12, b24 = top1_rows(rows1200, "d")["d"], top1_rows(rows2400, "d")["d"]
    rt1 = False
    print(f"\n  {'grade':<7}{'(a) R1200':>12}{'(a) R2400':>12}{'diff pp':>10}{'':3}"
          f"{'(b) R1200':>12}{'(b) R2400':>12}{'diff pp':>10}")
    for g in (1, 2, 3):
        da = 100 * (a24[g][0] - a12[g][0])
        db = 100 * (b24[g][0] - b12[g][0])
        print(f"  G{g:<6}{a12[g][0]:>12.1%}{a24[g][0]:>12.1%}{da:>+10.1f}{'':3}"
              f"{b12[g][0]:>12.1%}{b24[g][0]:>12.1%}{db:>+10.1f}")
    print("\n  sign test, paired per puzzle on the M-d top-1 indicator (median-rank caliber):")
    pz = sorted({p for (m, g, p) in m24 if g == 3})
    for g in (1, 2, 3):
        w = sum(1 for p in pz if m24[("d", g, p)] == 1 and m12[("d", g, p)] != 1)
        l = sum(1 for p in pz if m12[("d", g, p)] == 1 and m24[("d", g, p)] != 1)
        p = sign_test_p(w, l)
        d = 100 * (a24[g][0] - a12[g][0])
        hit = abs(d) >= 10 and p < 0.05
        rt1 |= hit
        print(f"    G{g}: R2400 wins {w} / R1200 wins {l} / ties {len(pz) - w - l}; "
              f"diff {d:+.1f} pp; p = {p:.3g} -> {'R-T1 hit' if hit else 'no hit'}")
    # row-level sign test as a sensitivity view (not the pre-registered unit)
    print("\n  sensitivity (row-level pairing by (puzzle, grade, rep); not the pre-registered unit):")
    k12 = {(r["puzzle"], r["grade"], r["rep"]): r["rank"] for r in rows1200 if r["mech"] == "d"}
    k24 = {(r["puzzle"], r["grade"], r["rep"]): r["rank"] for r in rows2400}
    common = sorted(set(k12) & set(k24))
    for g in (1, 2, 3):
        ks = [k for k in common if k[1] == g]
        w = sum(1 for k in ks if k24[k] == 1 and k12[k] != 1)
        l = sum(1 for k in ks if k12[k] == 1 and k24[k] != 1)
        print(f"    G{g}: n={len(ks)} R2400 wins {w} / R1200 wins {l} / ties {len(ks) - w - l}; "
              f"p = {sign_test_p(w, l):.3g}")
    verdict = "R-T1 (truncation has a behavioral effect)" if rt1 else "R-T0 (no behavioral effect of truncation)"
    print(f"\n  VERDICT: {verdict}")
    print("\n  secondary: duplicate-gold count, mean per row")
    d12, d24 = dup_medians(rows1200, "d")["d"], dup_medians(rows2400, "d")["d"]
    for g in (1, 2, 3):
        print(f"    G{g}: R1200 {d12[g][0]:.2f}  R2400 {d24[g][0]:.2f}  ({d24[g][0] - d12[g][0]:+.2f})")
    c12 = Counter(r["dup"] for r in rows1200 if r["mech"] == "d")
    c24 = Counter(r["dup"] for r in rows2400)
    print("    dup distribution (count -> rows):")
    print("      R1200:", dict(sorted(c12.items())))
    print("      R2400:", dict(sorted(c24.items())))


# ---- paper anchors (main.tex Table 2, merged 90 puzzles, top-1 counts at stages 1/2/3) ----
PAPER_T2 = {"a": [41, 46, 68], "b": [27, 33, 68], "c": [30, 37, 66], "d": [43, 63, 69]}


def merged_med(confirm_rows):
    """med[(mech, grade, puzzle)] over the merged 90: batch-1 30 puzzles at temp 0 + 60 confirmation."""
    b1 = json.loads((RES / "p25_reveal" / "p25_race_v2.json").read_text(encoding="utf-8"))["rows"]
    med = {(r["mech"], r["grade"], r["puzzle"]): r["rank"] for r in b1}
    med.update(med_rank(confirm_rows))
    return med


def boot_ci(med, pz, n=20000, seed=0):
    """Paired bootstrap over puzzles of (M-d top-1 rate - M-c top-1 rate) at stage 3, percentile CI."""
    import random
    rng = random.Random(seed)
    d = []
    for _ in range(n):
        s = [rng.choice(pz) for _ in pz]
        d.append((sum(1 for p in s if med[("d", 3, p)] == 1)
                  - sum(1 for p in s if med[("c", 3, p)] == 1)) / len(pz))
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n) - 1]


def exploratory_merged(old, new):
    print("\n===== EXPLORATORY (not pre-registered; re-adjudicates nothing) =====")
    print("merged 90 puzzles, confirmation 60 taken from P31 vs from R1200 (batch-1 30 at temp 0 fixed)")
    res = {}
    for label, rw in (("P31", old), ("R1200", new)):
        med = merged_med(rw)
        pz = sorted({p for (m, g, p) in med if g == 3})
        assert len(pz) == 90, len(pz)
        assert len({p for (m, g, p) in med if g == 3 and m == "c"}) == 90, "every puzzle must have a rank"
        res[label] = (med, pz)
        if label == "P31":
            got = {m: [sum(1 for p in pz if med[(m, g, p)] == 1) for g in (1, 2, 3)] for m in "abcd"}
            print(f"  anchor: paper Table 2 recomputed from P31 + batch-1 = {got} -> "
                  f"{'MATCH' if got == PAPER_T2 else 'MISMATCH vs ' + str(PAPER_T2)}")
    for m in "abcd":
        cells = []
        for label in ("P31", "R1200"):
            med, pz = res[label]
            cells.append("/".join(str(sum(1 for p in pz if med[(m, g, p)] == 1)) for g in (1, 2, 3)))
        print(f"  M-{m} top-1 /90 at G1/G2/G3:  P31 {cells[0]:>10}   R1200 {cells[1]:>10}")
    for label in ("P31", "R1200"):
        med, pz = res[label]
        parts = []
        for g in (1, 2, 3):
            w = sum(1 for p in pz if med[("d", g, p)] == 1 and med[("c", g, p)] != 1)
            l = sum(1 for p in pz if med[("c", g, p)] == 1 and med[("d", g, p)] != 1)
            parts.append(f"G{g} {w}/{l} p={sign_test_p(w, l):.3g}")
        d3 = sum(1 for p in pz if med[("d", 3, p)] == 1) - sum(1 for p in pz if med[("c", 3, p)] == 1)
        c3 = sum(1 for p in pz if med[("c", 3, p)] == 1)
        lo, hi = boot_ci(med, pz)
        rec = [100 * (sum(1 for p in pz if med[("c", g, p)] in (1, 2, 3)) - sum(1 for p in pz if med[("a", g, p)] in (1, 2, 3))) / 90
               for g in (1, 2)]
        print(f"  {label}: M-d vs M-c paired sign test (wins/losses) {'; '.join(parts)}")
        print(f"      endgame M-d - M-c = {d3:+d}/90 = {100 * d3 / 90:+.1f} pp, bootstrap 95% CI "
              f"[{100 * lo:+.1f}, {100 * hi:+.1f}] pp;  M-c endgame top-1 {c3}/90 = {100 * c3 / 90:.0f}% "
              f"(health band 50-95%);  M-c - M-a top-3 recall G1/G2 = {rec[0]:+.0f}/{rec[1]:+.0f} pp")


def flag_discrepancy(rows):
    print("\n  why the two truncation flags disagree (M-d): last 6 characters of the text")
    api_only = [r for r in rows if r["mech"] == "d" and r["finish_reason"] == "length" and not truncated_heur(r["text"])]
    heur_only = [r for r in rows if r["mech"] == "d" and r["finish_reason"] != "length" and truncated_heur(r["text"])]
    print(f"    API-cut but heuristic says complete: {len(api_only)} rows; last-character counts: "
          f"{dict(Counter(r['text'].rstrip()[-1] for r in api_only))}")
    for r in api_only[:6]:
        print(f"      ...{r['text'].rstrip()[-28:]!r}")
    print(f"    heuristic says cut but API says stop: {len(heur_only)} rows; lengths {[len(r['text']) for r in heur_only]}")
    for r in heur_only[:3]:
        print(f"      ...{r['text'].rstrip()[-28:]!r}")


def m_c_stability(old, new):
    """Is the M-c G2/G3 discrepancy a shift or median-of-3 sampling noise?

    (i) split-half of the re-run (rep 0 vs reps 1+2, and all rep pairs) in row-level top-1;
    (ii) paired sign test on the per-puzzle median top-1 indicator, P31 vs R1200;
    (iii) M-d's archived base step is the identical M-c prompt, scored independently.
    """
    print("\n===== EXPLORATORY: is the M-c gap a shift or sampling noise? =====")
    for g in (1, 2, 3):
        o = [r["rank"] for r in old if r["mech"] == "c" and r["grade"] == g]
        n = [r["rank"] for r in new if r["mech"] == "c" and r["grade"] == g]
        print(f"  M-c G{g} row-level top-1 per rep: P31 "
              + " ".join(f"{sum(1 for r in old if r['mech'] == 'c' and r['grade'] == g and r['rep'] == k and r['rank'] == 1) / 60:.1%}" for k in range(3))
              + "  | R1200 "
              + " ".join(f"{sum(1 for r in new if r['mech'] == 'c' and r['grade'] == g and r['rep'] == k and r['rank'] == 1) / 60:.1%}" for k in range(3)))
        rng_all = [sum(1 for x in o if x == 1) / len(o), sum(1 for x in n if x == 1) / len(n)]
        print(f"      pooled top-1 (180 rows each): P31 {rng_all[0]:.1%}  R1200 {rng_all[1]:.1%}  "
              f"diff {100 * (rng_all[1] - rng_all[0]):+.1f} pp")
    mo, mn = med_rank(old), med_rank(new)
    for g in (1, 2, 3):
        w = sum(1 for p in {k[2] for k in mo} if mn[("c", g, p)] == 1 and mo[("c", g, p)] != 1)
        l = sum(1 for p in {k[2] for k in mo} if mo[("c", g, p)] == 1 and mn[("c", g, p)] != 1)
        print(f"  M-c G{g} per-puzzle median top-1, paired P31 vs R1200: R1200 wins {w} / P31 wins {l} / "
              f"ties {60 - w - l}; sign p = {sign_test_p(w, l):.3g}")
    # split-half inside the re-run: how big a G3 gap do two disjoint halves of the same run show?
    print("\n  split-half of R1200 itself (median over disjoint reps is undefined for 3 reps, so use"
          " single-rep top-1):")
    for g in (2, 3):
        vals = [sum(1 for r in new if r["mech"] == "c" and r["grade"] == g and r["rep"] == k and r["rank"] == 1) / 60 for k in range(3)]
        pairs = [(abs(vals[i] - vals[j]), i, j) for i in range(3) for j in range(i + 1, 3)]
        print(f"    M-c G{g}: rep top-1 = {[round(100 * v, 1) for v in vals]}%; "
              f"max within-run gap between two reps = {100 * max(pairs)[0]:.1f} pp")
    # (iii) the base step
    print("\n  M-d's base step = the identical M-c prompt, scored on its own text:")
    d_rows = [r for r in new if r["mech"] == "d"]
    # re-score base texts with the archived v2 scorer (verbatim in p34_scorer_audit.py)
    import importlib.util
    spec = importlib.util.spec_from_file_location("p34", REPO / "code" / "p34_scorer_audit.py")
    p34 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p34)
    puzzles = None
    for cand in (REPO / "runs" / "v2_p31" / "p31_puzzles.json", REPO / "code" / "p31_puzzles.json"):
        if cand.exists():
            puzzles = cand
            break
    if puzzles is None:
        print("    puzzles file not found locally; base-step comparison skipped")
        return
    gold_of = {p["id"]: p["gold"] for p in json.loads(puzzles.read_text(encoding="utf-8"))}
    print(f"    puzzles loaded from {puzzles.relative_to(REPO)}; gold answers for {len(gold_of)} puzzles")
    # sanity: re-scoring the final M-d texts must reproduce the recorded ranks
    mism = sum(1 for r in d_rows if p34.gold_rank_v2(r["text"], gold_of[r["puzzle"]]) != r["rank"])
    print(f"    scorer replay on the 540 M-d final texts: {mism} mismatches")
    b_med = defaultdict(list)
    b_row = defaultdict(list)
    for r in d_rows:
        rk = p34.gold_rank_v2(r["base"], gold_of[r["puzzle"]])
        b_med[(r["grade"], r["puzzle"])].append(rk)
        b_row[r["grade"]].append(rk)
    for g in (1, 2, 3):
        med = [median(v) for (gg, _), v in b_med.items() if gg == g]
        t1_med = sum(1 for x in med if x == 1) / len(med)
        t1_row = sum(1 for x in b_row[g] if x == 1) / len(b_row[g])
        print(f"    base step, as M-c: G{g} top-1 median-per-puzzle {t1_med:.1%} | row-level {t1_row:.1%} "
              f"(P31 M-c: {sum(1 for k in mo if k[0] == 'c' and k[1] == g and mo[k] == 1) / 60:.1%} / "
              f"{sum(1 for r in old if r['mech'] == 'c' and r['grade'] == g and r['rank'] == 1) / 180:.1%}; "
              f"R1200 M-c: {sum(1 for k in mn if k[0] == 'c' and k[1] == g and mn[k] == 1) / 60:.1%} / "
              f"{sum(1 for r in new if r['mech'] == 'c' and r['grade'] == g and r['rank'] == 1) / 180:.1%})")


def main():
    mts = sys.argv[1:] or ["1200"]
    rows = {mt: load(mt) for mt in mts}
    ok = True
    for mt, rw in rows.items():
        if rw is None:
            print(f"[mt={mt}] result file not present; skipped")
            continue
        ok &= integrity(rw, mt)
    if rows.get("1200"):
        q1(rows["1200"], "R1200")
        flag_discrepancy(rows["1200"])
        show_top1("R1200 top-1, median rank per puzzle:", top1_medians(rows["1200"]))
        show_top1("R1200 top-1, row-level:", top1_rows(rows["1200"]))
        consistency(rows["1200"])
        if "--no-exploratory" not in sys.argv:
            old = json.loads((RES / "p31_confirm" / "p31_race.json").read_text(encoding="utf-8"))["rows"]
            exploratory_merged(old, rows["1200"])
            m_c_stability(old, rows["1200"])
    if rows.get("2400"):
        q1(rows["2400"], "R2400")
        q2(rows["1200"], rows["2400"])
    print(f"\nintegrity: {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
