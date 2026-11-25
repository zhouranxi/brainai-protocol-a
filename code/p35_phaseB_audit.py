#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit of the phase-B numbers written into paper/adr_0034.md (ADR-0034 sec. 6, phase B).

Each check pairs a number taken from the ADR TEXT with the same number recomputed from the
raw JSON (results/p35_fulltext/*.json, results/p31_confirm/p31_race.json). The two sides come
from different sources: the text vs. the data. Run after editing the ADR; all must PASS.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
ADR = REPO / "paper" / "adr_0034.md"

fails = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


def um(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    text = ADR.read_text(encoding="utf-8")
    # the phase-B block is everything after the '### 阶段 B' heading
    i = text.index("### 阶段 B（2026-11-26")
    phase_b = text[i:]
    print(f"phase-B block: {len(phase_b.splitlines())} lines, starts at ADR line "
          f"{text[:i].count(chr(10)) + 1}")

    allrows = json.loads((RES / "p35_fulltext" / "p35_race_1200.json").read_text(encoding="utf-8"))["rows"]
    r12 = [r for r in allrows if r["mech"] == "d"]
    r24 = json.loads((RES / "p35_fulltext" / "p35_race_2400.json").read_text(encoding="utf-8"))["rows"]
    p31 = [r for r in json.loads((RES / "p31_confirm" / "p31_race.json").read_text(encoding="utf-8"))["rows"]
           if r["mech"] == "d"]

    def med(rw):
        acc = defaultdict(list)
        for r in rw:
            acc[(r["grade"], r["puzzle"])].append(r["rank"])
        return {k: um(v) for k, v in acc.items()}

    m12, m24, m31 = med(r12), med(r24), med(p31)
    pz = sorted({k[1] for k in m24})

    def t1_med(m, g):
        v = [x for (gg, _), x in m.items() if gg == g]
        return sum(1 for x in v if x == 1), len(v)

    def t1_row(rw, g):
        v = [r["rank"] for r in rw if r["grade"] == g]
        return sum(1 for x in v if x == 1), len(v)

    print("\n-- integrity / execution record")
    check("R2400 has 540 rows", len(r24) == 540)
    check("R2400 is M-d only", {r["mech"] for r in r24} == {"d"})
    check("R2400: 60 puzzles x 3 grades x 3 reps, unique keys",
          len({(r["puzzle"], r["grade"], r["rep"]) for r in r24}) == 540
          and len({r["puzzle"] for r in r24}) == 60)
    check("R2400: no API errors", not any(r["finish_reason"] == "error" or r["text"].startswith("[ERROR") for r in r24))

    print("\n-- Q2 main table, median-rank caliber (a)")
    exp_a = {1: (29, 28), 2: (44, 37), 3: (45, 37)}
    for g in (1, 2, 3):
        a, b = t1_med(m12, g), t1_med(m24, g)
        check(f"G{g} (a) R1200 {a[0]}/60 and R2400 {b[0]}/60 as written ({exp_a[g][0]} -> {exp_a[g][1]})",
              (a[0], b[0]) == exp_a[g])
        d = 100 * (b[0] - a[0]) / 60
        written = {1: "−1.7", 2: "−11.7", 3: "−13.3"}[g]
        check(f"G{g} (a) diff printed as {written} pp (data {d:+.1f})", f"{d:+.1f}".replace("-", "−") == written)
        # the literal strings the ADR shows
        s_a = f"{100 * a[0] / 60:.1f}%（{a[0]}/60）"
        s_b = f"{100 * b[0] / 60:.1f}%（{b[0]}/60）"
        check(f"G{g} (a) literal '{s_a}' and '{s_b}' present in ADR", s_a in phase_b and s_b in phase_b)

    print("\n-- Q2 main table, row-level caliber (b)")
    exp_b = {1: (78, 68), 2: (116, 92), 3: (124, 108)}
    for g in (1, 2, 3):
        a, b = t1_row(r12, g), t1_row(r24, g)
        check(f"G{g} (b) R1200 {a[0]}/180 and R2400 {b[0]}/180 as written ({exp_b[g][0]} -> {exp_b[g][1]})",
              (a[0], b[0]) == exp_b[g])
        d = 100 * (b[0] - a[0]) / 180
        written = {1: "−5.6", 2: "−13.3", 3: "−8.9"}[g]
        check(f"G{g} (b) diff printed as {written} pp (data {d:+.1f})", f"{d:+.1f}".replace("-", "−") == written)

    print("\n-- pre-registered sign test, paired per puzzle")
    exp_wl = {1: (9, 10, 41, "1.000"), 2: (8, 15, 37, "0.210"), 3: (6, 14, 40, "0.115")}
    for g in (1, 2, 3):
        w = sum(1 for p in pz if m24[(g, p)] == 1 and m12[(g, p)] != 1)
        l = sum(1 for p in pz if m12[(g, p)] == 1 and m24[(g, p)] != 1)
        t = len(pz) - w - l
        p = sign_p(w, l)
        ew, el, et, ep = exp_wl[g]
        check(f"G{g} R2400 wins {w} / R1200 wins {l} / ties {t}; p = {p:.3f} (written {ew}/{el}/{et}, {ep})",
              (w, l, t) == (ew, el, et) and f"{p:.3f}" == ep)
    hit = any(abs(100 * (t1_med(m24, g)[0] - t1_med(m12, g)[0]) / 60) >= 10
              and sign_p(sum(1 for p in pz if m24[(g, p)] == 1 and m12[(g, p)] != 1),
                         sum(1 for p in pz if m12[(g, p)] == 1 and m24[(g, p)] != 1)) < 0.05 for g in (1, 2, 3))
    check("no grade meets both R-T1 conditions  =>  verdict R-T0", not hit and "判定：R-T0" in phase_b)

    print("\n-- row-level sensitivity")
    k12 = {(r["puzzle"], r["grade"], r["rep"]): r["rank"] for r in r12}
    k24 = {(r["puzzle"], r["grade"], r["rep"]): r["rank"] for r in r24}
    exp_s = {1: (27, 37, 0.26), 2: (27, 51, 0.009), 3: (32, 48, 0.093)}
    for g in (1, 2, 3):
        ks = [k for k in k12 if k[1] == g]
        w = sum(1 for k in ks if k24[k] == 1 and k12[k] != 1)
        l = sum(1 for k in ks if k12[k] == 1 and k24[k] != 1)
        p = sign_p(w, l)
        # ADR prints three decimals; compare the rounded value
        check(f"G{g} row-level {w}/{l} p = {p:.3g} (written {exp_s[g][0]}/{exp_s[g][1]}, {exp_s[g][2]})",
              (w, l) == exp_s[g][:2] and f"{p:.3f}" == f"{exp_s[g][2]:.3f}")

    print("\n-- power (n discordant, split needed)")
    import random
    for g, (n_exp, crit_exp, obs) in {1: (19, "15:4", "—"), 2: (23, "17:6", "8:15"), 3: (20, "15:5", "6:14")}.items():
        n = sum(1 for p in pz if (m24[(g, p)] == 1) != (m12[(g, p)] == 1))
        crit = next(k for k in range(n + 1) if sign_p(k, n - k) < 0.05 and k >= n - k)
        check(f"G{g}: {n} discordant puzzles, significance needs {crit}:{n - crit} (written {n_exp}, {crit_exp})",
              n == n_exp and f"{crit}:{n - crit}" == crit_exp)
        if obs != "—":
            w = sum(1 for p in pz if m24[(g, p)] == 1 and m12[(g, p)] != 1)
            l = sum(1 for p in pz if m12[(g, p)] == 1 and m24[(g, p)] != 1)
            check(f"G{g}: observed split written as {obs} (data {w}:{l})", f"{w}:{l}" == obs)
    check("10 pp on 60 puzzles is 6 puzzles", abs(0.10 * 60 - 6) < 1e-9 and "6 道题" in phase_b)

    print("\n-- bootstrap CIs (recomputed with the same stream: seed 0 per grade, 20,000 resamples)")
    exp_ci = {1: ("−16.7", "+11.7"), 2: ("−26.7", "+3.3"), 3: ("−26.7", "+1.7")}
    for g in (1, 2, 3):
        rng = random.Random(0)
        ds = []
        for _ in range(20000):
            s = [rng.choice(pz) for _ in pz]
            ds.append(sum(1 for p in s if m24[(g, p)] == 1) / 60 - sum(1 for p in s if m12[(g, p)] == 1) / 60)
        ds.sort()
        lo, hi = f"{100 * ds[500]:+.1f}".replace("-", "−"), f"{100 * ds[19500]:+.1f}".replace("-", "−")
        check(f"G{g} 95% CI [{lo}, {hi}] (written {exp_ci[g][0]}, {exp_ci[g][1]})", (lo, hi) == exp_ci[g])
        check(f"G{g} CI not inside +/-10 pp", not (ds[500] > -0.10 and ds[19500] < 0.10))

    print("\n-- three draws of the same cell")
    exp3 = {1: ("46.7", "48.3", "46.7"), 2: ("73.3", "73.3", "61.7"), 3: ("73.3", "75.0", "61.7")}
    for g in (1, 2, 3):
        got = tuple(f"{100 * t1_med(m, g)[0] / 60:.1f}" for m in (m31, m12, m24))
        check(f"G{g} P31/R1200/R2400 = {got} (written {exp3[g]})", got == exp3[g])
    for g, pps in {1: ("1.00", "1.00", "1.00"), 2: ("1.00", "0.25", "0.21"), 3: ("1.00", "0.19", "0.12")}.items():
        def pp(ma, mb):
            return sign_p(sum(1 for p in pz if mb[(g, p)] == 1 and ma[(g, p)] != 1),
                          sum(1 for p in pz if ma[(g, p)] == 1 and mb[(g, p)] != 1))
        got = tuple(f"{pp(m31, m12):.2f}", )
        got = (f"{pp(m31, m12):.2f}", f"{pp(m31, m24):.2f}", f"{pp(m12, m24):.2f}")
        check(f"G{g} paired p P31~R1200 / P31~R2400 / R1200~R2400 = {got} (written {pps})", got == pps)
    check("R1200-vs-R2400 p at G2/G3 quoted as .21/.12 in open question 5", "p=.21" in phase_b and "p=.12" in phase_b)

    print("\n-- finish_reason, R2400")
    cut = sum(1 for r in r24 if r["finish_reason"] == "length")
    check(f"M-d cut {cut}/540 = {100 * cut / 540:.1f}% (written 8.9%)", cut == 48 and f"{100 * cut / 540:.1f}" == "8.9")
    for g, (n_exp, s_exp) in {1: (1, "0.6%"), 2: (20, "11.1%"), 3: (27, "15.0%")}.items():
        n = sum(1 for r in r24 if r["grade"] == g and r["finish_reason"] == "length")
        check(f"G{g} residual cut {n}/180 = {100 * n / 180:.1f}% (written {n_exp}, {s_exp})",
              n == n_exp and f"{100 * n / 180:.1f}%" == s_exp)
    check("drop 60.9 -> 8.9 is 52pp as written", round(60.9 - 8.9) == 52)
    bs = Counter(r.get("base_finish_reason") for r in r24)
    check(f"R2400 base step stop {bs['stop']}, length {bs['length']} = {100 * bs['length'] / 540:.1f}% (written 531, 9, 1.7%)",
          (bs["stop"], bs["length"]) == (531, 9) and f"{100 * bs['length'] / 540:.1f}" == "1.7")
    bg = {g: sum(1 for r in r24 if r["grade"] == g and r.get("base_finish_reason") == "length") for g in (1, 2, 3)}
    check(f"R2400 base-step cut by grade G1/G2/G3 = {bg[1]}/{bg[2]}/{bg[3]} (written 0, 2, 7)",
          (bg[1], bg[2], bg[3]) == (0, 2, 7))
    both = sum(1 for r in r24 if r["finish_reason"] == "length" and r.get("base_finish_reason") == "length")
    check(f"R2400 cut rows with base also cut: {both}/48; critique pass share {100 * (1 - both / cut):.1f}% (written 4, 91.7%)",
          both == 4 and f"{100 * (1 - both / cut):.1f}" == "91.7")
    check("R1200 base-step cut quoted as 7.4% (phase-A value, re-derived)", 
          f"{100 * sum(1 for r in r12 if r.get('base_finish_reason') == 'length') / 540:.1f}" == "7.4")

    print("\n-- output length (characters), median [p10, p90]")
    exp_len = {("R1200", 1): (1846, 1361, 2063), ("R1200", 2): (1975, 1663, 2099), ("R1200", 3): (1974, 1657, 2113),
               ("R2400", 1): (1842, 1359, 2442), ("R2400", 2): (2392, 1699, 3989), ("R2400", 3): (2544, 1624, 4020)}
    for (lab, g), e in exp_len.items():
        L = sorted(len(r["text"]) for r in (r12 if lab == "R1200" else r24) if r["grade"] == g)
        got = (L[len(L) // 2], L[len(L) // 10], L[9 * len(L) // 10])
        check(f"{lab} G{g} {got[0]} [{got[1]}, {got[2]}] (written {e})", got == e)
    g1cut12 = 100 * sum(1 for r in r12 if r["grade"] == 1 and r["finish_reason"] == "length") / 180
    check(f"G1 cut rate R1200 = {g1cut12:.0f}% (written 39%)", f"{g1cut12:.0f}" == "39")

    print("\n-- dup")
    exp_dup = {1: ("0.48", "0.48"), 2: ("0.73", "1.38"), 3: ("0.96", "1.89")}
    for g in (1, 2, 3):
        a = sum(r["dup"] for r in r12 if r["grade"] == g) / 180
        b = sum(r["dup"] for r in r24 if r["grade"] == g) / 180
        check(f"G{g} mean dup R1200 {a:.2f} -> R2400 {b:.2f} (written {exp_dup[g]})", (f"{a:.2f}", f"{b:.2f}") == exp_dup[g])
    c12 = Counter(r["dup"] for r in r12)
    c24 = Counter(r["dup"] for r in r24)
    # the ADR writes the dict in Python's own repr form: "{0: 330, 1: 105, ...}"
    s12 = str(dict(sorted(c12.items())))
    s24 = str(dict(sorted(c24.items())))
    check(f"R1200 dup distribution literal '{s12}' in ADR", s12 in phase_b)
    check(f"R2400 dup distribution literal '{s24}' in ADR", s24 in phase_b)
    d12, d24 = defaultdict(list), defaultdict(list)
    for r in r12:
        d12[(r["grade"], r["puzzle"])].append(r["dup"])
    for r in r24:
        d24[(r["grade"], r["puzzle"])].append(r["dup"])
    for g, (w_exp, l_exp, p_exp) in {1: (6, 9, "0.607"), 2: (32, 8, "0.000182"), 3: (36, 8, "2.54e-05")}.items():
        ks = [k for k in d12 if k[0] == g]
        w = sum(1 for k in ks if um(d24[k]) > um(d12[k]))
        l = sum(1 for k in ks if um(d24[k]) < um(d12[k]))
        p = sign_p(w, l)
        check(f"dup per-puzzle sign G{g}: {w}/{l}, p = {p:.3g} (written {w_exp}/{l_exp}, {p_exp})",
              (w, l) == (w_exp, l_exp) and f"{p:.3g}" == p_exp)

    print("\n-- dup mechanism (exploratory): item lines and length-adjusted difference")
    ITEM_RE = re.compile(r"^\s*(?:\d+[\.、)]|[-•*]|\(\d+\)|（\d+）)")
    NUM_RE = re.compile(r"^\s*(\d+)[\.、)]")
    PCT_RE = re.compile(r"\d+(\.\d+)?\s*%")

    def n_items(text):
        lines = text.splitlines()
        last_no, n = 0, 0
        for i, ln in enumerate(lines):
            if not ITEM_RE.match(ln):
                continue
            m = NUM_RE.match(ln)
            last_no = int(m.group(1)) if m else last_no + 1
            if any(PCT_RE.search(lines[j]) for j in (i, i + 1, i + 2) if j < len(lines)):
                n += 1
        return n

    for g, (a_exp, b_exp) in {2: ("4.61", "8.21"), 3: ("3.58", "6.89")}.items():
        a = sum(n_items(r["text"]) for r in r12 if r["grade"] == g) / 180
        b = sum(n_items(r["text"]) for r in r24 if r["grade"] == g) / 180
        check(f"G{g} confidence-bearing item lines R1200 {a:.2f} -> R2400 {b:.2f} (written {a_exp} -> {b_exp})",
              (f"{a:.2f}", f"{b:.2f}") == (a_exp, b_exp))
    for g, (adj_exp, raw_exp) in {2: ("+0.50", "+0.66"), 3: ("+0.36", "+0.93")}.items():
        b12, b24 = defaultdict(list), defaultdict(list)
        for r in r12:
            if r["grade"] == g and r["finish_reason"] != "length":
                b12[len(r["text"]) // 200].append(r["dup"])
        for r in r24:
            if r["grade"] == g and r["finish_reason"] != "length":
                b24[len(r["text"]) // 200].append(r["dup"])
        common = [b for b in sorted(set(b12) & set(b24)) if len(b12[b]) >= 10 and len(b24[b]) >= 10]
        wt = {b: len(b12[b]) + len(b24[b]) for b in common}
        tot = sum(wt.values())
        adj = sum(wt[b] * (sum(b24[b]) / len(b24[b]) - sum(b12[b]) / len(b12[b])) for b in common) / tot
        raw = (sum(r["dup"] for r in r24 if r["grade"] == g) - sum(r["dup"] for r in r12 if r["grade"] == g)) / 180
        check(f"G{g} length-adjusted dup difference {adj:+.2f}, raw {raw:+.2f} (written {adj_exp}, {raw_exp})",
              f"{adj:+.2f}" == adj_exp and f"{raw:+.2f}" == raw_exp)
        lo_c, hi_c = min(common) * 200, max(common) * 200 + 199
        check(f"G{g} common strata span {lo_c}-{hi_c} (ADR says 1600-1999)", (lo_c, hi_c) == (1600, 1999))

    print("\n-- dup >= 2 vs top-1")
    exp_d = {("R1200", 1): (24, 4, 74), ("R1200", 2): (33, 13, 103), ("R1200", 3): (48, 12, 112),
             ("R2400", 1): (19, 4, 64), ("R2400", 2): (67, 27, 65), ("R2400", 3): (87, 33, 75)}
    for (lab, g), e in exp_d.items():
        rw = r12 if lab == "R1200" else r24
        s = [r for r in rw if r["grade"] == g]
        hi = [r for r in s if r["dup"] >= 2]
        lo = [r for r in s if r["dup"] < 2]
        got = (len(hi), sum(1 for r in hi if r["rank"] == 1), sum(1 for r in lo if r["rank"] == 1))
        check(f"{lab} G{g}: dup>=2 rows {got[0]}, top-1 among them {got[1]}, top-1 among dup<2 {got[2]} (written {e})", got == e)
    s24_2 = [r for r in r24 if r["grade"] == 2]
    hi = [r for r in s24_2 if r["dup"] >= 2]
    lo = [r for r in s24_2 if r["dup"] < 2]
    check(f"R2400 G2 percentages written 27/67 = 40%, 65/113 = 58%",
          f"{100 * sum(1 for r in hi if r['rank'] == 1) / len(hi):.0f}" == "40" and f"{100 * sum(1 for r in lo if r['rank'] == 1) / len(lo):.0f}" == "58")
    s24_3 = [r for r in r24 if r["grade"] == 3]
    hi = [r for r in s24_3 if r["dup"] >= 2]
    lo = [r for r in s24_3 if r["dup"] < 2]
    check(f"R2400 G3 percentages written 33/87 = 38%, 75/93 = 81%",
          f"{100 * sum(1 for r in hi if r['rank'] == 1) / len(hi):.0f}" == "38" and f"{100 * sum(1 for r in lo if r['rank'] == 1) / len(lo):.0f}" == "81")

    print("\n-- within-arm truncation contrast")
    exp_w = {("R1200", 1): (32, 71, 46, 109), ("R1200", 2): (84, 132, 32, 48), ("R1200", 3): (90, 126, 34, 54),
             ("R2400", 1): (1, 1, 67, 179), ("R2400", 2): (12, 20, 80, 160), ("R2400", 3): (17, 27, 91, 153)}
    for (lab, g), e in exp_w.items():
        rw = r12 if lab == "R1200" else r24
        s = [r for r in rw if r["grade"] == g]
        t = [r for r in s if r["finish_reason"] == "length"]
        c = [r for r in s if r["finish_reason"] != "length"]
        got = (sum(1 for r in t if r["rank"] == 1), len(t), sum(1 for r in c if r["rank"] == 1), len(c))
        check(f"{lab} G{g}: cut top-1 {got[0]}/{got[1]}, complete {got[2]}/{got[3]} (written {e})", got == e)

    print("\n-- md5 of the pulled file (full digest must appear verbatim in the ADR)")
    import hashlib
    h = hashlib.md5((RES / "p35_fulltext" / "p35_race_2400.json").read_bytes()).hexdigest()
    check(f"full md5 {h} appears verbatim in the phase-B block", h in phase_b)

    print("\n-- open question 5: cross-budget M-d vs M-c (exploratory; M-c has no 2,400 arm)")
    all12 = json.loads((RES / "p35_fulltext" / "p35_race_1200.json").read_text(encoding="utf-8"))["rows"]
    p31all = json.loads((RES / "p31_confirm" / "p31_race.json").read_text(encoding="utf-8"))["rows"]

    def medx(rw):
        acc = defaultdict(list)
        for r in rw:
            acc[(r["mech"], r["grade"], r["puzzle"])].append(r["rank"])
        return {k: um(v) for k, v in acc.items()}

    mx12, mx24, mx31 = medx(all12), medx(r24), medx(p31all)
    exp_x = {  # grade -> {partner: (wins, losses, diff, p printed to 3 decimals)}
        2: {"both@1200": (15, 6, 9, "0.078"), "R1200c": (15, 13, 2, "0.851"), "P31c": (21, 10, 11, "0.071")},
        3: {"R1200c": (11, 16, -5, "0.442"), "P31c": (6, 17, -11, "0.035")},
    }
    for g, part in exp_x.items():
        for partner, (ew, el, ed, eps) in part.items():
            md = mx12 if partner == "both@1200" else mx24
            mc = {"both@1200": mx12, "R1200c": mx12, "P31c": mx31}[partner]
            w = sum(1 for p in pz if md[("d", g, p)] == 1 and mc[("c", g, p)] != 1)
            l = sum(1 for p in pz if mc[("c", g, p)] == 1 and md[("d", g, p)] != 1)
            d = sum(1 for p in pz if md[("d", g, p)] == 1) - sum(1 for p in pz if mc[("c", g, p)] == 1)
            p = sign_p(w, l)
            # the ADR prints p to 3 decimals
            check(f"G{g} M-d vs M-c, partner {partner}: {w}/{l}, diff {d:+d}, p={p:.4f} (written {ew}/{el}, {ed:+d}, {eps})",
                  (w, l, d) == (ew, el, ed) and f"{p:.3f}" == eps)
            s_w = f"{ew} 胜 {el} 负" if not (partner == "both@1200") else f"{ew} 胜 {el} 负"
            check(f"G{g} partner {partner}: the wins/losses string '{s_w}' appears in the ADR open-question block",
                  s_w in phase_b)
    check("G2 differences written as +9 / +2 / +11 puzzles", all(s in phase_b for s in ("差 +9 题", "差 **+2** 题", "差 **+11** 题")))
    check("G3 differences written as −5 / −11 puzzles and p=.035 flagged", "差 −5 题（p=.442）" in phase_b and "差 −11 题（**p=.035**）" in phase_b)
    # the logical point: the stage-2 gap shrinks against one M-c draw and grows against the other
    g2 = {k: v for k, v in exp_x[2].items()}
    check("the stage-2 lead claim is draw-dependent: gap +2 vs R1200's M-c (below +9) but +11 vs P31's M-c (above +9)",
          g2["R1200c"][2] < g2["both@1200"][2] < g2["P31c"][2])
    # top-3 recall, median caliber
    t3 = {}
    for g in (1, 2, 3):
        a = 100 * sum(1 for p in pz if m12[(g, p)] in (1, 2, 3)) / 60
        b = 100 * sum(1 for p in pz if m24[(g, p)] in (1, 2, 3)) / 60
        t3[g] = b - a
    got = tuple(f"{t3[g]:+.1f}" for g in (1, 2, 3))
    check(f"top-3 recall R2400-R1200 G1/G2/G3 = {got} pp (written -3.3 / +1.7 / +1.7)",
          all(abs(t3[g] - e) < 0.06 for g, e in zip((1, 2, 3), (-3.3, 1.7, 1.7))))
    check("top-3 differences appear in the ADR as −3.3 / +1.7 / +1.7pp", "−3.3 / +1.7 / +1.7pp" in phase_b)
    check("top-3 recall is not lower at every stage (so 'budget effect is top-1 only' is a fair reading)", not all(v < 0 for v in t3.values()))

    print(f"\n{len(fails)} FAIL" + (f": {fails}" if fails else " -- all checks pass"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
