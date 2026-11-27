#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit of every number written into the phase-B edits of paper/mechanism/main.tex (ADR-0034).

Side A = the sentence in the .tex source. Side B = the same quantity recomputed from the raw
JSON. Also asserts the things phase B was told NOT to touch are byte-identical to HEAD:
the 'Three resamples per cell' paragraph, Table 2, and the \\section{early} block.
"""
import json
import re
import subprocess
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
TEX = REPO / "paper" / "mechanism" / "main.tex"

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


def load(mt, mech=None):
    rows = json.loads((RES / "p35_fulltext" / f"p35_race_{mt}.json").read_text(encoding="utf-8"))["rows"]
    return [r for r in rows if mech is None or r["mech"] == mech]


def med(rows):
    """(mech, grade, puzzle) -> upper-median rank."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["mech"], r["grade"], r["puzzle"])].append(r["rank"])
    return {k: um(v) for k, v in acc.items()}


# Baseline for every "unchanged" assertion: the last mechanism-paper commit BEFORE phase B
# (stage-A edits + Lance's 'Three resamples per cell' decision). Pinned, not HEAD, so the
# audit keeps the same meaning after the phase-B commits land.
BASELINE = "bf8391a"


def main():
    tex = TEX.read_text(encoding="utf-8")
    head = subprocess.run(["git", "show", f"{BASELINE}:paper/mechanism/main.tex"], cwd=REPO, capture_output=True).stdout.decode("utf-8")
    assert len(head) > 1000, f"baseline {BASELINE} not readable"

    r12, r24 = load(1200, "d"), load(2400, "d")
    c12 = load(1200, "c")
    # m12 / m24 are M-d only, keyed (grade, puzzle) -- the R2400-vs-R1200 comparison is within M-d
    m12 = {k[1:]: v for k, v in med(r12).items()}
    m24 = {k[1:]: v for k, v in med(r24).items()}
    assert all(k[0] == "d" for k in med(r12)) and all(k[0] == "d" for k in med(r24))
    pz = sorted({k[1] for k in m24})

    print(f"== untouched regions are byte-identical to baseline {BASELINE} (last commit before phase B) ==")
    def block(s, start, end):
        i = s.index(start)
        return s[i:s.index(end, i)]
    a = block(head, r"\textbf{Three resamples per cell.}", r"\textbf{Unequal output budget.}")
    b = block(tex, r"\textbf{Three resamples per cell.}", r"\textbf{Unequal output budget.}")
    check("'Three resamples per cell' paragraph unchanged", a == b and len(a) > 500)
    a = block(head, r"\begin{table}", r"\end{table}")
    b = block(tex, r"\begin{table}", r"\end{table}")
    check(f"Table 2 (first table environment, {len(a)} chars) unchanged", a == b)
    # Lance (2026-11-27 10:32) authorised ONE addition to sec:early: a budget qualifier appended
    # after 'from the opposite direction.'. The blanket byte-identical check over the whole
    # span is therefore replaced by three finer ones that keep the same protection:
    # Table 2 unchanged, Fig. 1 unchanged, and sec:early unchanged APART FROM that appended
    # sentence (original text must be a prefix, and the three stage numbers must still be there).
    a = block(head, r"\section{Self-critique decomposed", r"\section{The measurement artifact}")
    b = block(tex, r"\section{Self-critique decomposed", r"\section{The measurement artifact}")
    anchor = "from the opposite direction."
    assert a.count(anchor) == 1 and b.count(anchor) == 1
    orig, new = a[:a.index(anchor) + len(anchor)], b[:b.index(anchor) + len(anchor)]
    appended = b[b.index(anchor) + len(anchor):].strip()
    # compare the opening words with whitespace normalised (the qualifier wraps lines in the source)
    opening_ok = " ".join(appended.split()).startswith("One condition on these stage-resolved numbers:")
    # and lock the tail + length, so nothing can be smuggled INTO the authorised qualifier either
    tail_ok = " ".join(appended.split()).endswith("the budget's effect on it is not resolved.")
    check(f"sec:early original text is a byte-identical prefix of the current text ({len(orig)} chars); "
          f"exactly one qualifier appended ({len(appended)} chars) starting with 'One condition on these stage-resolved numbers:'",
          orig == new and len(orig) > 1000 and opening_ok)
    check(f"the authorised qualifier is exactly the authorised sentence: 430 <= len ({len(appended)}) <= 470 chars, "
          f"single paragraph, ends with 'the budget's effect on it is not resolved.'",
          430 <= len(appended) <= 470 and "\n\n" not in appended and tail_ok)
    for s in (r"\Mcrit{} $43/90$ vs.\ \Mmaint{} $30/90$", "paired 20 wins / 7 losses, sign test $p = 0.019$",
              r"$63/90$ vs.\ $37/90$; 32 wins / 6 losses", r"$69/90$ vs.\ $66/90$; 16 / 13, $p = 0.71$",
              "$+17$\,pp", "(hard), $-16$\,pp (medium), $-9$\,pp (easy)"):
        check(f"sec:early stage number still present verbatim: {s[:48]}", s in b)
    # numbers inside the qualifier must match the phase-B Limitations block and the JSON
    for s in ("61\\%", "-1.7$", "-11.7$", "-13.3$", "1.00$", "0.21$", "0.12$"):
        check(f"qualifier number matches Limitations / phase B: {s}", s in appended)
    # Fig. 1 figure environment must be untouched
    a = block(head, r"\begin{figure}", r"\end{figure}")
    b = block(tex, r"\begin{figure}", r"\end{figure}")
    check(f"Fig. 1 figure environment unchanged ({len(a)} chars)", a == b and len(a) > 200)
    # everything else in the old span (from first \section up to sec:early) is still blanket-checked
    a = head[head.index(r"\section{"):head.index(r"\section{Self-critique decomposed")]
    b = tex[tex.index(r"\section{"):tex.index(r"\section{Self-critique decomposed")]
    check(f"everything from the first \\section up to sec:early unchanged ({len(a)} chars; covers intro, Table 2)", a == b)
    a = head[head.index(r"\section{Discussion}"):head.index(r"\section{Limitations}")]
    b = tex[tex.index(r"\section{Discussion}"):tex.index(r"\section{Limitations}")]
    check("Discussion section unchanged", a == b)
    a = head[head.index(r"\section{Conclusion}"):head.index(r"\section*{Reproducibility statement}")]
    b = tex[tex.index(r"\section{Conclusion}"):tex.index(r"\section*{Reproducibility statement}")]
    check("Conclusion section unchanged", a == b)
    a = head[head.index(r"\section*{Ethics statement}"):]
    b = tex[tex.index(r"\section*{Ethics statement}"):]
    check("Ethics / AI-tools / everything after Reproducibility unchanged", a == b)
    a = block(head, r"\paragraph{Residual truncation, disclosed.}", r"\paragraph{Scorer tolerance, verified.}")
    b = block(tex, r"\paragraph{Residual truncation, disclosed.}", r"\paragraph{Scorer tolerance, verified.}")
    # phase A ended the paragraph with 'critique pass itself.'; phase B may only append after it.
    anchor = "critique pass itself."
    appended = b[b.index(anchor) + len(anchor):]
    kept = b[:b.index(anchor) + len(anchor)]
    check("phase-A text of 'Residual truncation' kept verbatim against the baseline "
          f"(kept {len(kept)} chars == baseline {len(a.rstrip())} chars); phase B appended {len(appended.strip())} chars after 'critique pass itself.'",
          a.count(anchor) == 1 and kept == a.rstrip() and len(appended.strip()) > 0)
    a = block(head, r"\paragraph{Scorer tolerance, verified.}", r"\section{Discussion}")
    b = block(tex, r"\paragraph{Scorer tolerance, verified.}", r"\section{Discussion}")
    check("'Scorer tolerance' paragraph unchanged", a == b)

    print("\n== Limitations, 'Unequal output budget' ==")
    lim = block(tex, r"\textbf{Unequal output budget.}", r"\section{Conclusion}")
    check("contradiction removed: no 'were not retained' / 'not retained to repeat' in Limitations",
          "not retained" not in block(tex, r"\section{Limitations}", r"\section{Conclusion}"))
    check("no full-text contradiction anywhere in the paper: every 'retained no full texts' is about the ORIGINAL batch",
          tex.count("retained no full texts") == 1)
    check("no template sentence claiming 'unchanged at every grade'", "unchanged at every grade" not in tex)
    check("no claim of verified harmlessness: 'verified' not used for the truncation",
          not re.search(r"truncation[^.]*\bverified\b", tex))
    # the paper must not claim the cap removed truncation. 'without eliminating it' is a negation and is fine.
    positive_claims = [m.group(0) for m in re.finditer(r"[^.]*\beliminat\w*[^.]*\.", lim)]
    bad_elim = [c for c in positive_claims if not re.search(r"without eliminating|not eliminat|never eliminat", c)]
    check(f"no sentence claims truncation was eliminated (sentences mentioning 'eliminat': {len(positive_claims)}, offending: {len(bad_elim)})",
          not bad_elim)
    check("residual truncation is stated in the Limitations sentence ('9\\% without eliminating it (15\\% at stage 3)')",
          "truncation rate to 9\\% without eliminating it (15\\% at stage 3)" in lim)

    # 58% batch-1, under 8% others: phase-A / earlier text, re-derived from the batch-1 archive
    b1 = json.loads((RES / "p25_reveal" / "p25_race_v2.json").read_text(encoding="utf-8"))["rows"]
    TERM = set("。！？.!)）**\"'`")

    def heur(t):
        t = t.rstrip()
        return len(t) > 800 and t[-1] not in TERM
    rates = {m: 100 * sum(1 for r in b1 if r["mech"] == m and heur(r["text"])) / sum(1 for r in b1 if r["mech"] == m) for m in "dcab"}
    check(f"batch-1 heuristic truncation: critique {rates['d']:.1f}% -> '58%', others max {max(rates['c'], rates['a'], rates['b']):.1f}% -> 'under 8%'",
          round(rates["d"]) == 58 and max(rates["c"], rates["a"], rates["b"]) < 8)
    all12 = load(1200)
    d_all = [r for r in all12 if r["mech"] == "d"]
    n61 = 100 * sum(1 for r in d_all if r["finish_reason"] == "length") / len(d_all)
    check(f"re-run truncation for critique {n61:.1f}% -> '61%'", round(n61) == 61 and "61\\%" in lim)

    cut24 = sum(1 for r in r24 if r["finish_reason"] == "length")
    check(f"relaxed arm truncation {cut24}/540 = {100 * cut24 / 540:.1f}% -> '9%'", round(100 * cut24 / 540) == 9)
    g3 = 100 * sum(1 for r in r24 if r["grade"] == 3 and r["finish_reason"] == "length") / 180
    check(f"stage-3 residual truncation {g3:.1f}% -> '15%'", round(g3) == 15 and "(15\\% at stage 3)" in lim)

    exp = {1: ("-1.7", "1.00"), 2: ("-11.7", "0.21"), 3: ("-13.3", "0.12")}
    for g in (1, 2, 3):
        a_ = sum(1 for p in pz if m12[(g, p)] == 1)
        b_ = sum(1 for p in pz if m24[(g, p)] == 1)
        d = 100 * (b_ - a_) / 60
        w = sum(1 for p in pz if m24[(g, p)] == 1 and m12[(g, p)] != 1)
        l = sum(1 for p in pz if m12[(g, p)] == 1 and m24[(g, p)] != 1)
        p = sign_p(w, l)
        check(f"stage {g}: diff {d:+.1f} pp, sign p = {p:.2f} (written {exp[g][0]}, {exp[g][1]})",
              f"{d:+.1f}" == exp[g][0] and f"{p:.2f}" == exp[g][1])
    check("differences printed in order $-1.7$, $-11.7$, $-13.3$ at stages 1/2/3",
          "$-1.7$, $-11.7$,\nand $-13.3$\\,pp at stages 1/2/3" in lim)
    check("p-values printed in order $p = 1.00$, $0.21$, $0.12$", "$p = 1.00$, $0.21$, and $0.12$" in lim)

    import random
    exp_ci = {2: ("[-26.7, +3.3]", "-26.7", "+3.3"), 3: ("[-26.7, +1.7]", "-26.7", "+1.7")}
    for g in (2, 3):
        rng = random.Random(0)
        ds = []
        for _ in range(20000):
            s = [rng.choice(pz) for _ in pz]
            ds.append((sum(1 for p in s if m24[(g, p)] == 1) - sum(1 for p in s if m12[(g, p)] == 1)) / 60)
        ds.sort()
        lo, hi = 100 * ds[500], 100 * ds[19500]
        s_lo, s_hi = f"{lo:+.1f}", f"{hi:+.1f}"
        check(f"stage {g} paired bootstrap 95% CI [{s_lo}, {s_hi}] (written {exp_ci[g][1]}, {exp_ci[g][2]})",
              (s_lo, s_hi) == exp_ci[g][1:])
        check(f"stage {g} CI does NOT exclude a 10 pp effect (statement in paper)", not (lo > -10 and hi < 10))
    check("CI text: $[-26.7, +3.3]$ and $[-26.7, +1.7]$ present",
          "$[-26.7, +3.3]$ and $[-26.7, +1.7]$" in lim)
    check("'stage-2 and stage-3 differences are larger than 10pp' is true",
          all(abs(100 * (sum(1 for p in pz if m24[(g, p)] == 1) - sum(1 for p in pz if m12[(g, p)] == 1)) / 60) > 10 for g in (2, 3)))
    nd = {g: sum(1 for p in pz if (m24[(g, p)] == 1) != (m12[(g, p)] == 1)) for g in (1, 2, 3)}
    check(f"discordant puzzles per stage {nd} -> 'about 20'", all(18 <= v <= 23 for v in nd.values()))

    dm = {g: (sum(r["dup"] for r in r12 if r["grade"] == g) / 180, sum(r["dup"] for r in r24 if r["grade"] == g) / 180) for g in (1, 2, 3)}
    check(f"dup stage 2 {dm[2][0]:.2f} vs {dm[2][1]:.2f}; stage 3 {dm[3][0]:.2f} vs {dm[3][1]:.2f} (written 0.73/1.38, 0.96/1.89)",
          (f"{dm[2][0]:.2f}", f"{dm[2][1]:.2f}", f"{dm[3][0]:.2f}", f"{dm[3][1]:.2f}") == ("0.73", "1.38", "0.96", "1.89"))
    check("dup sentence present verbatim", "0.73 vs.\\\n1.38 at stage 2, 0.96 vs.\\ 1.89 at stage 3" in lim)

    print("\n== direction statement ==")
    # 'On top-1 the relaxed arm scored critique no higher at any stage'
    hi_med = [sum(1 for p in pz if m24[(g, p)] == 1) - sum(1 for p in pz if m12[(g, p)] == 1) for g in (1, 2, 3)]
    lo_med = [100 * sum(1 for p in pz if m24[(g, p)] in (1, 2, 3)) / 60 - 100 * sum(1 for p in pz if m12[(g, p)] in (1, 2, 3)) / 60 for g in (1, 2, 3)]
    hi_row = [sum(1 for r in r24 if r["grade"] == g and r["rank"] == 1) - sum(1 for r in r12 if r["grade"] == g and r["rank"] == 1) for g in (1, 2, 3)]
    print(f"    top-1 count R2400-R1200 by stage (median caliber): {hi_med}; row-level: {hi_row}; top-3 recall pp: {[round(x, 1) for x in lo_med]}")
    check("'scored critique no higher at any stage' holds on top-1 in BOTH calibers", all(v <= 0 for v in hi_med) and all(v <= 0 for v in hi_row))
    check("statement is scoped to top-1: 'On top-1 the relaxed arm' / 'scored \\\\Mcrit{} no higher at any stage'",
          "On top-1 the relaxed arm\nscored \\Mcrit{} no higher at any stage" in lim)
    check("top-3 recall is NOT lower at every stage, so scoping the sentence to top-1 is necessary",
          not all(v < 0 for v in lo_med))
    # the paper must NOT claim the stage-2 lead shrinks: that depends on which M-c draw is the partner
    check("no claim that the cap inflates the stage-2 LEAD (the cross-budget comparison is draw-dependent)",
          "inflates it" not in lim and "therefore a figure" not in lim and "upper bound" not in lim)
    check("says only critique was re-run at the larger budget", "only \\Mcrit{} was re-run at the larger\nbudget" in lim)
    # three-key maps: (mech, grade, puzzle)
    m12x, m24x = med(r12), med(r24)
    m31x = med([r for r in json.loads((RES / "p31_confirm" / "p31_race.json").read_text(encoding="utf-8"))["rows"] if r["mech"] in "cd"])
    c12x = med(c12)
    g = 2
    gap_12 = sum(1 for p in pz if m12x[("d", g, p)] == 1) - sum(1 for p in pz if c12x[("c", g, p)] == 1)
    gap_rel_r12c = sum(1 for p in pz if m24x[("d", g, p)] == 1) - sum(1 for p in pz if c12x[("c", g, p)] == 1)
    gap_rel_p31c = sum(1 for p in pz if m24x[("d", g, p)] == 1) - sum(1 for p in pz if m31x[("c", g, p)] == 1)
    print(f"    stage-2 M-d minus M-c (top-1 puzzles of 60): both@1200 {gap_12:+d}; relaxed vs R1200 M-c {gap_rel_r12c:+d}; relaxed vs P31 M-c {gap_rel_p31c:+d}")
    check("why the lead claim is forbidden: the relaxed stage-2 gap shrinks against one M-c draw and grows against the other",
          (gap_rel_r12c - gap_12) < 0 < (gap_rel_p31c - gap_12))
    # 'stage-2 lead' exists in the paper
    early = block(tex, r"\section{", r"\section{The measurement artifact}")
    check("'stage-2 lead' is a real quantity in sec:early (critique above maintenance at stage 2 in the preregistered draw)",
          "73\\%" in early and "87\\%" in early)

    print("\n== measurement artifact section, appended sentences ==")
    art = block(tex, r"\paragraph{Residual truncation, disclosed.}", r"\paragraph{Scorer tolerance, verified.}")
    app = art[art.index("A further arm raised the cap"):]
    check("appended text reports 2,400 tokens", "2{,}400 tokens" in app)
    check("appended text reports 9% and 15% at stage 3", "lowered the cut rate to 9\\% (15\\% at stage 3)" in app)
    check("appended text reports the three differences", "$-1.7$, $-11.7$, and $-13.3$\\,pp at stages 1/2/3" in app)
    check("appended text reports the three p-values", "$p = 1.00$, $0.21$, $0.12$" in app)
    check("appended text neither declares harmlessness nor a confound",
          "not a demonstrated\nbehavioral confound" in app and "nor is it demonstrated to be harmless" in app)
    check("appended text cross-references Limitations / Unequal output budget",
          "\\S\\ref{sec:limitations}, \\emph{Unequal output budget}" in app)
    check("'only critique' was re-run at 2,400 (R2400 is M-d only)", {r["mech"] for r in load(2400)} == {"d"} and "for\n\\Mcrit{} alone" in app)

    print("\n== Reproducibility statement ==")
    rep = block(tex, r"\section*{Reproducibility statement}", r"\section*{Ethics statement}")
    check("ADR-0034 added to the preregistered list", "under ADR-0034" in rep)
    check("ADR-0033 half-sentence intact", "the distribution-intervention batch under\nADR-0033 (three arms, three seeds, reversal gates fixed before launch)" in rep)
    check("re-run released in results/p35_fulltext", "results/p35\\_fulltext" in rep)
    check("2,400-token arm mentioned as released", "2{,}400-token \\Mcrit{} arm" in rep)
    check("'full texts archived' statement present (phase A, kept)", "archived them" in rep)

    print("\n== build ==")
    log = Path("/tmp/p35b_tex3.log")
    if log.exists():
        L = log.read_text(encoding="utf-8", errors="replace")
        check("0 TeX errors", len(re.findall(r"^!", L, flags=re.M)) == 0)
        check("0 undefined references/citations", not re.search(r"undefined (references?|citations?)", L, flags=re.I))
        check("0 overfull hboxes", len(re.findall(r"^Overfull \\hbox", L, flags=re.M)) == 0)
        m = re.search(r"Output written on main\.pdf \((\d+) pages", L)
        print(f"    pages: {m.group(1) if m else '?'}")
    else:
        print("    build log not found at /tmp/p35b_tex3.log; build checks skipped")

    print(f"\n{len(fails)} FAIL" + (f": {fails}" if fails else " -- all checks pass"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
