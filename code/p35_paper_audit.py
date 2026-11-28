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
    # Lance's 2026-11-27 P37 read-out (ADR-0036 sec 12; instructions paper/p37_judge_prompt.md sec 6)
    # authorises ONE new Limitations entry, placed right after this paragraph. The end anchor is
    # therefore moved from 'Unequal output budget' to the new entry's own start, so the protection
    # of the original paragraph is kept exactly (it is still compared byte-for-byte with the baseline).
    NEW_ENTRY = r"\textbf{Output-format sampling noise.}"
    # Defensive: if the entry is missing, record a FAIL and fall back to the old end anchor, so the
    # protected paragraph is still compared and the audit keeps running instead of crashing.
    _i_tr, _i_ne, _i_uo = tex.find(r"\textbf{Three resamples per cell.}"), tex.find(NEW_ENTRY), tex.find(r"\textbf{Unequal output budget.}")
    check("the new entry appears exactly once and sits between 'Three resamples per cell' and 'Unequal output budget', in this order",
          tex.count(NEW_ENTRY) == 1 and 0 <= _i_tr < _i_ne < _i_uo)
    a = block(head, r"\textbf{Three resamples per cell.}", r"\textbf{Unequal output budget.}")
    b = block(tex, r"\textbf{Three resamples per cell.}", NEW_ENTRY if _i_ne > _i_tr else r"\textbf{Unequal output budget.}")
    check("'Three resamples per cell' paragraph unchanged (end anchor now = the authorised new entry)", a == b and len(a) > 500)
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


    print("\n== ADR-0035 (P36): where the truncated critique output was scored ==")
    ro = json.loads((RES / "p36_commit" / "readout.json").read_text(encoding="utf-8"))
    J = ro["judgements"]
    # the judgements file itself must still merge cleanly and reproduce the read-out
    r2 = subprocess.run([sys.executable, str(REPO / "code" / "p36_judge.py")], cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    check("p36_judge.py reruns cleanly and reports R-C1", r2.returncode == 0 and "VERDICT: R-C1" in r2.stdout)
    # independent recount straight from the jsonl (no reuse of p36_judge's numbers)
    jl = [json.loads(l) for l in (RES / "p36_commit" / "judgments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    key = json.loads((RES / "p36_commit" / "key.json").read_text(encoding="utf-8"))
    for j in jl: j.update(key[j["id"]])
    j12 = [j for j in jl if j["id"].startswith("R1200-")]; j24 = [j for j in jl if j["id"].startswith("R2400-")]
    n12 = sum(1 for j in j12 if j["verdict"] == "N"); n24 = sum(1 for j in j24 if j["verdict"] == "N")
    t1n = sum(1 for j in j12 if j["verdict"] == "N" and j["rank"] == 1); t1c = sum(1 for j in j12 if j["verdict"] == "C" and j["rank"] == 1)
    u = __import__("re")
    by = {k: sum(1 for j in j12 if j["verdict"] == "N" and u.match(r"U(\d)", j["rule"]).group(1) == k[-1]) for k in ("U1", "U3", "U4")}
    u24_4 = sum(1 for j in j24 if j["verdict"] == "N" and u.match(r"U(\d)", j["rule"]).group(1) == "4")
    check(f"sample sizes frozen (60 / 30), judged blind, ids unique", len(j12) == 60 and len(j24) == 30 and len({j['id'] for j in jl}) == 90)
    check(f"no-commitment R1200 = {n12}/60 = {n12/60:.1%}", n12 == 29)
    check(f"no-commitment R2400 = {n24}/30 = {n24/30:.1%}", n24 == 14)
    check(f"no-commitment by rule U1/U3/U4 = {by['U1']}/{by['U3']}/{by['U4']}", by == {"U1": 16, "U3": 3, "U4": 10})
    check(f"rank==1 among no-commitment rows = {t1n}/{n12} = {t1n/n12:.0%}", t1n == 23)
    check(f"rank==1 rows without commitment = {t1n}/{t1n+t1c} = {t1n/(t1n+t1c):.0%}", t1n + t1c == 37)
    check(f"R2400: incomplete-list share of no-commitment = {u24_4}/{n24} = '{u24_4/n24:.0%}'", u24_4 * 2 == n24)
    # the paper text
    # Look up the correction block defensively: if it is missing the audit must record a FAIL and go on,
    # not crash (a crash would skip every later check and look like 'nothing failed' to a log reader).
    ci = tex.find(r"\emph{Correction (ADR-0035).}")
    check("correction block present in the artifact section", ci >= 0)
    corr = tex[ci:tex.find(r"\paragraph{Scorer tolerance, verified.}", ci)] if ci >= 0 else ""
    art = block(tex, r"\paragraph{Residual truncation, disclosed.}", r"\paragraph{Scorer tolerance, verified.}")
    for lit in ("48\\% (29/60; Wilson 95\\% CI 36--61\\%)", "in 16 the", "in 3 it ranks", "in 10 the list",
                "23 of the 29 no-commitment rows carry rank 1", "62\\% of\nthe rank-1 rows", "47\\% (14/30)"):
        check(f"correction states: {lit!r}", lit in corr)
    check("the withdrawn claim is named explicitly", 'that the cut ``does\nnot reach the scored list\'\' was inferred' in corr or "does\nnot reach the scored list'' was inferred" in corr)
    check("withdrawal sentence present", "withdraw the claim that truncation spares the scored list for \\Mcrit{}" in corr)
    check("no unverified claim about M-maint's cuts (the read covered critique only)",
          "was not judged, and the claim is\nnot made for it either" in corr and "nearly every case" not in corr)
    check("Limitations no longer says ranks were unaffected", "Ranks were unaffected" not in block(tex, r"\textbf{Unequal output budget.}", r"\section{Conclusion}"))
    check("Limitations points to the correction and says alike scores != same scored position",
          "alike scores do not\nmean the scored position was the same" in tex)
    # the original (wrong) sentence is still there verbatim, immediately followed by the correction -> history is visible
    check("original sentence kept verbatim (history visible), correction appended in the same paragraph",
          "The cut does not\nreach the scored list---\\Mcrit{} writes its revised candidates before\nits closing commentary---" in art
          and art.find("Correction (ADR-0035)") > art.find("closing commentary") >= 0)

    # Simpson guard: the pooled truncated-minus-complete top-1 gap for critique (+9.5 pp) DISAPPEARS once
    # stratified by grade. The paper must not cite the pooled gap as a truncation effect, and the
    # correction paragraph must not claim that truncation shifts critique's top-1 in either direction.
    print("\n== ADR-0035: pooled truncated-vs-complete gap is a Simpson artefact ==")
    r12d = [r for r in json.loads((RES / "p35_fulltext" / "p35_race_1200.json").read_text(encoding="utf-8"))["rows"] if r["mech"] == "d"]
    def _t1(rs): return sum(1 for r in rs if r["rank"] == 1) / len(rs)
    pooled = _t1([r for r in r12d if r["finish_reason"] == "length"]) - _t1([r for r in r12d if r["finish_reason"] == "stop"])
    strat = {g: _t1([r for r in r12d if r["grade"] == g and r["finish_reason"] == "length"]) - _t1([r for r in r12d if r["grade"] == g and r["finish_reason"] == "stop"]) for g in (1, 2, 3)}
    check(f"pooled gap = {pooled*100:+.1f} pp but within-grade gaps are small: " + ", ".join(f"G{g} {v*100:+.1f}" for g, v in strat.items()),
          0.08 < pooled < 0.11 and all(abs(v) < 0.10 for v in strat.values()))
    check("paper does not claim truncation raised (or lowered) critique's top-1",
          not re.search(r"truncat\w*[^.]*\b(raised|lowered|inflated|shifted)\b[^.]*top-1", tex, flags=re.I))
    check("paper does not cite the pooled 63% vs 53% comparison", "63\\% vs.\\ 53\\%" not in tex and "63\\% vs\\ 53\\%" not in tex)
    check("ADR-0035 sec 7 records the withdrawal of the pooled-gap inference",
          "辛普森" in (REPO / "paper" / "adr_0035.md").read_text(encoding="utf-8"))

    print("\n== ADR-0036 (P37): the 'Output-format sampling noise' Limitations entry ==")
    D37 = RES / "p37_stability"
    A37 = {"T0": "p37_c_t00_r3.json", "T7": "p37_c_t07_r3.json", "T7b": "p37_b_t07_r3.json"}
    R37 = {a: json.loads((D37 / f).read_text(encoding="utf-8"))["rows"] for a, f in A37.items()}
    for a, rows in R37.items():
        keys = {(r["puzzle"], r["grade"], r["rep"]) for r in rows}
        check(f"{a}: 540 rows, 540 unique cells, no API errors", len(rows) == 540 and len(keys) == 540 and not any(r["finish_reason"] == "error" for r in rows))

    def cells(rows, key):
        o = defaultdict(dict)
        for r in rows:
            o[(r["puzzle"], r["grade"])][r["rep"]] = r[key]
        return o

    def flip3(rows, key):
        c = cells(rows, key)
        return [sum(1 for (p, g), v in c.items() if g == gg and len({v[k] == 1 for k in v}) > 1) for gg in (1, 2, 3)]

    f0 = flip3(R37["T0"], "rank"); f7 = flip3(R37["T7"], "rank"); fb = flip3(R37["T7b"], "rank"); t3 = flip3(R37["T7"], "top3")
    check(f"T0 top-1 flips at stages 1/2/3 = {f0} (entry: 0/60 each)", f0 == [0, 0, 0])
    check(f"T0: the three outputs of every cell are byte-identical (180/180)",
          sum(1 for v in cells(R37["T0"], "text").values() if len(set(v.values())) == 1) == 180)
    check(f"T7 top-1 flips = {f7} -> {sum(f7[:1])}/60, {f7[1]}/60, {f7[2]}/60 = {f7[0]/60:.0%}/{f7[1]/60:.0%}/{f7[2]/60:.0%} (entry: 32/63/62%)",
          f7 == [19, 38, 37] and [round(100 * x / 60) for x in f7] == [32, 63, 62])
    check(f"T7-b top-1 flips = {fb} -> {[round(100 * x / 60) for x in fb]} (entry: 17/28/15%)",
          fb == [10, 17, 9] and [round(100 * x / 60) for x in fb] == [17, 28, 15])
    check(f"T7 top-3 flips = {t3} -> {[round(100 * x / 60) for x in t3]} (entry: 25/18/3%)",
          t3 == [15, 11, 2] and [round(100 * x / 60) for x in t3] == [25, 18, 3])
    pooled3 = sum(t3) / 180
    check(f"T7 top-3 flips pooled = {sum(t3)}/180 = {pooled3:.1%} (entry: 15.6%, gate 15% not met)", f"{pooled3:.1%}" == "15.6%" and pooled3 > 0.15)
    # every top-3 flip is gold LEAVING the list (a 0 among the three ranks), never a move inside it
    c7 = cells(R37["T7"], "top3"); r7 = cells(R37["T7"], "rank")
    fl = [(k, v) for k, v in c7.items() if len(set(v.values())) > 1]
    check(f"all {len(fl)} top-3 flips are gold leaving the list (rank 0 present in each)",
          len(fl) == 28 and all(0 in r7[k].values() for k, _ in fl))
    # majority vote (descriptive): mode of T7's k draws vs the T0 reading, top-1
    def agree(k):
        agree_ = n = 0
        for key, v in cells(R37["T7"], "rank").items():
            truth = 1 if 1 in cells(R37["T0"], "rank")[key].values() else 0
            draws = [int(v[i] == 1) for i in range(k)]
            pred = max(set(draws), key=draws.count) if len(set(draws)) != len(draws) else draws[0]
            agree_ += pred == truth; n += 1
        return agree_ / n
    a1, a3 = agree(1), agree(3)
    check(f"majority vote agreement with the temperature-0 reading: k=1 {a1:.0%}, k=3 {a3:.0%} (entry: 64% and 64%)",
          f"{a1:.0%}" == "64%" and f"{a3:.0%}" == "64%")
    # set level agreement between the two temperatures: median-rank top-1 differs by <= 5 pp at every stage
    def med_top1(rows, g):
        return sum(1 for (p, gg), v in cells(rows, "rank").items() if gg == g and sorted(v.values())[1] == 1)
    diffs = [med_top1(R37["T7"], g) - med_top1(R37["T0"], g) for g in (1, 2, 3)]
    check(f"set-level top-1, T7 minus T0, in puzzles = {diffs} (entry: 'within five points at every stage')",
          all(abs(d) <= 3 for d in diffs))
    # the frozen gates, recomputed
    check("S-1 PASS (T0 flip(3) <= 10%)", sum(f0) / 180 <= 0.10)
    check("S-2 PASS (>=30 pp pooled, same sign per stage)", sum(f7) / 180 - sum(f0) / 180 >= 0.30 and all(x > y for x, y in zip(f7, f0)))
    check("S-3 PASS (>=20 pp pooled, same sign per stage)", sum(f7) / 180 - sum(fb) / 180 >= 0.20 and all(x > y for x, y in zip(f7, fb)))
    check("S-4' NOT DETECTED as frozen (pooled <=15% AND every stage <=20%)", not (pooled3 <= 0.15 and all(x / 60 <= 0.20 for x in t3)))
    # the entry's own words
    # Defensive: an empty entry text makes every literal check FAIL explicitly (no exception, no skipped checks).
    e0, e1 = tex.find(NEW_ENTRY), tex.find(r"\textbf{Unequal output budget.}")
    ent = " ".join(tex[e0:e1].split()) if 0 <= e0 < e1 else ""
    for lit in ("top-1 of 32\\%, 63\\%, and 62\\% of puzzles differed", "17\\%, 28\\%, and 15\\% for the single-answer mechanism",
                "it changed for 25\\%, 18\\%, and 3\\% of puzzles, always by the gold answer leaving the list altogether",
                "A majority vote over three resamples did not help (agreement with the temperature-0 reading 64\\% for one draw and for three)",
                "was not met (15.6\\%)", "agree to within five points at every stage"):
        check(f"entry states: {lit[:70]!r}", lit in ent)
    check("entry does not claim top-3 is noise-free", "top-3 recall is not immune" in ent.lower() and "noise-free" not in ent.lower())
    check("entry does not claim majority voting helps", "majority vote over three resamples did not help" in ent.lower())

    print("\n== ADR-0036/0037 addendum: the 'Stage 1 cannot be decided' sentences appended to the sampling-noise entry ==")
    # every number re-derived from the raw JSON with no shared helper beyond sign_p
    t3 = lambda r: int(r["rank"] in (1, 2, 3))
    A0 = json.loads((RES / "p38_stateless" / "p38_a_t00_s0r3.json").read_text(encoding="utf-8"))["rows"]
    C0 = json.loads((RES / "p37_stability" / "p37_c_t00_r3.json").read_text(encoding="utf-8"))["rows"]
    C7 = json.loads((RES / "p37_stability" / "p37_c_t07_r3.json").read_text(encoding="utf-8"))["rows"]
    P31 = [r for r in json.loads((RES / "p31_confirm" / "p31_race.json").read_text(encoding="utf-8"))["rows"] if r["mech"] == "a"]
    R12 = [r for r in json.loads((RES / "p35_fulltext" / "p35_race_1200.json").read_text(encoding="utf-8"))["rows"] if r["mech"] == "a"]
    B38 = json.loads((RES / "p38_stateless" / "p38_a_t07_s3r4.json").read_text(encoding="utf-8"))["rows"]
    B1 = json.loads((RES / "p25_reveal" / "p25_race_v2.json").read_text(encoding="utf-8"))["rows"]
    a0 = {(r["puzzle"], r["grade"]): t3(r) for r in A0 if r["rep"] == 0}
    c0 = {(r["puzzle"], r["grade"]): r["top3"] for r in C0 if r["rep"] == 0}
    def cellset(rows, key):
        o = defaultdict(dict)
        for r in rows: o[(r["puzzle"], r["grade"])][r["rep"]] = r[key]
        return o
    def med3(o): return {k: int(sorted(v.values())[1] == 1) for k, v in o.items()}
    c7 = med3(cellset(C7, "top3")); a7 = med3(cellset(P31, "rank"))
    pool = defaultdict(dict)
    for r in P31: pool[(r["puzzle"], r["grade"])][r["rep"]] = t3(r)
    for r in R12: pool[(r["puzzle"], r["grade"])][3 + r["rep"]] = t3(r)
    for r in B38: pool[(r["puzzle"], r["grade"])][6 + (r["rep"] - 3)] = t3(r)
    maj = {k: int(sum(v.values()) >= 6) for k, v in pool.items()}
    def four(c, a, g):
        ks = [k for k in c if k[1] == g and k in a]
        both = sum(1 for k in ks if c[k] and a[k]); nei = sum(1 for k in ks if not c[k] and not a[k])
        w = sum(1 for k in ks if c[k] and not a[k]); l = sum(1 for k in ks if a[k] and not c[k])
        return len(ks), both, nei, w, l
    n, both, nei, w, l = four(c0, a0, 1)
    check(f"stage 1 at temp 0: same membership on {both+nei} of {n} (both {both}, neither {nei}); {w} favour maintenance, {l} favour stateless = +{round((w-l)*100/60)} points; p={sign_p(w, l):.2f}",
          (n, both, nei, w, l) == (60, 25, 28, 5, 2) and round((w - l) * 100 / 60) == 5 and f"{sign_p(w, l):.2f}" == "0.45")
    n2, _, _, w2, l2 = four(c0, a0, 2)
    pts = lambda x: round(x * 100 / 60)          # puzzles -> percentage points on 60
    check(f"stage 2 at temp 0: {w2+l2} of {n2} discordant, +{w2-l2} puzzles = +{pts(w2-l2)} points, p<0.001",
          (n2, w2, l2) == (60, 27, 1) and pts(w2 - l2) == 43 and sign_p(w2, l2) < 0.001)
    # temperature movement of each mechanism's set-level reading, per stage
    def lvl(d, g): return sum(v for k, v in d.items() if k[1] == g)
    dmv_a = [lvl(maj, g) - lvl(a0, g) for g in (1, 2, 3)]
    dmv_c = [lvl(c7, g) - lvl(c0, g) for g in (1, 2, 3)]
    dmv_a_pts = [pts(x) for x in dmv_a]; dmv_c_pts = [pts(x) for x in dmv_c]
    check(f"stateless set-level reading, 0.7 minus 0: {dmv_a} puzzles = {dmv_a_pts} points at stages 1/2/3 (entry: -3, -2, -8)", dmv_a_pts == [-3, -2, -8])
    check(f"maintenance set-level reading, 0.7 minus 0: {dmv_c} puzzles = {dmv_c_pts} points (entry: at most 2 in absolute value, and it reaches 2)",
          all(abs(x) <= 2 for x in dmv_c_pts) and max(abs(x) for x in dmv_c_pts) == 2)
    g0 = pts(lvl(c0, 1) - lvl(a0, 1)); g7 = pts(lvl(c7, 1) - lvl(a7, 1))
    check(f"stage-1 gain at temp 0 = {g0} points and at 0.7 (median-3) = {g7} points (entry: +5 and +8)", (g0, g7) == (5, 8))
    # batch 1 at stage 1: both at temp 0, single draw
    b1 = defaultdict(dict)
    for r in B1: b1[(r["puzzle"], r["grade"])][r["mech"]] = t3(r)
    ks = [k for k in b1 if k[1] == 1 and "a" in b1[k] and "c" in b1[k]]
    w1 = sum(1 for k in ks if b1[k]["c"] and not b1[k]["a"]); l1 = sum(1 for k in ks if b1[k]["a"] and not b1[k]["c"])
    check(f"batch-1 stage 1: {w1+l1} discordant of {len(ks)}, +{w1-l1} puzzles = +{round((w1-l1)*100/30)} points, p={sign_p(w1, l1):.2f} (entry: 8 of 30, +20, p=0.07)",
          (len(ks), w1, l1) == (30, 7, 1) and round((w1 - l1) * 100 / 30) == 20 and f"{sign_p(w1, l1):.2f}" == "0.07")
    # the entry's own words (appended after the locked P37 sentences, before 'Unequal output budget')
    e0 = tex.find(NEW_ENTRY); e1 = tex.find(r"\textbf{Unequal output budget.}")
    ent = " ".join(tex[e0:e1].split()) if 0 <= e0 < e1 else ""
    for lit in (r"\textbf{Stage 1 cannot be decided.}",
                "same candidate-set membership on 53 of the 60 confirmation puzzles at stage 1 (both list gold on 25, neither on 28)",
                "5 puzzles favour \\Mmaint{} and 2 favour \\Mstate{}, a difference of 5 points ($p = 0.45$)",
                "moved by $-3$, $-2$, and $-8$ points between temperatures 0 and 0.7 (ADR-0037), and the maintenance mechanism's by at most 2",
                "so the stage-1 gain is $+5$ points at temperature 0 and $+8$ points at 0.7",
                "Stage 2 is where the mechanisms actually separate (28 of 60 puzzles discordant, $+43$ points, $p < 0.001$)",
                "differ only in wording (the maintenance prompt adds confidence percentages and a no-collapse instruction)",
                "they diverge structurally from the second turn on",
                "should therefore be read as a descriptive property of this puzzle set, not as evidence for or against hypothesis maintenance",
                "the batch-1 figure of $+20$ points rests on 8 discordant puzzles out of 30 ($p = 0.07$)"):
        check(f"stage-1 sentences state: {lit[:64]!r}", lit in ent)
    check("stage-1 sentences do NOT claim the two prompts are equivalent / identical at turn 1",
          "equivalent" not in ent[ent.find("Stage 1 cannot be decided"):] and "identical at" not in ent[ent.find("Stage 1 cannot be decided"):])
    check("stage-1 sentences appear exactly once, inside the sampling-noise entry, before 'Unequal output budget'",
          tex.count(r"\textbf{Stage 1 cannot be decided.}") == 1 and 0 <= e0 < tex.find(r"\textbf{Stage 1 cannot be decided.}") < e1)
    print("\n== build ==")
    log = next((q for q in (Path("/tmp/tex3.log"), Path("/tmp/p35b_tex3.log")) if q.exists()), Path("/tmp/tex3.log"))
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
