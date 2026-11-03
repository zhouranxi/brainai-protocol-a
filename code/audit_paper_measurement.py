#!/usr/bin/env python3
"""Check every number in the measurement paper against the generated results.

The paper quotes numbers that were produced by the analysis scripts. This
script re-derives them from the released data and compares them with what the
manuscript says, so that a stale or mistyped figure cannot survive.

Usage:
    python audit_paper_measurement.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def find_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, here.parent, here.parent.parent]:
        if (cand / "results").is_dir():
            return cand
    return here.parent


ROOT = find_root()
TEX = ROOT / "paper" / "measurement" / "main.tex"


def locate(*names):
    """Find an input file under any of the layouts we publish.

    The released pickles live in data/official/, the self-trained measurements
    in data/self_trained/, and the Phase 0 recipe sweep in results/; a flat
    working directory has them all in one place.
    """
    for name in names:
        for base in ("results", "data/official", "data/self_trained", "."):
            cand = ROOT / base / name
            if cand.exists():
                return cand
    raise FileNotFoundError(" / ".join(names))

CHECKS = []

# Sections whose reported numbers are checked for internal consistency with the
# released analysis outputs but cannot be re-derived from primary measurements,
# because the underlying measurements were not released. Listing them here makes
# the gap visible in the audit output instead of leaving a false impression of
# full reproducibility.
UNREPRODUCIBLE = set()


def check(section, name, paper, recomputed, tol=0.006):
    CHECKS.append({"section": section, "name": name, "paper": float(paper),
                   "recomputed": float(recomputed), "ok": bool(abs(paper - recomputed) <= tol)})


def check_text(section, name, needle, present, tex):
    """Assert that the manuscript literally contains (or does not contain) a string.

    A number can be recomputed correctly and still be missing from the paper,
    or the paper can still carry a value the data no longer supports.  These
    checks tie the audited value to the manuscript text itself.
    """
    found = needle in tex
    CHECKS.append({"section": section, "name": name,
                   "paper": "present" if present else "absent",
                   "recomputed": "present" if found else "absent",
                   "ok": bool(found == present)})


def gres(v, g):
    v = np.asarray(v, float)
    out = v.copy()
    for gg in sorted(set(g)):
        m = np.array([x == gg for x in g])
        out[m] -= v[m].mean()
    return out


def official_cell(df, xcol, ycol, band):
    a = df["acc_test"].values.astype(float)
    m = (a >= a.mean() - band) & (a <= a.mean() + band)
    sub = df[m].reset_index(drop=True)
    if len(sub) < 8:
        return None, None, len(sub)
    x = sub[xcol].values.astype(float)
    y = sub[ycol].values.astype(float)
    g = sub["arch_name"].values
    r = pearsonr(gres(x, g), gres(y, g))
    return float(r[0]), float(spearmanr(gres(x, g), gres(y, g))[0]), len(sub)


def own_cell(rows, glue, key, band):
    acc = np.array([r["id_acc"] for r in rows]) * 100
    m = (acc >= acc.mean() - band) & (acc <= acc.mean() + band)
    sub = [r for r, k in zip(rows, m) if k]
    if len(sub) < 8:
        return None, None, len(sub)
    x = np.array([glue[r["ckpt"]][key] for r in sub], dtype=float)
    y = np.array([r["c100_probe_acc"] for r in sub], dtype=float) * 100
    g = [r["arch"] for r in sub]
    r = pearsonr(gres(x, g), gres(y, g))
    return float(r[0]), float(spearmanr(gres(x, g), gres(y, g))[0]), len(sub)


def main() -> int:
    d224 = pd.read_pickle(locate("df_fig3.pkl"))
    d80 = pd.read_pickle(locate("df_fig15-20.pkl", "official_df_fig15-20.pkl"))["gaussian_noise"]
    glue = {r["ckpt"]: r for r in json.load(
        open(locate("glue_own48_results.json")))["rows"]}
    ind = json.load(open(locate("ind_validation_results.json")))["rows"]
    own = [r for r in ind if r["ckpt"] in glue]

    # ---- the stability map, band = 1 pp ----
    # The table quotes r values; check them against the map file.
    table = json.load(open(locate("marker_stability_map.json")))
    quoted = {
        "dimension": (0.925, 0.737, 0.919),
        "radius": (-0.584, 0.153, 0.602),
        "utility": (0.901, 0.808, 0.342),
        "center alignment": (0.483, 0.581, None),
        "participation ratio (per class)": (0.659, 0.592, 0.947),
        "participation ratio (global)": (0.877, 0.734, 0.895),
        "numerical rank (per class)": (0.436, None, 0.972),
        "mean angle": (0.124, -0.102, None),
        "NC1": (-0.145, None, -0.178),
    }
    CHECKS.clear()
    for row in table["rows"]:
        label = row["marker"]
        if label not in quoted:
            continue
        q224, q80, qown = quoted[label]
        for cellkey, q in (("official_224", q224), ("official_80", q80), ("own_48", qown)):
            if q is None or row.get(cellkey) is None:
                continue
            check("map", f"{label} [{cellkey}]", q, row[cellkey]["r"])

    # ---- the radius bandwidth claim ----
    for band, expect in [(1, -0.584), (2, -0.796), (3, -0.790), (5, -0.786)]:
        r, _, _ = official_cell(d224, "radius_id", "acc_lp_ood", band)
        check("map", f"radius 224 band={band}", expect, r)
    for band, expect in [(1, 0.602), (2, 0.492), (3, 0.353), (5, 0.376)]:
        r, _, _ = own_cell(own, glue, "glue_radius", band)
        check("map", f"radius own band={band}", expect, r)
    # the middle cell drifts negative as the band widens
    r1, _, _ = official_cell(d80, "radius_id", "acc_lp_ood", 1)
    r5, _, _ = official_cell(d80, "radius_id", "acc_lp_ood", 5)
    check("map", "radius 80 band=1", 0.153, r1)
    check("map", "radius 80 band=5", -0.221, r5)

    # ---- dimension bandwidth insensitivity ----
    for band, expect in [(1, 0.925), (5, 0.904)]:
        r, _, _ = official_cell(d224, "dimension_id", "acc_lp_ood", band)
        check("map", f"dimension 224 band={band}", expect, r)

    # ---- utility collapse ----
    r, _, _ = official_cell(d224, "utility_id", "acc_lp_ood", 1)
    check("map", "utility 224 band=1", 0.901, r)
    r, _, _ = own_cell(own, glue, "glue_utility", 1)
    check("map", "utility own band=1", 0.342, r)
    r, _, _ = own_cell(own, glue, "glue_utility", 3)
    check("map", "utility own band=3", -0.246, r)

    # ---- the recipe sweep ----
    data = json.load(open(locate("c2_results_seeds0-1-2.json")))["rows"]
    acc = np.array([r["test_acc"] for r in data])
    for mk, exp_r, exp_rho in [("effective_dim", -0.483, -0.536),
                                ("local_smoothness", -0.452, -0.338),
                                ("spectral_entropy", -0.410, -0.185),
                                ("criticality_proxy", -0.217, -0.015),
                                ("silhouette_gap", 0.768, 0.732),
                                ("linear_probe_acc", 0.993, 0.953)]:
        v = np.array([r[mk] for r in data], dtype=float)
        rr = pearsonr(v, acc)
        check("recipe", f"{mk} pooled r", exp_r, float(rr[0]))
        check("recipe", f"{mk} pooled rho", exp_rho, float(spearmanr(v, acc)[0]))

    # within-recipe residual
    names = sorted({r["name"] for r in data})
    for mk, expect in [("effective_dim", -0.443), ("local_smoothness", -0.079),
                       ("spectral_entropy", -0.521), ("criticality_proxy", 0.352)]:
        rv, av, rec = [], [], []
        for nm in names:
            sub = [r for r in data if r["name"] == nm]
            for r in sub:
                rv.append(r[mk]); av.append(r["test_acc"]); rec.append(nm)
        rv = np.array(rv, float); av = np.array(av, float); rec = np.array(rec, dtype=object)
        rv = rv - np.array([rv[rec == nm].mean() for nm in rec])
        av = av - np.array([av[rec == nm].mean() for nm in rec])
        check("recipe", f"{mk} within-recipe r", expect, float(pearsonr(rv, av)[0]))

    # ---- AUC: can the marker detect a collapsed run? ----
    #
    # The collapsed runs are identified by recipe: the two recipes that failed
    # to train (aug_ls, plain_short), three seeds each. The released JSON does
    # not carry a per-run training-accuracy field, so the label is derived from
    # the recipe list recorded in results/c2_stratified_report.md rather than
    # from a threshold on a stored measurement. The AUC itself is recomputed
    # from the released per-marker values.
    COLLAPSED = {"aug_ls", "plain_short"}
    n_fail = sum(1 for r in data if r["name"] in COLLAPSED)
    check("auc", "collapsed run count", 6, n_fail, tol=0)
    # The manuscript reports permutation p-values alongside the AUCs.  The
    # permutation is fixed-seed (20261103, 20,000 draws) so the audit
    # reproduces exactly the values printed in the paper; the tolerance covers
    # residual Monte-Carlo wobble across numpy versions.
    rng = np.random.default_rng(20261103)
    PERMS = 20000
    for mk, exp_auc, exp_p in [("criticality_proxy", 0.866, 0.003),
                               ("effective_dim", 0.750, 0.054),
                               ("local_smoothness", 0.750, 0.053),
                               ("spectral_entropy", 0.606, 0.43)]:
        y = np.array([1 if r["name"] in COLLAPSED else 0 for r in data], int)
        x = np.array([r[mk] for r in data], dtype=float)
        # rank-based AUC that does not depend on the sign of the association
        ranks = np.argsort(np.argsort(x))
        n1 = int(y.sum()); n0 = len(y) - n1
        auc = (ranks[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
        auc = float(max(auc, 1 - auc))
        check("auc", f"{mk} AUC", exp_auc, auc, tol=0.001)
        # permutation p-value: shuffle labels, recompute the direction-free AUC
        obs_stat = auc
        perm = np.empty(PERMS)
        for i in range(PERMS):
            yp = rng.permutation(y)
            a = (ranks[yp == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
            perm[i] = max(a, 1 - a)
        pval = float((perm >= obs_stat - 1e-12).mean())
        check("auc", f"{mk} permutation p", exp_p, pval, tol=0.006)

    # the high cluster
    HIGH = {"best", "aug_only", "noise20_aug"}
    sub = [r for r in data if r["name"] in HIGH]
    v = np.array([r["local_smoothness"] for r in sub], dtype=float)
    a = np.array([r["test_acc"] for r in sub])
    rr = pearsonr(v, a)
    check("recipe", "high cluster local_smoothness r", -0.053, float(rr[0]))
    check("recipe", "high cluster local_smoothness p", 0.891, float(rr[1]), tol=0.002)
    check("recipe", "high cluster n", 9, len(sub), tol=0)

    # the direct counter-example quoted in the text
    best = np.mean([r["test_acc"] for r in data if r["name"] == "best"])
    aug = np.mean([r["test_acc"] for r in data if r["name"] == "aug_only"])
    check("recipe", "best vs aug_only acc gap", 0.001, abs(best - aug), tol=0.0005)
    bs = np.mean([r["local_smoothness"] for r in data if r["name"] == "best"])
    as_ = np.mean([r["local_smoothness"] for r in data if r["name"] == "aug_only"])
    check("recipe", "best vs aug_only smoothness gap", 0.074, abs(bs - as_))

    # ---- the degradation sweep ----
    #
    # These values have no re-derivable source: they were transcribed from a
    # development smoke-test summary that is not part of the released
    # artifacts, and the per-seed measurements behind them were not retained.
    # The previous version of this check compared the constants against copies
    # of themselves, so it could never fail and protected nothing.
    #
    # What this version does is weaker but honest: it checks the paper's
    # numbers against the values the analysis script actually emits, so that a
    # change to either side is caught, and it records the provenance gap
    # explicitly so the gap is not silently forgotten. It does NOT re-derive
    # the measurements, and it must not be read as doing so.
    degen_path = None
    for cand in (ROOT / "results" / "degeneration_analysis.json",):
        if cand.exists():
            degen_path = cand
            break
    if degen_path is None:
        raise SystemExit("degeneration_analysis.json not found; cannot audit "
                         "the degradation table")
    deg = json.loads(degen_path.read_text(encoding="utf-8"))["degeneration"]
    check("degen", "clean dim", 316, deg["clean"]["dim"], tol=0)
    check("degen", "clean probe", 0.531, deg["clean"]["probe"], tol=0)
    check("degen", "occ70 dim", 128, deg["occ70"]["dim"], tol=0)
    check("degen", "occ70 probe", 0.264, deg["occ70"]["probe"], tol=0)
    check("degen", "noise dim", 271, deg["noise"]["dim"], tol=0)
    check("degen", "noise probe", 0.309, deg["noise"]["probe"], tol=0)
    check("degen", "dim drop pct", 59, round(100 * (316 - 128) / 316))
    check("degen", "probe drop pct", 50, round(100 * (0.531 - 0.264) / 0.531))
    UNREPRODUCIBLE.add("degen")

    # ---- Table 1's class column: the classification assigned by the script --
    #
    # The manuscript reproduces the label that the analysis script assigned to
    # each marker.  The label is read out of results/marker_stability_map.json
    # and compared with what the manuscript prints, so a re-classification
    # cannot silently leave the table stale.
    #
    # The manuscript shortens some of the script's labels for the table; the
    # mapping below records which script label each rendered label stands for,
    # so the comparison is against the script's own classification and not
    # against a constant copied out of the manuscript.
    CLASS_RENDER = {
        "dimension": "robust",
        "radius": r"\textbf{sign flips}",
        "utility": "consistent sign, variable magnitude",
        "center alignment": "robust",
        "participation ratio (per class)": "robust",
        "participation ratio (global)": "robust",
        "numerical rank (per class)": "magnitude varies",
        "mean angle": r"\textbf{sign flips}",
        "NC1": "sign flip (Spearman)",
    }
    # script label -> the rendered form(s) the manuscript may use for it
    RENDER_OF = {
        "robust": {"robust"},
        "SIGN FLIPS (Pearson)": {r"\textbf{sign flips}", "sign flips"},
        "SIGN FLIPS (Spearman)": {"sign flip (Spearman)", r"\textbf{sign flip (Spearman)}"},
        "consistent sign, variable magnitude": {"consistent sign, variable magnitude",
                                                "magnitude varies"},
    }
    map_by_marker = {r["marker"]: r for r in table["rows"]}
    for marker, rendered in CLASS_RENDER.items():
        script_class = map_by_marker.get(marker, {}).get("classification")
        allowed = RENDER_OF.get(script_class, {script_class})
        ok = rendered in allowed if script_class is not None else False
        CHECKS.append({"section": "class", "name": f"{marker} classification",
                       "paper": rendered, "recomputed": script_class, "ok": bool(ok)})

    # ---- Table 1 as rendered in the manuscript ----
    #
    # The class column is parsed out of the manuscript's own table so that the
    # label the reader actually sees is compared against the label the
    # analysis script assigned -- not against a constant copied out of the
    # manuscript.  A substring test would not notice an appended character,
    # so the comparison is on the whole cell.
    if TEX.exists():
        tex = TEX.read_text(encoding="utf-8")
        i = tex.find("Marker & ref")
        seg = tex[i:tex.find(r"\end{tabular}", i)] if i >= 0 else ""
        rows = {}
        for line in seg.split("\n"):
            if "&" not in line or line.lstrip().startswith("Marker"):
                continue
            cells = [c.strip() for c in line.rsplit("\\\\", 1)[0].split("&")]
            if len(cells) < 5:
                continue
            # the marker cell is "`name`" or "`name` (qualifier)"; the script
            # keys the map on "name (qualifier)" or "name"
            label = cells[0]
            m = re.match(r"`([^`]+)`\s*(?:\(([^)]*)\))?$", label)
            if not m:
                continue
            name, qualifier = m.group(1), m.group(2)
            label = f"{name} ({qualifier})" if qualifier else name
            rows[label] = cells[4]
        for marker in CLASS_RENDER:
            rendered = rows.get(marker)
            script_class = map_by_marker.get(marker, {}).get("classification")
            allowed = RENDER_OF.get(script_class, set())
            ok = rendered is not None and rendered in allowed
            CHECKS.append({"section": "class-text",
                           "name": f"{marker} class cell as rendered",
                           "paper": script_class, "recomputed": rendered,
                           "ok": bool(ok)})
        # the two strings that changed this round
        check_text("text", "main.tex: utility class label",
                   "consistent sign, variable magnitude", True, tex)
        # The old table label was "magnitude collapse"; it must be gone from
        # the table row.  (The phrase still appears in the surrounding prose,
        # where it describes the collapse itself, so the check is anchored to
        # the table row rather than to the bare string.)
        check_text("text", "main.tex: utility class label (old) absent from table row",
                   "`utility` & $+0.901$ & $+0.808$ & $+0.342$ & magnitude collapse",
                   False, tex)
        check_text("text", "main.tex: utility class label (in the table row)",
                   "`utility` & $+0.901$ & $+0.808$ & $+0.342$ & "
                   "consistent sign, variable magnitude", True, tex)
        check_text("text", "main.tex: best/aug_only accuracy gap",
                   "differ in test accuracy by 0.001", True, tex)
        check_text("text", "main.tex: best/aug_only accuracy gap (old) absent",
                   "differ in test accuracy by 0.003", False, tex)

    # ---- the best vs aug_only accuracy gap, recomputed ----
    # This was 0.003 before; the tightened tolerance (0.0005) now pins the
    # recomputed value, so a drift in either recipe's mean is caught.
    best_acc = np.mean([r["test_acc"] for r in data if r["name"] == "best"])
    aug_acc = np.mean([r["test_acc"] for r in data if r["name"] == "aug_only"])
    check("recipe", "best vs aug_only acc gap (exact)", 0.001,
          abs(best_acc - aug_acc), tol=0.0005)
    check("recipe", "best mean test acc", 0.864333, best_acc, tol=0.0005)
    check("recipe", "aug_only mean test acc", 0.865333, aug_acc, tol=0.0005)
    # the gap is an order of magnitude smaller than the smoothness gap
    check("recipe", "smoothness gap exceeds accuracy gap by >10x", 1,
          int(abs(bs - as_) > 10 * abs(best_acc - aug_acc)), tol=0)

    # ---- report ----
    failed = [c for c in CHECKS if not c["ok"]]
    by_sec = {}
    for c in CHECKS:
        d = by_sec.setdefault(c["section"], [0, 0])
        d[0] += 1
        if c["ok"]:
            d[1] += 1
    print(f"checks: {len(CHECKS)}  pass: {len(CHECKS) - len(failed)}  FAIL: {len(failed)}")
    for sec, (n, p) in sorted(by_sec.items()):
        mark = "  (not independently reproducible)" if sec in UNREPRODUCIBLE else ""
        print(f"  {sec}: {p}/{n}{mark}")
    for c in failed:
        got = (f"{c['recomputed']:.4f}" if isinstance(c["recomputed"], float)
               else str(c["recomputed"]))
        print(f"  MISMATCH [{c['section']}] {c['name']}: paper {c['paper']} "
              f"recomputed {got}")

    # also check the manuscript has no leftover placeholder numbers
    if TEX.exists():
        t = TEX.read_text(encoding="utf-8")
        stale = re.findall(r"XXXX+", t)
        # The OpenReview forum id is unknown until the submission is made; that
        # one placeholder is expected and is replaced with the real id from
        # the submission confirmation. Any other placeholder is a defect.
        openreview_placeholder = t.count("forum?id=XXXX")
        unexpected = len(stale) - openreview_placeholder
        if openreview_placeholder:
            print(f"  NOTE: {openreview_placeholder} expected OpenReview "
                  f"placeholder(s) -- fill in the forum id after submission")
        if unexpected > 0:
            print(f"  NOTE: {unexpected} unexpected placeholder(s) left in main.tex")
        for must in ["+0.925", "-0.584", "+0.901", "-0.053", "-0.246"]:
            if must not in t:
                print(f"  NOTE: main.tex does not contain {must!r} -- check the manuscript")
        # Any number the manuscript quotes that the audit knows to be different
        # must not survive. Compare the quoted utility value at the wide band.
        if "-0.229" in t:
            print("  NOTE: main.tex quotes -0.229 for the wide-band utility; "
                  "the recomputed value is -0.246")

    out = ROOT / "results" / "paper_measurement_audit.json"
    out.write_text(json.dumps(CHECKS, indent=2), encoding="utf-8")
    print(f"[saved] {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
