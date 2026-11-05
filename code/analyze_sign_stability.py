#!/usr/bin/env python3
"""Sign-stability map of representation-geometry markers as OOD predictors.

Extends the pilot (analyze_impl_stability.py, 3 GLUE markers x 3 cells) to the
full preregistered design of ADR-0016 candidate A, section 5:

    18 marker families x {task, 6 corruptions} x {2 implementations}
    x {5 shift/falsification directions}

Data sources
------------
Official implementation:

  * `official224`: 224-model release (df_fig3.pkl), ID markers -> acc_lp_ood.
  * `official80/<c>`: 80-model release, 6 CIFAR-C corruptions
    (df_fig15-20.pkl), ID markers -> acc_lp_ood (corruption accuracy).

Our own implementation:

  * `own48`: 48 self-trained resnet18 recipes, `glue_*` + `id_*` markers
    -> {c100, tinyimg, corr_<c>} linear-probe accuracies.

Marker families pair the two implementations via aliases
(dimension<->glue_dimension, radius<->glue_radius, utility<->glue_utility,
nc1<->id_nc1, numerical_rank<->id_numerical_rank,
participation_ratio<->id_pr_global); official-only and own-only families are
kept and flagged `single_source`.

Comparison groups

  task              : official224  +  own48/c100  +  own48/tinyimg
  corruption:<name> : official80/<name> + own48/<name>   (6 shared shifts)

Verdicts per family (primary band 1 pp; usable = n >= 8, |r| >= 0.05),
in precedence order:

  flip            : sign disagreement inside any group (cross-implementation,
                    same shift) -- the pilot's radius headline
  shift-dependent : signs agree within groups but differ across shift groups
                    inside one implementation (e.g. gaussian vs brightness)
  unstable-n      : sign flips across ID-match bands within a group
  stable-same     : same sign everywhere, |r| >= 0.3 in >= 2 groups
  weak-same       : same sign, weaker magnitudes
  na              : no usable cells

Flags: `magnitude_collapse` (max/min |r| >= 2.5 across group representatives),
`single_source` (only one implementation has cells).

S4 = seeded half-sample resampling (sign-agreement with full-sample r).
S3 = per-lr-tier severity sweep (auxiliary, reported but not verdict-bearing).

Usage:
    python analyze_sign_stability.py --outdir results
Outputs: stability_map.json, stability_map_report.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

MIN_N = 8
PRIMARY_BAND = 1.0
R_FLOOR = 0.05
STRONG_R = 0.30
COLLAPSE_RATIO = 2.5


def find_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, here.parent, here.parent.parent]:
        if (cand / "results").is_dir() or (cand / "data").is_dir():
            return cand
    return here.parent


ROOT = find_root()


def locate(*names) -> Path:
    for name in names:
        for base in ("data/official", "data/self_trained", "results", "."):
            cand = ROOT / base / name
            if cand.exists():
                return cand
    raise FileNotFoundError(" / ".join(names))


# --------------------------------------------------------------------------
# protocol primitives (identical to the pilot)
# --------------------------------------------------------------------------

def group_residual(v, groups):
    v = np.asarray(v, float)
    out = v.copy()
    for g in sorted(set(groups)):
        m = np.array([x == g for x in groups])
        out[m] -= v[m].mean()
    return out


def resid_corr(x, y, groups):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < MIN_N or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None, None
    rx, ry = group_residual(x, groups), group_residual(y, groups)
    if np.allclose(rx, rx[0]) or np.allclose(ry, ry[0]):
        return None, None
    return float(pearsonr(rx, ry)[0]), float(spearmanr(rx, ry)[0])


def cell_corr(table, xcol, ycol, idcol, band):
    acc = table[idcol].values.astype(float)
    keep = (acc >= acc.mean() - band) & (acc <= acc.mean() + band)
    sub = table[keep]
    if len(sub) < MIN_N:
        return None, None, len(sub)
    gcol = "arch_name" if "arch_name" in sub.columns else "arch"
    r, rho = resid_corr(sub[xcol].values, sub[ycol].values, sub[gcol].values)
    return r, rho, len(sub)


# --------------------------------------------------------------------------
# data + marker inventory
# --------------------------------------------------------------------------

CORRUPTIONS = ["gaussian_noise", "fog", "brightness", "contrast",
               "pixelate", "jpeg_compression"]

OWN_GLUE = ["glue_dimension", "glue_radius", "glue_utility"]
OWN_ID = ["id_nc1", "id_numerical_rank", "id_pr_global", "id_pr_perclass",
          "id_effective_dim_pca", "id_local_smoothness", "id_class_separability"]

# family name -> (official marker or None, own marker or None)
FAMILIES = {
    "dimension": ("dimension", "glue_dimension"),
    "radius": ("radius", "glue_radius"),
    "utility": ("utility", "glue_utility"),
    "nc1": ("nc1", "id_nc1"),
    "numerical_rank": ("numerical_rank", "id_numerical_rank"),
    "participation_ratio": ("participation_ratio", "id_pr_global"),
    "participation_ratio_perclass": (None, "id_pr_perclass"),
    "effective_dim_pca": (None, "id_effective_dim_pca"),
    "local_smoothness": (None, "id_local_smoothness"),
    "class_separability": (None, "id_class_separability"),
    "sparsity": ("sparsity", None),
    "mean_distance": ("mean_distance", None),
    "mean_angle": ("mean_angle", None),
    "mean_covariance": ("mean_covariance", None),
    "axis_alignment": ("axis_alignment", None),
    "capacity": ("capacity", None),
    "center_alignment": ("center_alignment", None),
    "center_axis_alignment": ("center_axis_alignment", None),
}
AVG_VARIANTS = {  # official-only secondary measurements, separate families
    "sparsity": "sparsity_avg", "mean_distance": "mean_distance_avg",
    "mean_angle": "mean_angle_avg", "mean_covariance": "mean_covariance_avg",
    "participation_ratio": "participation_ratio_avg",
    "numerical_rank": "numerical_rank_avg",
}


def load_official224() -> pd.DataFrame:
    return pd.read_pickle(locate("df_fig3.pkl"))


def load_official80() -> dict:
    return pd.read_pickle(locate("df_fig15-20.pkl"))


def load_own() -> pd.DataFrame:
    glue = {r["ckpt"]: r for r in
            json.load(open(locate("glue_own48_results.json")))["rows"]}
    ind = json.load(open(locate("ind_validation_results.json")))["rows"]
    rows = []
    for r in ind:
        if r["ckpt"] not in glue:
            continue
        g = glue[r["ckpt"]]
        row = {"arch_name": r["arch"], "acc_test": r["id_acc"] * 100.0}
        for k in OWN_GLUE:
            row[k] = g[k]
        for k in OWN_ID:
            row[k] = r[k]
        row["target_c100"] = r["c100_probe_acc"] * 100.0
        row["target_tinyimg"] = r["tinyimg_probe_acc"] * 100.0
        for c in CORRUPTIONS:
            row[f"target_corr_{c}"] = r[f"corr_{c}_acc"] * 100.0
        row["lr_tier"] = f"lr{r['lr']}"
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# per-cell computation
#   cells[(cell_label, kind, marker)] = {band: {r, rho, n}}
#   marker is the NATIVE column name; families group them afterwards.
# --------------------------------------------------------------------------

def compute_all_cells(bands):
    d224 = load_official224()
    d80 = load_official80()
    own = load_own()
    cells = {}

    # ---- official224: all markers, band sweep (task) ----
    for fam, (om, _ow) in FAMILIES.items():
        for m in (om, AVG_VARIANTS.get(fam)):
            if m and f"{m}_id" in d224.columns:
                for band in bands:
                    r, rho, n = cell_corr(d224, f"{m}_id", "acc_lp_ood",
                                          "acc_test", band)
                    cells.setdefault((f"official224:{m}", "task", m), {})[band] = \
                        {"r": r, "rho": rho, "n": n}

    # ---- official80 x 6 corruptions (primary band; sweep on gaussian) ----
    for fam, (om, _ow) in FAMILIES.items():
        for m in (om, AVG_VARIANTS.get(fam)):
            if not (m and f"{m}_id" in d80["gaussian_noise"].columns):
                continue
            for corr in CORRUPTIONS:
                r, rho, n = cell_corr(d80[corr], f"{m}_id", "acc_test_c",
                                      "acc_test", PRIMARY_BAND)
                cells.setdefault((f"official80/{corr}:{m}", "corruption", m),
                                 {})[PRIMARY_BAND] = {"r": r, "rho": rho, "n": n}
            for band in bands:
                r, rho, n = cell_corr(d80["gaussian_noise"], f"{m}_id",
                                      "acc_lp_ood", "acc_test", band)
                cells.setdefault((f"official80/gaussian_noise:{m}",
                                  "corruption", m), {})[band] = \
                    {"r": r, "rho": rho, "n": n}

    # ---- own48: 8 targets, band sweep ----
    targets = (["c100", "tinyimg"] + CORRUPTIONS)
    for fam, (_om, ow) in FAMILIES.items():
        if not ow:
            continue
        for label in targets:
            tcol = (f"target_corr_{label}" if label in CORRUPTIONS
                    else f"target_{label}")
            kind = "task" if label in ("c100", "tinyimg") else "corruption"
            for band in bands:
                r, rho, n = cell_corr(own, ow, tcol, "acc_test", band)
                cells.setdefault((f"own48/{label}:{ow}", kind, ow), {})[band] = \
                    {"r": r, "rho": rho, "n": n}

    # ---- own48 lr-tier severity sweep (S3, auxiliary) ----
    for fam, (_om, ow) in FAMILIES.items():
        if not ow:
            continue
        for tier, sub in own.groupby("lr_tier"):
            if len(sub) < MIN_N:
                continue
            r, rho = resid_corr(sub[ow].values, sub["target_c100"].values,
                                sub["arch_name"].values)
            cells.setdefault((f"own48/lr_tier:{tier}:{ow}", "severity", ow),
                             {})[PRIMARY_BAND] = {"r": r, "rho": rho,
                                                  "n": len(sub)}
    return cells


# --------------------------------------------------------------------------
# S4 resample sign stability
# --------------------------------------------------------------------------

def resample_agreement(cells, n_resample=1000, seed=0):
    rng = np.random.default_rng(seed)
    d224 = load_official224()
    d80 = load_official80()
    own = load_own()
    tables = {
        "official224": (d224, "acc_lp_ood", "acc_test", "arch_name"),
        "official80/gaussian_noise": (d80["gaussian_noise"], "acc_lp_ood",
                                      "acc_test", "arch_name"),
    }
    out = {}
    for (cell, kind, marker), per_band in cells.items():
        base = cell.split(":", 1)[0].split("/", 1)
        key = cell.split(":", 1)[0]
        m = marker
        # column names
        if key.startswith("official"):
            xcol = f"{m}_id"
            ycol = "acc_lp_ood"
            idcol = "acc_test"
        elif key.startswith("own48/lr_tier"):
            continue
        elif key.startswith("own48/"):
            xcol = m
            ycol = ("target_corr_" + key.split("/", 1)[1]
                    if key.split("/", 1)[1] in CORRUPTIONS
                    else "target_" + key.split("/", 1)[1])
            idcol = "acc_test"
        else:  # official80/<corruption>:<marker>
            xcol = f"{m}_id"
            ycol = "acc_test_c"
            idcol = "acc_test"
        if key in tables:
            table = tables[key][0]
        else:
            table = own
        gcol = "arch_name" if "arch_name" in table.columns else "arch"
        if xcol not in table.columns:
            out[(cell, marker)] = None
            continue
        acc = table[idcol].values.astype(float)
        keep = ((acc >= acc.mean() - PRIMARY_BAND)
                & (acc <= acc.mean() + PRIMARY_BAND))
        idx = np.where(keep)[0]
        if len(idx) < MIN_N:
            out[(cell, marker)] = None
            continue
        full_r, _ = resid_corr(table[xcol].values[idx], table[ycol].values[idx],
                               table[gcol].values[idx])
        if full_r is None or abs(full_r) < R_FLOOR:
            out[(cell, marker)] = None
            continue
        agree = total = 0
        for _ in range(n_resample):
            pick = rng.choice(idx, size=max(MIN_N, len(idx) // 2),
                              replace=False)
            r, _ = resid_corr(table[xcol].values[pick],
                              table[ycol].values[pick],
                              table[gcol].values[pick])
            if r is not None:
                total += 1
                agree += 1 if np.sign(r) == np.sign(full_r) else 0
        out[(cell, marker)] = agree / total if total >= n_resample * 0.8 else None
    return out


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------

def group_of(cell, kind):
    if kind == "severity":
        return None
    if kind == "task":
        return "task"
    return "corruption:" + cell.split(":", 1)[0].split("/", 1)[1]


def family_cells(cells):
    """Group native-marker cells into families via FAMILIES aliases."""
    by_marker = {}
    for (cell, kind, marker), per_band in cells.items():
        by_marker.setdefault(marker, {})[(cell, kind, marker)] = per_band
    fams = {}
    for fam, (om, ow) in FAMILIES.items():
        entry = {}
        if om and om in by_marker:
            entry.update(by_marker[om])
        if ow and ow in by_marker:
            entry.update(by_marker[ow])
        if entry:
            fams[fam] = entry
    for m, entry in by_marker.items():  # avg variants as own families
        if m.endswith("_avg"):
            fams[m] = entry
    return fams


def verdict_of(entry, resample):
    """entry: {(cell, kind, marker): {band: {...}}} for one family."""
    impls = {("official" if k[0].startswith("official") else "own")
             for k in entry if group_of(k[0], k[1]) is not None}
    group_cells, aux_cells = {}, {}
    for (cell, kind, marker), per_band in entry.items():
        g = group_of(cell, kind)
        if g is None:
            aux_cells[cell] = per_band
        else:
            group_cells.setdefault(g, {}).setdefault(marker, {})[cell] = per_band

    def usable_at(per_band, band=PRIMARY_BAND):
        v = per_band.get(band, {})
        r = v.get("r")
        return r if (r is not None and abs(r) >= R_FLOOR) else None

    detail, flips, band_flips, shift_flips = {}, [], [], []
    strong_groups, abs_vals = 0, []
    for g, by_marker in sorted(group_cells.items()):
        for marker, cells_m in by_marker.items():
            for cell, per_band in cells_m.items():
                r = usable_at(per_band)
                ra = resample.get((cell, marker))
                detail.setdefault(g, {})[cell] = {
                    "r": r, "rho": per_band.get(PRIMARY_BAND, {}).get("rho"),
                    "n": per_band.get(PRIMARY_BAND, {}).get("n"),
                    "resample_agree": round(ra, 3) if ra is not None else None,
                }
        # gather usable rs in this group (any marker/impl)
        rs = {}
        for marker, cells_m in by_marker.items():
            for cell, per_band in cells_m.items():
                r = usable_at(per_band)
                if r is not None:
                    rs[cell] = r
        if not rs:
            continue
        if len({np.sign(v) for v in rs.values()}) > 1:
            flips.append((g, {c: round(v, 3) for c, v in rs.items()}))
            continue
        # band sweep: any cell flipping sign across bands
        for cell, per_band in ((c, b) for mk in by_marker.values()
                               for c, b in mk.items()):
            band_rs = [np.sign(v) for v in
                       (usable_at(per_band, b) for b in per_band)
                       if v is not None]
            if len(set(band_rs)) > 1:
                band_flips.append(g)
                break
        if abs(next(iter(rs.values()))) >= STRONG_R:
            strong_groups += 1
        abs_vals.append(max(abs(v) for v in rs.values()))

    # shift-dependence: within one implementation, do shift groups disagree?
    for impl in ("official", "own"):
        sig = set()
        for g, by_marker in group_cells.items():
            for marker, cells_m in by_marker.items():
                for cell, per_band in cells_m.items():
                    if (("official" if cell.startswith("official") else "own")
                            == impl):
                        r = usable_at(per_band)
                        if r is not None:
                            sig.add((np.sign(r), g))
        gs = {g for _s, g in sig}
        ss = {s for s, _g in sig}
        if len(gs) > 1 and len(ss) > 1:
            # need actual disagreement between shift groups of this impl
            per_shift = {}
            for s, g in sig:
                per_shift.setdefault(g, set()).add(s)
            if any(len(v) == 1 for v in per_shift.values()) and \
                    len({next(iter(v)) for v in per_shift.values()}) > 1:
                shift_flips.append(impl)

    if flips:
        return {"verdict": "flip", "groups": detail, "flip_groups": flips,
                "impls": sorted(impls), "aux": aux_cells}
    if not abs_vals:
        return {"verdict": "na", "groups": detail, "impls": sorted(impls),
                "aux": aux_cells}
    flags = {"band_flip_groups": sorted(set(band_flips))}
    if len(impls) == 1:
        flags["single_source"] = True
    if max(abs_vals) / min(abs_vals) >= COLLAPSE_RATIO:
        flags["magnitude_collapse"] = True
    if shift_flips:
        flags["shift_flips"] = shift_flips
    if band_flips:
        v = "unstable-n"
    elif shift_flips:
        v = "shift-dependent"
    else:
        v = "stable-same" if strong_groups >= 2 else "weak-same"
    return {"verdict": v, "groups": detail, "flags": flags,
            "impls": sorted(impls), "aux": aux_cells}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--bands", type=float, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--resample", type=int, default=1000)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bands = args.bands

    print(f"[1/3] computing cells (bands={bands}) ...", flush=True)
    cells = compute_all_cells(bands)
    print(f"      {len(cells)} entries", flush=True)
    print(f"[2/3] resample sign-stability (n={args.resample}) ...", flush=True)
    rs = resample_agreement(cells, n_resample=args.resample, seed=0)
    print(f"      {sum(v is not None for v in rs.values())} usable", flush=True)

    fams = family_cells(cells)
    verdicts = {fam: verdict_of(entry, rs) for fam, entry in sorted(fams.items())}
    counts = Counter(v["verdict"] for v in verdicts.values())
    print("[3/3] writing outputs ...", flush=True)

    # ---- markdown report ----
    L = ["# Sign-stability map: geometry markers vs OOD, full design", "",
         "Protocol: ID-match (±1 pp around mean `acc_test`) + architecture",
         "residualisation (own48 is single-architecture → mean-centering).",
         f"Usable = n≥{MIN_N}, |r|≥{R_FLOOR}. Groups pair implementations:",
         "`task` = official224 + own48/c100 + own48/tinyimg;",
         "`corruption:<name>` = official80/<name> + own48/<name>.",
         "resample-agree = fraction of seeded half-samples keeping the",
         "full-sample sign (1 pp band, gaussian_noise / task cells).", "",
         "## Verdict summary", "",
         "| verdict | # families |", "|---|---|"]
    for k in ("stable-same", "weak-same", "flip", "shift-dependent",
              "unstable-n", "na"):
        L.append(f"| {k} | {counts.get(k, 0)} |")

    L += ["", "## Per-family matrix (primary band = 1 pp)", ""]
    for fam, v in sorted(verdicts.items()):
        flags = v.get("flags", {})
        flag_s = " ".join(
            f"[{k}={'/'.join(map(str, v2))}]" if isinstance(v2, list)
            else f"[{k}]" if v2 is True else f"[{k}={v2}]"
            for k, v2 in flags.items()) if flags else ""
        L.append(f"### `{fam}` — **{v['verdict']}** "
                 f"({'+'.join(v.get('impls', []))}) {flag_s}".rstrip())
        L.append("")
        L.append("| group | cell | r | rho | n | resample-agree |")
        L.append("|---|---|---|---|---|---|")
        for g, cellmap in sorted(v.get("groups", {}).items()):
            for cell, d in sorted(cellmap.items()):
                r = "—" if d["r"] is None else f"{d['r']:+.3f}"
                rho = "—" if d["rho"] is None else f"{d['rho']:+.3f}"
                ra = "—" if d["resample_agree"] is None else \
                    f"{d['resample_agree']:.2f}"
                L.append(f"| {g} | {cell} | {r} | {rho} | {d['n']} | {ra} |")
        for g, dd in v.get("flip_groups", []):
            L.append(f"- **flip in `{g}`**: {dd}")
        if v.get("aux"):
            L.append("")
            L.append("| aux (severity, not verdict-bearing) | r | n |")
            L.append("|---|---|---|")
            for cell, per_band in sorted(v["aux"].items()):
                b = per_band.get(PRIMARY_BAND, {})
                r = "—" if b.get("r") is None else f"{b['r']:+.3f}"
                L.append(f"| {cell} | {r} | {b.get('n')} |")
        L.append("")

    out_md = outdir / "stability_map_report.md"
    out_md.write_text("\n".join(L), encoding="utf-8")

    out_json = outdir / "stability_map.json"
    out_json.write_text(json.dumps(
        {"protocol": {"bands": bands, "primary_band": PRIMARY_BAND,
                      "min_n": MIN_N, "r_floor": R_FLOOR,
                      "strong_r": STRONG_R,
                      "collapse_ratio": COLLAPSE_RATIO,
                      "resample": args.resample, "seed": 0},
         "verdict_summary": dict(counts),
         "verdicts": verdicts,
         "raw_cells": {f"{c}|{k}|{m}|band{b}": val
                       for (c, k, m), pb in cells.items()
                       for b, val in pb.items()}},
        indent=1, default=str), encoding="utf-8")

    print(f"[saved] {out_md}")
    print(f"[saved] {out_json}")
    print("verdict counts:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
