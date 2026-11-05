"""Analysis for the E5d-ceiling experiment (saturation / headroom conjectures).

Reads runs/v2_e5d_ceiling/intervene_e5d_ceiling.json and emits:
  1. Per-model conditioning improvement and probe-benefit at gamma=0.
  2. C1 (operator saturation): is lr1.0_wd5e-05_s0's conditioning improvement
     anomalously small?  Does its reachable loss fail to drop?
  3. C2 (high-D headroom): across the 8 E3 models, does baseline D_eff
     correlate negatively with the whitening benefit?
  4. Replication check: the 3 models overlapping E5c must reproduce the
     E5c conditioning numbers and curves (seed-pinned).
Output: runs/v2_e5d_ceiling/ceiling_report.json + stdout summary.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FP = ROOT / "runs" / "v2_e5d_ceiling" / "intervene_e5d_ceiling.json"
E5C_FP = ROOT / "runs" / "v2_e5c.json"
OUT = ROOT / "runs" / "v2_e5d_ceiling" / "ceiling_report.json"

# glue_d from the E3 runs (baseline D_eff of each model, v2_e3.json)
E3 = json.load(open(ROOT / "runs" / "v2_e3.json"))["rows"]
GLUE_D = {r["model"]: r["glue_d"] for r in E3 if r.get("kind") == "train_whiten"}
# E3 c100 benefit (formation-phase whitening gain), for the benefit axis
C100_GAIN = {r["model"]: 100 * (r["c100_probe"] - r["c100_probe_ctrl"])
             for r in E3 if r.get("kind") == "train_whiten"}


def curve_at(rows, model, param, space, epoch=40):
    for r in rows:
        if (r["model"] == model and r["param"] == param
                and r["space"] == space):
            for c in r["curve"]:
                if c["epoch"] == epoch:
                    return c
    return None


def main():
    rows = json.load(open(FP))["rows"]
    models = sorted({r["model"] for r in rows})
    rep = {"n_models": len(models), "models": {}}

    print(f"{'model':44s} {'D_eff':>6s} {'e3gain':>7s} "
          f"{'cond@0 orig':>12s} {'cond@0 dose':>12s} {'ratio':>10s} "
          f"{'acc orig':>8s} {'acc dose':>8s} {'benefit':>8s} "
          f"{'loss40 o':>8s} {'loss40 d':>8s}")
    for m in models:
        spec = {(r["param"], r["space"]): r["in_rank_cond"]
                for r in rows if r["model"] == m}
        co = curve_at(rows, m, 0.0, "orig")
        cd = curve_at(rows, m, 0.0, "dose")
        if co is None or cd is None or (0.0, "orig") not in spec:
            continue
        cond_o = spec[(0.0, "orig")]
        cond_d = spec[(0.0, "dose")]
        acc_o, acc_d = co["c100_acc"], cd["c100_acc"]
        benefit = 100 * (acc_d - acc_o)
        loss_o, loss_d = co["train_loss"], cd["train_loss"]
        ratio = cond_o / cond_d if cond_d > 0 else float("inf")
        rep["models"][m] = {
            "glue_d": GLUE_D.get(m),
            "e3_gain_pp": C100_GAIN.get(m),
            "cond_orig_g0": cond_o, "cond_dose_g0": cond_d,
            "cond_improve_ratio": round(ratio, 2),
            "probe_acc_orig": acc_o, "probe_acc_dose": acc_d,
            "probe_benefit_pp": round(benefit, 3),
            "train_loss40_orig": loss_o, "train_loss40_dose": loss_d,
            "loss_reduced": loss_d < loss_o,
            "epoch1_loss_orig": curve_at(rows, m, 0.0, "orig", 1)["train_loss"],
            "epoch1_loss_dose": curve_at(rows, m, 0.0, "dose", 1)["train_loss"],
        }
        print(f"{m:44s} {GLUE_D.get(m, 0):6.2f} {C100_GAIN.get(m, 0):+7.2f} "
              f"{cond_o:12.3g} {cond_d:12.3g} {ratio:10.2g} "
              f"{acc_o*100:8.2f} {acc_d*100:8.2f} {benefit:+8.2f} "
              f"{loss_o:8.3f} {loss_d:8.3f}")

    ms = rep["models"]
    # ---- C2: D_eff vs benefit across models ----
    xs = np.array([ms[m]["glue_d"] for m in ms])
    ys = np.array([ms[m]["probe_benefit_pp"] for m in ms])
    if len(xs) >= 3:
        r = float(np.corrcoef(xs, ys)[0, 1])
        from scipy import stats
        rho, p = stats.spearmanr(xs, ys)
        rep["C2_def_v_benefit"] = {"pearson_r": round(r, 3),
                                   "spearman_rho": round(float(rho), 3),
                                   "spearman_p": round(float(p), 4),
                                   "n": len(xs)}
        print(f"\nC2 high-D headroom: D_eff vs probe benefit: "
              f"r={r:+.3f} rho={rho:+.3f} (p={p:.4f}, n={len(xs)})")

    # also vs conditioning improvement
    xs2 = np.array([ms[m]["cond_improve_ratio"] for m in ms])
    r2 = float(np.corrcoef(np.log10(xs2), ys)[0, 1])
    rep["C2_logcondratio_v_benefit"] = {"pearson_r": round(r2, 3), "n": len(xs2)}
    print(f"C2 cond-improvement vs benefit: r={r2:+.3f} (log10 ratio)")

    # ---- C1: the negative model ----
    neg = "resnet18_lr1.0_wd5e-05_s0"
    if neg in ms:
        n = ms[neg]
        others = [ms[m] for m in ms if m != neg]
        med_ratio = float(np.median([o["cond_improve_ratio"] for o in others]))
        rep["C1_negative_model"] = {
            "cond_improve_ratio": n["cond_improve_ratio"],
            "median_others": round(med_ratio, 1),
            "loss_reduced": n["loss_reduced"],
            "probe_benefit_pp": n["probe_benefit_pp"],
            "cond_orig_g0": n["cond_orig_g0"],
        }
        print(f"\nC1 saturation: {neg}")
        print(f"  cond orig(g=0) = {n['cond_orig_g0']:.3g} "
              f"(others' median improve ratio {med_ratio:.3g}x vs "
              f"this {n['cond_improve_ratio']:.3g}x)")
        print(f"  reachable loss reduced: {n['loss_reduced']}, "
              f"probe benefit {n['probe_benefit_pp']:+.2f}pp")

    # ---- replication vs E5c ----
    if E5C_FP.exists():
        e5c = json.load(open(E5C_FP))["rows"]
        e5c_spec = {(r["model"], r["param"], r["space"]): r["in_rank_cond"]
                    for r in e5c}
        e5c_cur = {(r["model"], r["param"], r["space"]): r["curve"][-1]
                   for r in e5c}
        reps = []
        for m in ms:
            for g in (0.75, 0.0, -0.5):
                for sp in ("orig", "dose"):
                    k = (m, g, sp)
                    if k not in e5c_spec:
                        continue
                    mine = [r for r in rows if r["model"] == m
                            and r["param"] == g and r["space"] == sp]
                    if not mine:
                        continue
                    c_old, c_new = e5c_spec[k], mine[0]["in_rank_cond"]
                    a_old = e5c_cur[k]["c100_acc"]
                    a_new = mine[0]["curve"][-1]["c100_acc"]
                    reps.append({
                        "model": m, "gamma": g, "space": sp,
                        "cond_e5c": c_old, "cond_now": c_new,
                        "acc_e5c": a_old, "acc_now": a_new,
                        "cond_match": abs(c_old - c_new)
                        <= 1e-6 * max(abs(c_old), 1.0) or c_old == c_new,
                        "acc_match": a_old == a_new,
                    })
        rep["replication_vs_e5c"] = reps
        ok_cond = sum(r["cond_match"] for r in reps)
        ok_acc = sum(r["acc_match"] for r in reps)
        print(f"\nReplication vs E5c ({len(reps)} conditions): "
              f"cond exact {ok_cond}/{len(reps)}, acc exact {ok_acc}/{len(reps)}")

    json.dump(rep, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
