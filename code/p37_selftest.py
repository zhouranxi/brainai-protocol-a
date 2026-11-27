"""Synthetic-data self-test of code/p37_judge.py BEFORE launch. Each scenario plants a known
truth and checks the frozen read-out returns it. Also checks fisher() against scipy-free reference."""
import json, random, sys, importlib.util, math, tempfile, shutil
from pathlib import Path
spec = importlib.util.spec_from_file_location("pj", str(__import__("pathlib").Path(__file__).resolve().parent / "p37_judge.py"))
pj = importlib.util.module_from_spec(spec); spec.loader.exec_module(pj)

# 1. fisher vs a brute-force reference on a few tables
def fisher_ref(a,b,c,d):
    n=a+b+c+d; r1=a+b; c1=a+c
    pr=lambda x: math.comb(r1,x)*math.comb(n-r1,c1-x)/math.comb(n,c1)
    lo,hi=max(0,c1-(n-r1)),min(r1,c1); p0=pr(a)
    return min(1.0,sum(pr(x) for x in range(lo,hi+1) if pr(x)<=p0+1e-12))
for t in [(30,30,2,58),(50,10,20,40),(5,55,5,55),(1,59,0,60)]:
    assert abs(pj.fisher(*t)-fisher_ref(*t))<1e-12, t
print("fisher() matches brute-force reference on 4 tables")

PZ = [f"P{i:02d}" for i in range(60)]
def make(seed, p_flip_by_grade, truth_by_puzzle=None):
    """rows where each puzzle's top-1 is Bernoulli(p_stable) around a truth, tuned so that the
    flip rate is roughly p_flip_by_grade[g]. We build directly: with prob p_flip the puzzle's 5 reps
    are mixed, else all equal to its truth."""
    rnd = random.Random(seed); rows = []
    truth = truth_by_puzzle or {(p,g): rnd.random()<0.6 for p in PZ for g in (1,2,3)}
    for (p,g),t in truth.items():
        mixed = rnd.random() < p_flip_by_grade[g]
        reps = []
        for k in range(5):
            if not mixed: reps.append(int(t))
            else: reps.append(int(rnd.random()<0.5))
        if mixed and len(set(reps))==1: reps[0] = 1-reps[0]   # force a real flip
        for k,v in enumerate(reps): rows.append({"puzzle":p,"mech":"c","grade":g,"rep":k,"temp":0,"rank":1 if v else 3,
                                                 "top3":1,"dup":0,"difficulty":"medium","secs":1.0,"finish_reason":"stop","text":""})
    return rows
def run_scenario(name, t0, t7, t7b, expect):
    d = Path(tempfile.mkdtemp()) ; 
    for arm, rows in (("T0",t0),("T7",t7),("T7b",t7b)):
        (d / pj.ARMS[arm]).write_text(json.dumps({"rows":rows}), encoding="utf-8")
    pj.D = d
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): pj.main()
    got = json.loads((d/"readout.json").read_text()) if (d/"readout.json").exists() else {"S1":"FAIL"}
    shutil.rmtree(d)
    ok = got == expect
    print(f"  [{'OK' if ok else 'MISMATCH'}] {name}: expected {expect} got {got}")
    if not ok: print(buf.getvalue())
    return ok
allok = True
rnd = random.Random(1); TRUTH = {(p,g): rnd.random()<0.6 for p in PZ for g in (1,2,3)}
# A: clean floor (T0 deterministic), big temperature & format effect, majority vote helps
allok &= run_scenario("A clean: S1 pass, S2/S3/S4 pass",
    make(1, {1:0,2:0,3:0}, TRUTH), make(2, {1:0.5,2:0.6,3:0.6}, TRUTH), make(3, {1:0.1,2:0.1,3:0.1}, TRUTH),
    {"S1":"PASS","S2":"PASS","S3":"PASS","S4":"PASS"})
# B: noisy engine at temp 0 -> batch invalid, later gates not read
allok &= run_scenario("B noisy floor: S1 FAIL", make(4, {1:0.2,2:0.2,3:0.2}, TRUTH), make(5,{1:0.6,2:0.6,3:0.6},TRUTH), make(6,{1:0.1,2:0.1,3:0.1},TRUTH),
    {"S1":"FAIL"})
# C: no temperature effect -> S2 not detected
allok &= run_scenario("C no temp effect: S2 NOT DETECTED", make(7,{1:0,2:0,3:0},TRUTH), make(8,{1:0.05,2:0.1,3:0.1},TRUTH), make(9,{1:0.05,2:0.05,3:0.05},TRUTH),
    {"S1":"PASS","S2":"NOT DETECTED","S3":"NOT DETECTED","S4":"NOT DETECTED"})
# D: Simpson trap -> pooled S2 diff >= 30pp but one grade negative: must be NOT DETECTED
t7d = make(10,{1:0.55,2:0.7,3:0.7},TRUTH)
for r in t7d:
    if r["grade"]==2 and r["rep"]<3: r["rank"] = 1      # make G2 perfectly stable at temp 0.7 (flip 0)
allok &= run_scenario("D Simpson trap: pooled large, one grade flat -> S2 NOT DETECTED",
    make(11,{1:0,2:0,3:0},TRUTH), t7d, make(12,{1:0.1,2:0.1,3:0.1},TRUTH),
    {"S1":"PASS","S2":"NOT DETECTED","S3":"NOT DETECTED","S4":"PASS"})
print("\n(first four scenarios done; S-4 reachability check follows)")

# ---- S-4 is only reachable if the per-puzzle draws are BIASED coins (majority vote cannot help fair coins).
# Verify the gate CAN fire on biased coins, otherwise it is a dead gate.
def make_biased(seed, p_bias_by_grade, truth):
    """unstable puzzles draw their top-1 with P(correct)=p_bias (e.g. 0.7) instead of 0.5."""
    rnd = random.Random(seed); rows = []
    for (p,g),t in truth.items():
        pb = p_bias_by_grade[g]
        reps = [int(rnd.random() < pb) if t else int(rnd.random() > pb) for _ in range(5)]
        if len(set(reps)) == 1 and rnd.random() < 0.3: reps[rnd.randrange(5)] = 1 - reps[0]
        for k,v in enumerate(reps): rows.append({"puzzle":p,"mech":"c","grade":g,"rep":k,"temp":0,"rank":1 if v else 3,
                                                 "top3":1,"dup":0,"difficulty":"medium","secs":1.0,"finish_reason":"stop","text":""})
    return rows
print("\n--- S-4 reachability: biased coins (P(correct)=0.75) -> majority vote must help ---")
run_scenario("E biased coins: S4 PASS", make(21,{1:0,2:0,3:0},TRUTH), make_biased(22,{1:0.75,2:0.75,3:0.75},TRUTH), make_biased(23,{1:0.9,2:0.9,3:0.9},TRUTH),
             {"S1":"PASS","S2":"PASS","S3":"PASS","S4":"PASS"})
