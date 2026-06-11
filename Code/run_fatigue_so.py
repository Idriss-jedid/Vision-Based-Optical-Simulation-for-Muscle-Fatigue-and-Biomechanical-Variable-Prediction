# pyright: reportMissingImports=false
"""
STAGE 2 - 3CC-coupled Static Optimization  (the fatigue label core)
===================================================================

ID + SO alone gave a near-flat activation across reps, because Stage-1 kinematic
degradation REDUCES mechanical demand. Real fatigue is the opposite: capacity
falls while demand is held, so activation RISES (Potvin & Bent 2010: aEMG up,
MPF -25..29%). This script adds that missing mechanism.

Method (post-hoc 3CC + SO coupling, the scheme of Related-Work 2.2.8):
  1. Start from the baseline SO outputs (stage2_labels.csv): per-flexor force F0
     and activation a0, and the ID elbow moment M(t).
  2. Per-frame force capacity (incl. force-length/velocity) inferred from SO:
         Cap0_m(t) = F0_m / max(a0_m, eps)        [= Fmax * fL * fV]
  3. Three-compartment controller (Xia & Frey-Law 2008) per flexor, ELBOW rates
     from Frey-Law et al. (2012):  F = 0.00912 /s,  R = 0.00094 /s
         dMF/dt = F * MA - R * MF        (MA, MF in % of motor units)
         capacity(t) = 1 - MF(t)/100
  4. Re-solve elbow load-sharing each frame with the fatigued bound:
         min  sum (F_m / Cap0_m)^2          (= sum activation^2, Crowninshield-Brand)
         s.t. sum r_m(theta) * F_m = M(t)
              0 <= F_m <= capacity_m(t) * Cap0_m         (can't exceed fatigued max)
     The active fraction MA = 100 * (F_m / Cap0_m) then drives the next dMF step
     (positive feedback: more fatigue -> more activation -> more fatigue).
  If the flexors' summed capacity < demand -> task failure (saturation) is flagged.

Moment arms r_m(theta) are computed analytically (validated virtual-work method).
Needs scipy -> run in the biomech env.

Run:  conda run -n biomech python run_fatigue_so.py
"""
import os
import csv
import math
import xml.etree.ElementTree as ET
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "Model", "arm26_paper_loaded_brd_elbow_research.osim")
MOTION = os.path.join(HERE, "..", "Data", "paper_minjerk_fatigue_10cycles.mot")
LABELS = os.path.join(HERE, "..", "Results", "Stage2", "stage2_labels.csv")
OUT = os.path.join(HERE, "..", "Results", "Stage2", "stage2_fatigue_labels.csv")

F_ELBOW = 0.00912    # Frey-Law 2012 elbow fatigue rate (1/s)
R_ELBOW = 0.00094    # Frey-Law 2012 elbow recovery rate (1/s)
FLEXORS = ["BIClong", "BICshort", "BRA", "BRD_hand"]
SHOULDER_DEG = 20.0
DT = 0.01


# ---- geometry: analytic moment arms r_m(theta) -----------------------------
def vec(t): return tuple(float(x) for x in t.split())
def nrm(a):
    n = math.sqrt(sum(x * x for x in a)); return tuple(x / n for x in a)
def rod(a, th, v):
    ax, ay, az = a; vx, vy, vz = v; c, s = math.cos(th), math.sin(th)
    dot = ax * vx + ay * vy + az * vz
    cx, cy, cz = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx
    return (vx*c+cx*s+ax*dot*(1-c), vy*c+cy*s+ay*dot*(1-c), vz*c+cz*s+az*dot*(1-c))
def add(p, q): return (p[0]+q[0], p[1]+q[1], p[2]+q[2])
def sub(p, q): return (p[0]-q[0], p[1]-q[1], p[2]-q[2])
def dist(p, q): return math.sqrt(sum((p[i]-q[i])**2 for i in range(3)))


def parse_joint(root, name, off_name):
    axis = off = None
    for j in root.iter("CustomJoint"):
        if j.get("name") != name: continue
        for ta in j.iter("TransformAxis"):
            if ta.get("name") == "rotation1": axis = nrm(vec(ta.find("axis").text))
        for pof in j.iter("PhysicalOffsetFrame"):
            if pof.get("name") == off_name: off = vec(pof.find("translation").text)
    return axis, off


def build_moment_arm_fns():
    root = ET.parse(MODEL).getroot()
    a_s, Ts = parse_joint(root, "r_shoulder", "base_offset")
    a_e, Te = parse_joint(root, "r_elbow", "r_humerus_offset")
    ths = math.radians(SHOULDER_DEG)

    def to_hum(body, P, the):
        if body == "r_humerus": return P
        if body == "r_ulna_radius_hand": return add(Te, rod(a_e, the, P))
        if body == "base": return rod(a_s, -ths, sub(P, Ts))
        return P

    paths = {}
    for m in root.iter("Thelen2003Muscle"):
        if m.get("name") in FLEXORS:
            pts = [(pp.find("socket_parent_frame").text.split("/")[-1],
                    vec(pp.find("location").text)) for pp in m.iter("PathPoint")]
            paths[m.get("name")] = pts

    def mtu(name, the):
        P = [to_hum(b, p, the) for b, p in paths[name]]
        return sum(dist(P[i], P[i+1]) for i in range(len(P)-1))

    def r_of(name, deg):
        th = math.radians(deg); d = math.radians(0.1)
        return -(mtu(name, th+d) - mtu(name, th-d)) / (2*d)   # m

    return r_of


def read_labels():
    rows = list(csv.DictReader(open(LABELS)))
    return rows


def read_elbow_angle():
    ang = []
    with open(MOTION) as f:
        for i, line in enumerate(f):
            if i < 7: continue
            p = line.split()
            if len(p) >= 3: ang.append(float(p[2]))
    return ang


def solve_frame(M, r, cap0, cap):
    """min sum (F_m/cap0_m)^2  s.t. sum r_m F_m = M, 0<=F_m<=cap*cap0."""
    ub = [cap[m] * cap0[m] for m in FLEXORS]
    rr = [r[m] for m in FLEXORS]
    Mmax = sum(rr[i] * ub[i] for i in range(len(FLEXORS)))
    if M <= 0:
        return {m: 0.0 for m in FLEXORS}, False
    if M >= Mmax:                      # task failure: everything saturates
        return {FLEXORS[i]: ub[i] for i in range(len(FLEXORS))}, True
    w = [1.0 / (cap0[m] ** 2) for m in FLEXORS]
    obj = lambda x: sum(w[i] * x[i] * x[i] for i in range(len(x)))
    jac = lambda x: [2 * w[i] * x[i] for i in range(len(x))]
    cons = [{"type": "eq",
             "fun": lambda x: sum(rr[i] * x[i] for i in range(len(x))) - M,
             "jac": lambda x: rr}]
    x0 = [min(ub[i], max(0.0, M / (rr[i] * len(FLEXORS)) if rr[i] > 1e-6 else 0)) for i in range(len(FLEXORS))]
    res = minimize(obj, x0, jac=jac, bounds=[(0, ub[i]) for i in range(len(FLEXORS))],
                   constraints=cons, method="SLSQP", options={"ftol": 1e-9, "maxiter": 80})
    return {FLEXORS[i]: max(0.0, res.x[i]) for i in range(len(FLEXORS))}, (not res.success)


def main():
    r_of = build_moment_arm_fns()
    rows = read_labels()
    ang = read_elbow_angle()
    n = min(len(rows), len(ang))

    MF = {m: 0.0 for m in FLEXORS}     # % fatigued
    out_rows = []
    fail_frames = 0
    for i in range(n):
        t = float(rows[i]["time"])
        M = float(rows[i]["elbow_moment_Nm"])
        deg = ang[i]
        r = {m: r_of(m, deg) for m in FLEXORS}
        cap0 = {m: float(rows[i]["frc_" + m]) / max(float(rows[i]["act_" + m]), 0.05) for m in FLEXORS}
        cap = {m: max(1e-3, 1.0 - MF[m] / 100.0) for m in FLEXORS}
        F, failed = solve_frame(M, r, cap0, cap)
        fail_frames += int(failed)
        a_fat = {m: (F[m] / cap0[m] if cap0[m] > 1e-6 else 0.0) for m in FLEXORS}
        # advance 3CC: active fraction MA = 100*a_fat drives fatigue
        for m in FLEXORS:
            MA = 100.0 * a_fat[m]
            MF[m] += DT * (F_ELBOW * MA - R_ELBOW * MF[m])
            MF[m] = min(max(MF[m], 0.0), 100.0)
        rec = {"time": t, "rep_index": rows[i]["rep_index"],
               "fatigue_level": rows[i]["fatigue_level"], "elbow_moment_Nm": M,
               "task_failure": int(failed)}
        for m in FLEXORS:
            rec["a0_" + m] = float(rows[i]["act_" + m])
            rec["afat_" + m] = a_fat[m]
            rec["MF_" + m] = MF[m]
            rec["cap_" + m] = cap[m]
        out_rows.append(rec)

    cols = (["time", "rep_index", "fatigue_level", "elbow_moment_Nm", "task_failure"]
            + sum([["a0_" + m, "afat_" + m, "MF_" + m, "cap_" + m] for m in FLEXORS], []))
    with open(OUT, "w", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for rec in out_rows:
            f.write(",".join(("%.5f" % rec[c] if isinstance(rec[c], float) else str(rec[c])) for c in cols) + "\n")

    # ---- report ----
    reps = sorted(set(int(r["rep_index"]) for r in out_rows))
    def rep_mean(rep, key):
        v = [r[key] for r in out_rows if int(r["rep_index"]) == rep]
        return sum(v) / len(v) if v else 0.0
    print("=========== 3CC-COUPLED SO (fatigue label) ===========")
    print("elbow F=%.5f /s  R=%.5f /s  (Frey-Law 2012)" % (F_ELBOW, R_ELBOW))
    print("frames=%d  task-failure frames=%d" % (n, fail_frames))
    print("\nrep   mean a_fresh   mean a_fatigued   rise%%   meanMF(BIClong)")
    a_fresh_sum = lambda rep: sum(rep_mean(rep, "a0_" + m) for m in FLEXORS)
    a_fat_sum = lambda rep: sum(rep_mean(rep, "afat_" + m) for m in FLEXORS)
    for k in reps:
        af, ff = a_fresh_sum(k), a_fat_sum(k)
        print("%3d      %.3f          %.3f         %+5.1f    %.1f%%"
              % (k, af, ff, 100 * (ff - af) / af if af else 0, rep_mean(k, "MF_BIClong")))
    r1, rN = reps[0], reps[-1]
    print("\nfatigued activation rise rep%d->rep%d: %+.1f%%  (Potvin/Bent: aEMG rises)"
          % (r1, rN, 100 * (a_fat_sum(rN) - a_fat_sum(r1)) / a_fat_sum(r1)))
    print("capacity decline (BIClong): %.1f%% -> %.1f%% fatigued MUs"
          % (rep_mean(r1, "MF_BIClong"), rep_mean(rN, "MF_BIClong")))
    print("wrote", OUT)
    print("======================================================")


if __name__ == "__main__":
    main()
