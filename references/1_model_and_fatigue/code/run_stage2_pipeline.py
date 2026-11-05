# pyright: reportMissingImports=false
"""
STAGE 2 - Full biomechanical pipeline orchestrator  (publication grade)
=======================================================================

Runs the complete chain  ID -> SO -> 3CC-coupled SO  for several conditions and
consolidates a literature-anchored validation:

  conditions:
    loaded_10rep    arm26 + 2 kg, 10-rep motion     (primary)
    unloaded_10rep  arm26 + 0 kg, 10-rep motion     (Gastaldi 0-vs-2 kg control)
    loaded_30rep    arm26 + 2 kg, 30-rep motion     (pronounced fatigue protocol)

For each condition:
  1. Inverse Dynamics  (6 Hz low-pass; welded load -> gravity auto)   -> joint torques
  2. Static Optimization (weak 1 N.m elbow reserve so muscles work)   -> activations/forces
  3. 3CC-coupled SO (Xia & Frey-Law 2008; elbow F/R Frey-Law 2012)    -> fatigue labels
  -> Results/Stage2/<tag>/stage2_labels.csv  and  stage2_fatigue_labels.csv

Validation (Hicks 2015 + Gastaldi 2021 + Potvin & Bent 2010):
  - reserve actuator use small;  antagonist co-contraction low
  - BRD load share present;  activation rises with LOAD (0 vs 2 kg)
  - capacity declines / activation rises with REPETITIONS (fatigue)

Run:  conda run -n biomech python run_stage2_pipeline.py
"""
import os
import csv
import math
import xml.etree.ElementTree as ET
import opensim as osim
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "Model")
DATA = os.path.join(HERE, "..", "Data")
RESULTS = os.path.join(HERE, "..", "Results", "Stage2")
LOWPASS_HZ = 6.0
ELBOW_RESERVE_NM = 1.0
SHOULDER_DEG = 20.0
F_ELBOW, R_ELBOW = 0.00912, 0.00094      # Frey-Law 2012 elbow rates (1/s)
FLEXORS = ["BIClong", "BICshort", "BRA", "BRD_hand"]
MUSCLES = ["TRIlong", "TRIlat", "TRImed"] + FLEXORS
DT = 0.01

JOBS = [
    ("loaded_10rep",    "arm26_paper_loaded_brd_elbow_research.osim",     "paper_minjerk_fatigue_10cycles.mot"),
    ("unloaded_10rep",  "arm26_paper_loaded_brd_elbow_research_0kg.osim", "paper_minjerk_fatigue_10cycles.mot"),
    ("loaded_30rep",    "arm26_paper_loaded_brd_elbow_research.osim",     "paper_minjerk_fatigue_30cycles.mot"),
    ("loaded4kg_30rep", "arm26_paper_loaded_brd_elbow_research_4kg.osim", "paper_minjerk_fatigue_30cycles.mot"),
]


# ----------------------------- helpers --------------------------------------
def read_sto(path):
    L = open(path).read().splitlines(); i = 0
    while L[i].strip().lower() != "endheader": i += 1
    cols = L[i + 1].split()
    rows = [[float(x) for x in ln.split()] for ln in L[i + 2:] if len(ln.split()) == len(cols)]
    return cols, rows


def col(cols, rows, n): j = cols.index(n); return [r[j] for r in rows]
def vec(t): return tuple(float(x) for x in t.split())
def nrm(a): n = math.sqrt(sum(x * x for x in a)); return tuple(x / n for x in a)
def rod(a, th, v):
    ax, ay, az = a; vx, vy, vz = v; c, s = math.cos(th), math.sin(th)
    dot = ax * vx + ay * vy + az * vz
    cx, cy, cz = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx
    return (vx*c+cx*s+ax*dot*(1-c), vy*c+cy*s+ay*dot*(1-c), vz*c+cz*s+az*dot*(1-c))
def add(p, q): return (p[0]+q[0], p[1]+q[1], p[2]+q[2])
def sub(p, q): return (p[0]-q[0], p[1]-q[1], p[2]-q[2])
def dist(p, q): return math.sqrt(sum((p[i]-q[i])**2 for i in range(3)))


def motion_range(mot):
    t = []
    with open(mot) as f:
        for i, line in enumerate(f):
            if i < 7: continue
            p = line.split()
            if p:
                try: t.append(float(p[0]))
                except ValueError: pass
    return t[0], t[-1]


def elbow_angles(mot):
    a = []
    with open(mot) as f:
        for i, line in enumerate(f):
            if i < 7: continue
            p = line.split()
            if len(p) >= 3: a.append(float(p[2]))
    return a


# --------------------------- ID + SO ----------------------------------------
def prep_model(model_path, outdir):
    model = osim.Model(model_path)
    fs = model.getForceSet()
    for i in range(fs.getSize()):
        if fs.get(i).getName() == "elbow_assist":
            osim.CoordinateActuator.safeDownCast(fs.get(i)).setOptimalForce(ELBOW_RESERVE_NM)
    model.initSystem()
    out = os.path.join(outdir, "model_for_so.osim")
    model.printToXML(out)
    return out


def run_id(model_path, motion, outdir, t0, t1):
    out = os.path.join(outdir, "ID_genforces.sto")
    if os.path.exists(out):
        return out
    idt = osim.InverseDynamicsTool()
    idt.setModelFileName(model_path)
    idt.setCoordinatesFileName(motion)
    idt.setLowpassCutoffFrequency(LOWPASS_HZ)
    ex = osim.ArrayStr(); ex.append("Muscles"); idt.setExcludedForces(ex)
    idt.setStartTime(t0); idt.setEndTime(t1)
    idt.setResultsDir(outdir); idt.setOutputGenForceFileName("ID_genforces.sto")
    idt.run()
    return out


def run_so(model_path, motion, outdir, tag, t0, t1):
    act = os.path.join(outdir, "%s_SO_activation.sto" % tag)
    frc = os.path.join(outdir, "%s_SO_force.sto" % tag)
    if os.path.exists(act) and os.path.exists(frc):
        return act, frc
    model = osim.Model(model_path)
    so = osim.StaticOptimization(); so.setName("SO")
    so.setUseModelForceSet(True); so.setActivationExponent(2.0); so.setUseMusclePhysiology(True)
    so.setStartTime(t0); so.setEndTime(t1)
    model.addAnalysis(so); model.initSystem()
    tool = osim.AnalyzeTool(model); tool.setName(tag); tool.setModel(model)
    tool.setInitialTime(t0); tool.setFinalTime(t1)
    tool.setLowpassCutoffFrequency(LOWPASS_HZ)
    tool.setCoordinatesFileName(motion); tool.setLoadModelAndInput(True)
    tool.setResultsDir(outdir)
    tool.run()
    return act, frc


def merge_labels(id_sto, act_sto, frc_sto, dataset_csv, outdir):
    idc, idr = read_sto(id_sto); ac, ar = read_sto(act_sto); fc, fr = read_sto(frc_sto)
    t = col(ac, ar, "time"); idt = col(idc, idr, "time")
    elcol = [c for c in idc if "elbow" in c and "moment" in c][0]
    elm = col(idc, idr, elcol)
    drows = list(csv.DictReader(open(dataset_csv))); dts = [float(r["time"]) for r in drows]
    out = os.path.join(outdir, "stage2_labels.csv")
    with open(out, "w", newline="\n") as f:
        hdr = (["time", "elbow_moment_Nm"] + ["act_" + m for m in MUSCLES + ["elbow_assist"]]
               + ["frc_" + m for m in MUSCLES] + ["rep_index", "fatigue_level"])
        f.write(",".join(hdr) + "\n")
        for i, ti in enumerate(t):
            j = min(range(len(idt)), key=lambda k: abs(idt[k] - ti))
            k = min(range(len(dts)), key=lambda z: abs(dts[z] - ti))
            row = [ti, elm[j]] + [col(ac, ar, m)[i] if m in ac else 0.0 for m in MUSCLES + ["elbow_assist"]] \
                  + [col(fc, fr, m)[i] if m in fc else 0.0 for m in MUSCLES] \
                  + [int(drows[k]["rep_index"]), float(drows[k]["fatigue_level"])]
            f.write(",".join("%.5f" % v if isinstance(v, float) else str(v) for v in row) + "\n")
    return out


# --------------------------- 3CC-coupled SO ---------------------------------
def moment_arm_fn(model_path):
    root = ET.parse(model_path).getroot()
    def pj(name, off):
        ax = of = None
        for j in root.iter("CustomJoint"):
            if j.get("name") != name: continue
            for ta in j.iter("TransformAxis"):
                if ta.get("name") == "rotation1": ax = nrm(vec(ta.find("axis").text))
            for p in j.iter("PhysicalOffsetFrame"):
                if p.get("name") == off: of = vec(p.find("translation").text)
        return ax, of
    a_s, Ts = pj("r_shoulder", "base_offset"); a_e, Te = pj("r_elbow", "r_humerus_offset")
    ths = math.radians(SHOULDER_DEG)
    def to_hum(b, P, th):
        if b == "r_humerus": return P
        if b == "r_ulna_radius_hand": return add(Te, rod(a_e, th, P))
        if b == "base": return rod(a_s, -ths, sub(P, Ts))
        return P
    paths = {}
    for m in root.iter("Thelen2003Muscle"):
        if m.get("name") in FLEXORS:
            paths[m.get("name")] = [(pp.find("socket_parent_frame").text.split("/")[-1],
                                     vec(pp.find("location").text)) for pp in m.iter("PathPoint")]
    def mtu(name, th):
        P = [to_hum(b, p, th) for b, p in paths[name]]
        return sum(dist(P[i], P[i+1]) for i in range(len(P)-1))
    def r_of(name, deg):
        th = math.radians(deg); d = math.radians(0.1)
        return -(mtu(name, th+d) - mtu(name, th-d)) / (2*d)
    return r_of


def solve_frame(M, r, cap0, cap):
    ub = [cap[m]*cap0[m] for m in FLEXORS]; rr = [r[m] for m in FLEXORS]
    if M <= 0: return {m: 0.0 for m in FLEXORS}, False
    if M >= sum(rr[i]*ub[i] for i in range(4)): return {FLEXORS[i]: ub[i] for i in range(4)}, True
    w = [1.0/(cap0[m]**2) for m in FLEXORS]
    obj = lambda x: sum(w[i]*x[i]*x[i] for i in range(4))
    jac = lambda x: [2*w[i]*x[i] for i in range(4)]
    cons = [{"type": "eq", "fun": lambda x: sum(rr[i]*x[i] for i in range(4)) - M, "jac": lambda x: rr}]
    x0 = [min(ub[i], max(0.0, M/(rr[i]*4) if rr[i] > 1e-6 else 0)) for i in range(4)]
    res = minimize(obj, x0, jac=jac, bounds=[(0, ub[i]) for i in range(4)],
                   constraints=cons, method="SLSQP", options={"ftol": 1e-9, "maxiter": 80})
    return {FLEXORS[i]: max(0.0, res.x[i]) for i in range(4)}, (not res.success)


def run_3cc(labels_csv, model_path, motion, outdir):
    r_of = moment_arm_fn(model_path)
    rows = list(csv.DictReader(open(labels_csv))); ang = elbow_angles(motion)
    n = min(len(rows), len(ang)); MF = {m: 0.0 for m in FLEXORS}; out_rows = []; fails = 0
    for i in range(n):
        M = float(rows[i]["elbow_moment_Nm"]); deg = ang[i]
        r = {m: r_of(m, deg) for m in FLEXORS}
        cap0 = {m: float(rows[i]["frc_"+m])/max(float(rows[i]["act_"+m]), 0.05) for m in FLEXORS}
        cap = {m: max(1e-3, 1.0 - MF[m]/100.0) for m in FLEXORS}
        F, failed = solve_frame(M, r, cap0, cap); fails += int(failed)
        a_fat = {m: (F[m]/cap0[m] if cap0[m] > 1e-6 else 0.0) for m in FLEXORS}
        for m in FLEXORS:
            MA = 100.0*a_fat[m]; MF[m] += DT*(F_ELBOW*MA - R_ELBOW*MF[m]); MF[m] = min(max(MF[m], 0.0), 100.0)
        rec = {"time": float(rows[i]["time"]), "rep_index": rows[i]["rep_index"],
               "elbow_moment_Nm": M, "task_failure": int(failed)}
        for m in FLEXORS:
            rec["a0_"+m] = float(rows[i]["act_"+m]); rec["afat_"+m] = a_fat[m]
            rec["MF_"+m] = MF[m]; rec["cap_"+m] = cap[m]
        out_rows.append(rec)
    cols = (["time", "rep_index", "elbow_moment_Nm", "task_failure"]
            + sum([["a0_"+m, "afat_"+m, "MF_"+m, "cap_"+m] for m in FLEXORS], []))
    out = os.path.join(outdir, "stage2_fatigue_labels.csv")
    with open(out, "w", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for rc in out_rows:
            f.write(",".join("%.5f" % rc[c] if isinstance(rc[c], float) else str(rc[c]) for c in cols) + "\n")
    return out, fails


# ------------------------------ driver --------------------------------------
def run_job(tag, model_file, motion_file):
    outdir = os.path.join(RESULTS, tag); os.makedirs(outdir, exist_ok=True)
    model_path = os.path.join(MODELS, model_file); motion = os.path.join(DATA, motion_file)
    dataset = os.path.join(DATA, motion_file.replace(".mot", "_dataset.csv"))
    t0, t1 = motion_range(motion)
    print("\n##### JOB %s  (%s, %s)  t=%.1f..%.1f #####" % (tag, model_file, motion_file, t0, t1))
    mp = prep_model(model_path, outdir)
    id_sto = run_id(mp, motion, outdir, t0, t1); print("  ID done")
    act, frc = run_so(mp, motion, outdir, tag, t0, t1); print("  SO done")
    labels = merge_labels(id_sto, act, frc, dataset, outdir); print("  labels merged")
    fat, fails = run_3cc(labels, model_path, motion, outdir); print("  3CC done (fail frames=%d)" % fails)
    return dict(tag=tag, labels=labels, fatigue=fat, id=id_sto)


def summarize(job):
    ac, ar = read_sto(os.path.join(RESULTS, job["tag"], "%s_SO_activation.sto" % job["tag"]))
    fc, fr = read_sto(os.path.join(RESULTS, job["tag"], "%s_SO_force.sto" % job["tag"]))
    idc, idr = read_sto(job["id"]); elcol = [c for c in idc if "elbow" in c and "moment" in c][0]
    elm = col(idc, idr, elcol)
    res = {}
    res["peak_elbow_Nm"] = max(abs(v) for v in elm)
    res["reserve_max"] = max(abs(v) for v in col(ac, ar, "elbow_assist")) if "elbow_assist" in ac else 0
    res["flex_act_mean"] = sum(sum(col(ac, ar, m)[i] for m in FLEXORS) for i in range(len(ar))) / len(ar)
    res["tri_act_mean"] = sum(sum(col(ac, ar, m)[i] for m in ["TRIlong", "TRIlat", "TRImed"]) for i in range(len(ar))) / len(ar)
    jpk = max(range(len(elm)), key=lambda k: abs(elm[k]))
    tot = sum(col(fc, fr, m)[jpk] for m in FLEXORS) or 1
    res["brd_share_pct"] = 100 * col(fc, fr, "BRD_hand")[jpk] / tot
    # fatigue trend
    frows = list(csv.DictReader(open(job["fatigue"])))
    reps = sorted(set(int(r["rep_index"]) for r in frows))
    def repmean(rep, key): v = [float(r[key]) for r in frows if int(r["rep_index"]) == rep]; return sum(v)/len(v) if v else 0
    af1 = sum(repmean(reps[0], "afat_"+m) for m in FLEXORS); afN = sum(repmean(reps[-1], "afat_"+m) for m in FLEXORS)
    res["act_rise_pct"] = 100*(afN-af1)/af1 if af1 else 0
    res["MF_end_BIClong"] = repmean(reps[-1], "MF_BIClong")
    res["task_fail"] = sum(int(r["task_failure"]) for r in frows)
    return res


def main():
    os.makedirs(RESULTS, exist_ok=True)
    jobs = [run_job(*j) for j in JOBS]
    print("\n================= STAGE 2 CONSOLIDATED VALIDATION =================")
    S = {j["tag"]: summarize(j) for j in jobs}
    hdr = "%-16s %10s %9s %10s %9s %9s %9s %9s"
    print(hdr % ("condition", "peakElbNm", "reserve", "flexActMu", "triActMu", "BRD%", "actRise%", "MF_end%"))
    for tag in S:
        s = S[tag]
        print(hdr % (tag, "%.2f" % s["peak_elbow_Nm"], "%.3f" % s["reserve_max"],
                     "%.3f" % s["flex_act_mean"], "%.3f" % s["tri_act_mean"],
                     "%.1f" % s["brd_share_pct"], "%+.1f" % s["act_rise_pct"], "%.1f" % s["MF_end_BIClong"]))
    print("\nVALIDATION CHECKS:")
    if "loaded_10rep" in S and "unloaded_10rep" in S:
        lo, un = S["loaded_10rep"]["flex_act_mean"], S["unloaded_10rep"]["flex_act_mean"]
        print("  [Gastaldi 0-vs-2kg] flexor activation 0kg %.3f -> 2kg %.3f  (%+.0f%% with load): %s"
              % (un, lo, 100*(lo-un)/un if un else 0, "PASS" if lo > un else "CHECK"))
    if "loaded_10rep" in S and "loaded_30rep" in S:
        m10, m30 = S["loaded_10rep"]["MF_end_BIClong"], S["loaded_30rep"]["MF_end_BIClong"]
        print("  [fatigue scales with reps] MF_end 10rep %.1f%% -> 30rep %.1f%%: %s"
              % (m10, m30, "PASS" if m30 > m10 else "CHECK"))
    allres = all(S[t]["reserve_max"] < 0.10 for t in S)
    print("  [Hicks reserve<0.10] all conditions: %s" % ("PASS" if allres else "CHECK"))
    allco = all(S[t]["tri_act_mean"] < 0.10 for t in S)
    print("  [low antagonist co-contraction] triceps mean<0.10: %s" % ("PASS" if allco else "CHECK"))
    print("===================================================================")


if __name__ == "__main__":
    main()
