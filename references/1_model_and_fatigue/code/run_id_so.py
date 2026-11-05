# pyright: reportMissingImports=false
"""
STAGE 2 - Biomechanical analysis: Inverse Dynamics (ID) + Static Optimization (SO)
==================================================================================

Generates the physics-based AI labels for the fatigue project, driven by the
Stage-1 minimum-jerk motion.

WHY ID + SO (not CMC) for this case
-----------------------------------
* ID gives the net elbow/shoulder joint torque (a label + sanity check).
* SO distributes that torque among the redundant muscles -> per-muscle activation
  and force, the core internal labels a camera cannot see.
* SO is justified for a CYCLIC task (Anderson & Pandy 2001: SO ~ dynamic
  optimization) and is the only method that scales to a multi-subject dataset.
  Its cost (min sum activation^2) is Crowninshield & Brand's (1981) fatigue proxy.
* SO is also the substrate for the later 3CC fatigue coupling (re-run SO with
  Fmax scaled by the available-force fraction; Frey-Law 2012 / Carbone 2023).
* CMC (activation dynamics) is reserved for a small validation subset only -
  ~30 min/half-cycle historically, far too slow for the whole corpus.

Setup choices
-------------
* Low-pass the kinematics at 6 Hz: removes the 6-12 Hz physiological tremor
  (which would blow up ID torques / SO activations) while preserving the curl and
  the fatigue trends (<1 Hz). The raw tremor stays in the *vision* channel.
* Weaken the elbow reserve actuator to ~1 N.m so the muscles carry the elbow load
  (Hicks 2015: reserves should stay small; large reserve use = a red flag).

Run:  conda run -n biomech python run_id_so.py
"""
import os
import math
import opensim as osim

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "Model", "arm26_paper_loaded_brd_elbow_research.osim")
MOTION = os.path.join(HERE, "..", "Data", "paper_minjerk_fatigue_10cycles.mot")
DATASET = os.path.join(HERE, "..", "Data", "paper_minjerk_fatigue_10cycles_dataset.csv")
RESULTS = os.path.join(HERE, "..", "Results", "Stage2")
LOWPASS_HZ = 6.0
ELBOW_RESERVE_NM = 1.0     # weak reserve so muscles do the elbow work

os.makedirs(RESULTS, exist_ok=True)


def motion_time_range(mot):
    t = []
    with open(mot) as f:
        for i, line in enumerate(f):
            if i < 7:
                continue
            p = line.split()
            if len(p) >= 1:
                try:
                    t.append(float(p[0]))
                except ValueError:
                    pass
    return t[0], t[-1]


def read_sto(path):
    """Parse an OpenSim .sto/.mot into (labels, list-of-rows)."""
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines) and lines[i].strip().lower() != "endheader":
        i += 1
    cols = lines[i + 1].split()
    rows = []
    for line in lines[i + 2:]:
        p = line.split()
        if len(p) == len(cols):
            rows.append([float(x) for x in p])
    return cols, rows


def col(cols, rows, name):
    j = cols.index(name)
    return [r[j] for r in rows]


def prep_model():
    """Load model, weaken the elbow reserve, save a copy for the tools to use."""
    model = osim.Model(MODEL)
    fs = model.getForceSet()
    for i in range(fs.getSize()):
        act = fs.get(i)
        if act.getName() == "elbow_assist":
            ca = osim.CoordinateActuator.safeDownCast(act)
            ca.setOptimalForce(ELBOW_RESERVE_NM)
    model.initSystem()
    out = os.path.join(RESULTS, "model_for_so.osim")
    model.printToXML(out)
    print("Prepared SO model (elbow reserve = %.1f N.m): %s" % (ELBOW_RESERVE_NM, out))
    return out


def run_id(model_path, t0, t1):
    idt = osim.InverseDynamicsTool()
    idt.setModelFileName(model_path)
    idt.setCoordinatesFileName(MOTION)
    idt.setLowpassCutoffFrequency(LOWPASS_HZ)
    excl = osim.ArrayStr()
    excl.append("Muscles")
    idt.setExcludedForces(excl)
    idt.setStartTime(t0)
    idt.setEndTime(t1)
    idt.setResultsDir(RESULTS)
    idt.setOutputGenForceFileName("ID_genforces.sto")
    print("\n[ID] running inverse dynamics ...")
    idt.run()
    return os.path.join(RESULTS, "ID_genforces.sto")


def run_so(model_path, t0, t1):
    model = osim.Model(model_path)
    so = osim.StaticOptimization()
    so.setName("SO")
    so.setUseModelForceSet(True)
    so.setActivationExponent(2.0)
    so.setUseMusclePhysiology(True)
    so.setStartTime(t0)
    so.setEndTime(t1)
    model.addAnalysis(so)
    model.initSystem()

    tool = osim.AnalyzeTool(model)
    tool.setName("SO")
    tool.setModel(model)
    tool.setInitialTime(t0)
    tool.setFinalTime(t1)
    tool.setLowpassCutoffFrequency(LOWPASS_HZ)
    tool.setCoordinatesFileName(MOTION)
    tool.setLoadModelAndInput(True)
    tool.setResultsDir(RESULTS)
    print("[SO] running static optimization ...")
    tool.run()
    return (os.path.join(RESULTS, "SO_SO_activation.sto"),
            os.path.join(RESULTS, "SO_SO_force.sto"))


def merge_and_report(id_sto, act_sto, frc_sto):
    # fatigue level + rep per frame from the Stage-1 dataset
    import csv
    drows = list(csv.DictReader(open(DATASET)))
    dts = [float(r["time"]) for r in drows]

    def fat_rep(t):
        k = min(range(len(dts)), key=lambda i: abs(dts[i] - t))
        return float(drows[k]["fatigue_level"]), int(drows[k]["rep_index"])

    idc, idr = read_sto(id_sto)
    ac, ar = read_sto(act_sto)
    fc, fr = read_sto(frc_sto)
    t = col(ac, ar, "time")

    # elbow torque from ID (column name ends with _moment)
    elcol = [c for c in idc if "elbow" in c and "moment" in c][0]
    idt = col(idc, idr, "time")
    elbow_moment = col(idc, idr, elcol)

    muscles = ["TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA", "BRD_hand"]
    flex = ["BIClong", "BICshort", "BRA", "BRD_hand"]

    out = os.path.join(RESULTS, "stage2_labels.csv")
    with open(out, "w", newline="\n") as f:
        hdr = (["time", "elbow_moment_Nm"]
               + ["act_%s" % m for m in muscles + ["elbow_assist"]]
               + ["frc_%s" % m for m in muscles]
               + ["rep_index", "fatigue_level"])
        f.write(",".join(hdr) + "\n")
        for i, ti in enumerate(t):
            j = min(range(len(idt)), key=lambda k: abs(idt[k] - ti))
            fl, rp = fat_rep(ti)
            vals = [ti, elbow_moment[j]]
            for m in muscles + ["elbow_assist"]:
                vals.append(col(ac, ar, m)[i] if m in ac else 0.0)
            for m in muscles:
                vals.append(col(fc, fr, m)[i] if m in fc else 0.0)
            vals += [rp, fl]
            f.write(",".join("%.5f" % v if isinstance(v, float) else str(v) for v in vals) + "\n")

    # ---- validation summary ----
    print("\n================ VALIDATION SUMMARY ================")
    print("ID  elbow moment   : peak %.2f N.m" % max(abs(v) for v in elbow_moment))
    rea = col(ac, ar, "elbow_assist") if "elbow_assist" in ac else [0]
    print("SO  elbow RESERVE  : max activation %.3f (should be small)" % max(abs(v) for v in rea))
    print("SO  peak activations:")
    for m in muscles:
        if m in ac:
            print("     %-9s %.3f" % (m, max(col(ac, ar, m))))
    # flexor force sharing at the most-loaded frame
    jpk = max(range(len(elbow_moment)), key=lambda k: abs(elbow_moment[k]))
    tpk = idt[jpk]
    ipk = min(range(len(t)), key=lambda k: abs(t[k] - tpk))
    print("SO  flexor FORCE share at peak load (t=%.2fs):" % tpk)
    tot = sum(col(fc, fr, m)[ipk] for m in flex if m in fc)
    for m in flex:
        if m in fc:
            fv = col(fc, fr, m)[ipk]
            print("     %-9s %7.1f N  (%4.1f%%)" % (m, fv, 100 * fv / tot if tot else 0))
    print("Wrote labels: %s" % out)
    print("====================================================")


if __name__ == "__main__":
    t0, t1 = motion_time_range(MOTION)
    print("Motion %s  (t=%.2f..%.2f s)" % (os.path.basename(MOTION), t0, t1))
    mpath = prep_model()
    id_sto = run_id(mpath, t0, t1)
    act_sto, frc_sto = run_so(mpath, t0, t1)
    merge_and_report(id_sto, act_sto, frc_sto)
