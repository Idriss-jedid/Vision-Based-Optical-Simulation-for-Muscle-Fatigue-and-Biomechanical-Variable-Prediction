# pyright: reportMissingImports=false
"""
APPROCHE A — extraction des labels OpenSim (teacher) + cinématique, par sujet.
Pour chaque sujet de batch/ : ID (torque) + SO (forces/activations) + 3CC (fatigue MF)
sur curl.mot + arm26 scalé. Assemble batch/<subj>/labels_ml.csv :
  INPUT  : q_sh, q_el, qd_sh, qd_el, qdd_sh, qdd_el, time, rep
  OUTPUT : elbow_moment, act_<4flex>, frc_<4flex>, MF_<4flex>
Concatène tout dans batch/ml_dataset_A.csv. biomech env (lent : SO).
"""
import csv, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "Code"))
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "Code"))
import run_stage2_pipeline as P  # noqa: E402

BATCH = os.path.join(ROOT, "batch")
SUBJECTS = sys.argv[1:] if len(sys.argv) > 1 else ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]


def read_mot(path):
    L = open(path).read().splitlines()
    i = next(k for k, l in enumerate(L) if l.strip().lower() == "endheader")
    cols = L[i + 1].split("\t")
    d = np.array([[float(x) for x in r.split("\t")] for r in L[i + 2:] if r.strip()])
    return cols, d


def deriv(x, dt):
    v = np.gradient(x, dt); a = np.gradient(v, dt); return v, a


def process(subj):
    motion = os.path.join(BATCH, subj, "motion", "curl.mot")
    model = os.path.join(BATCH, subj, "opensim", "arm26_%s_scaled.osim" % subj)
    outdir = os.path.join(BATCH, subj, "labels"); os.makedirs(outdir, exist_ok=True)
    cols, d = read_mot(motion)
    t = d[:, 0]; qsh = d[:, cols.index("r_shoulder_elev")]; qel = d[:, cols.index("r_elbow_flex")]
    dt = float(np.median(np.diff(t)))
    # ds (time, rep_index, fatigue_level) — rep par seuil sur le coude
    thr = (qel.max() + qel.min()) / 2; above = qel > thr; rep = np.ones(len(t), int); r = 1
    for i in range(1, len(t)):
        if above[i] and not above[i - 1]: r += 1
        rep[i] = r
    ds = os.path.join(outdir, "ds.csv")
    with open(ds, "w", newline="\n") as f:
        f.write("time,rep_index,fatigue_level\n")
        for i in range(len(t)): f.write("%.4f,%d,0\n" % (t[i], rep[i]))
    # OpenSim ID + SO + 3CC
    t0, t1 = P.motion_range(motion)
    mp = P.prep_model(model, outdir)
    ids = P.run_id(mp, motion, outdir, t0, t1)
    act, frc = P.run_so(mp, motion, outdir, subj, t0, t1)
    labels = P.merge_labels(ids, act, frc, ds, outdir)
    fatcsv, fails = P.run_3cc(labels, model, motion, outdir)
    # read merged labels (act+frc+torque) + fatigue (MF)
    fcd = _read_csv(fatcsv)
    lab = _read_csv(labels)
    # kinematics
    qdsh, qddsh = deriv(qsh, dt); qdel, qddel = deriv(qel, dt)
    # build per-frame ML rows (align by index; SO frames == motion frames)
    n = min(len(lab["time"]), len(t))
    out = os.path.join(BATCH, subj, "labels_ml.csv")
    hdr = (["subj", "time", "rep", "q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el",
            "elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX])
    with open(out, "w", newline="\n") as f:
        f.write(",".join(hdr) + "\n")
        for i in range(n):
            row = [subj, t[i], rep[i], qsh[i], qel[i], qdsh[i], qdel[i], qddsh[i], qddel[i],
                   float(lab["elbow_moment_Nm"][i])]
            row += [float(lab["act_" + m][i]) for m in FLEX]
            row += [float(lab["frc_" + m][i]) for m in FLEX]
            row += [float(fcd["MF_" + m][i]) if ("MF_" + m) in fcd else 0.0 for m in FLEX]
            f.write(",".join(str(x) if not isinstance(x, float) else "%.5f" % x for x in row) + "\n")
    print("%s: labels_ml.csv (%d frames, fails=%d)" % (subj, n, fails))
    return out


def _read_csv(path):
    rows = list(csv.DictReader(open(path)))
    keys = rows[0].keys()
    return {k: [r[k] for r in rows] for k in keys}


def main():
    outs = []
    for subj in SUBJECTS:
        try:
            outs.append(process(subj))
        except Exception as e:
            print("%s FAILED: %s" % (subj, e))
    # concat
    allrows = []; hdr = None
    for o in outs:
        L = open(o).read().splitlines()
        if hdr is None: hdr = L[0]
        allrows += L[1:]
    with open(os.path.join(BATCH, "ml_dataset_A.csv"), "w", newline="\n") as f:
        f.write(hdr + "\n" + "\n".join(allrows) + "\n")
    print("\nwrote batch/ml_dataset_A.csv (%d frames total)" % len(allrows))


if __name__ == "__main__":
    main()
