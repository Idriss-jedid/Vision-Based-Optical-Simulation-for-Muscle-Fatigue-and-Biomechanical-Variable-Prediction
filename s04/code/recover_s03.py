# pyright: reportMissingImports=false
"""
Récupère s03 : le SO se bloque à t=1.32s. CAUSE = le wrap WrapEllipsoid 'BIClonghh' (BIClong
sur la tête humérale) devient dégénéré à cette config précise -> moment arm NaN -> SO boucle.
DÉCOUVERTE : sur tout le ROM du curl (0-130°) le path de BIClong NE wrap PAS sur cet ellipsoïde
(moment arm IDENTIQUE avec/sans wrap, diff = 0.0000 m). Donc retirer 'BIClonghh' = effet
biomécanique NUL ; ça supprime juste le solver de wrap dégénéré -> plus de blocage.
-> on crée un modèle sans ce wrap, on relance ID+SO+3CC -> batch/s03/labels_ml.csv. biomech env.
"""
import csv, os, sys
import xml.etree.ElementTree as ET
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "Code"))
import run_stage2_pipeline as P  # noqa: E402

BATCH = os.path.join(ROOT, "batch"); SUBJ = "s03"
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
WRAP = "BIClonghh"


def strip_wrap(src, dst):
    tree = ET.parse(src); root = tree.getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "WrapEllipsoid" and child.get("name") == WRAP:
                parent.remove(child)
            elif child.tag == "PathWrap":
                wo = child.find("wrap_object")
                if wo is not None and wo.text and wo.text.strip() == WRAP:
                    parent.remove(child)
    tree.write(dst)
    return dst


def read_mot(path):
    L = open(path).read().splitlines()
    i = next(k for k, l in enumerate(L) if l.strip().lower() == "endheader")
    cols = L[i + 1].split("\t")
    d = np.array([[float(x) for x in r.split("\t")] for r in L[i + 2:] if r.strip()])
    return cols, d


def deriv(x, dt):
    v = np.gradient(x, dt); return v, np.gradient(v, dt)


def _rd(path):
    rows = list(csv.DictReader(open(path)))
    return {k: [r[k] for r in rows] for k in rows[0].keys()}


def main():
    motion = os.path.join(BATCH, SUBJ, "motion", "curl.mot")
    scaled = os.path.join(BATCH, SUBJ, "opensim", "arm26_%s_scaled.osim" % SUBJ)
    model = strip_wrap(scaled, os.path.join(BATCH, SUBJ, "opensim", "arm26_%s_scaled_nowrap.osim" % SUBJ))
    print("%s: modèle sans wrap '%s' créé (effet moment-arm = 0)" % (SUBJ, WRAP))
    outdir = os.path.join(BATCH, SUBJ, "labels"); os.makedirs(outdir, exist_ok=True)
    cols, d = read_mot(motion)
    t = d[:, 0]; qsh = d[:, cols.index("r_shoulder_elev")]; qel = d[:, cols.index("r_elbow_flex")]
    dt = float(np.median(np.diff(t)))
    thr = (qel.max() + qel.min()) / 2; above = qel > thr; rep = np.ones(len(t), int); r = 1
    for i in range(1, len(t)):
        if above[i] and not above[i - 1]: r += 1
        rep[i] = r
    ds = os.path.join(outdir, "ds.csv")
    with open(ds, "w", newline="\n") as f:
        f.write("time,rep_index,fatigue_level\n")
        for i in range(len(t)): f.write("%.4f,%d,0\n" % (t[i], rep[i]))

    t0, t1 = P.motion_range(motion)
    mp = P.prep_model(model, outdir)
    print("  ID..."); ids = P.run_id(mp, motion, outdir, t0, t1)
    print("  SO... (point critique t=1.32)"); act, frc = P.run_so(mp, motion, outdir, SUBJ, t0, t1)
    print("  SO OK"); labels = P.merge_labels(ids, act, frc, ds, outdir)
    fatcsv, fails = P.run_3cc(labels, model, motion, outdir)
    lab = _rd(labels); fcd = _rd(fatcsv)
    qdsh, qddsh = deriv(qsh, dt); qdel, qddel = deriv(qel, dt)
    n = min(len(lab["time"]), len(t))
    out = os.path.join(BATCH, SUBJ, "labels_ml.csv")
    hdr = (["subj", "time", "rep", "q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el", "elbow_moment"]
           + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX])
    with open(out, "w", newline="\n") as f:
        f.write(",".join(hdr) + "\n")
        for i in range(n):
            row = [SUBJ, t[i], rep[i], qsh[i], qel[i], qdsh[i], qdel[i], qddsh[i], qddel[i],
                   float(lab["elbow_moment_Nm"][i])]
            row += [float(lab["act_" + m][i]) for m in FLEX]
            row += [float(lab["frc_" + m][i]) for m in FLEX]
            row += [float(fcd["MF_" + m][i]) if ("MF_" + m) in fcd else 0.0 for m in FLEX]
            f.write(",".join(str(x) if not isinstance(x, float) else "%.5f" % x for x in row) + "\n")
    print("%s: labels_ml.csv ECRIT (%d frames, fails=%d)" % (SUBJ, n, fails))


if __name__ == "__main__":
    main()
