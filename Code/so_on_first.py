# pyright: reportMissingImports=false
"""Run ID + SO on the FIRST motion (paper_minjerk_fatigue_10cycle_first.mot) with the
7-muscle arm26 model, and report per-muscle activation/force stats so we can build the
11->7 comparison table. biomech env."""
import os
import numpy as np
import run_stage2_pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "Model", "arm26_paper_loaded_brd_elbow_research.osim")
MOTION = os.path.join(HERE, "..", "Data", "paper_minjerk_fatigue_10cycle_first.mot")
OUT = os.path.join(HERE, "..", "Results", "Stage2", "first_10rep")
MUS = ["BIClong", "BICshort", "BRA", "BRD_hand", "TRIlong", "TRIlat", "TRImed"]

os.makedirs(OUT, exist_ok=True)
t0, t1 = P.motion_range(MOTION)
mp = P.prep_model(MODEL, OUT)
ids = P.run_id(mp, MOTION, OUT, t0, t1)
act, frc = P.run_so(mp, MOTION, OUT, "first", t0, t1)
ac, ar = P.read_sto(act); fc, fr = P.read_sto(frc)
idc, idr = P.read_sto(ids)
elcol = [c for c in idc if "elbow" in c and ("moment" in c or "flex" in c)][0]
M = np.abs(P.col(idc, idr, elcol))
print("Motion: FIRST (t=%.2f..%.2f s, %d frames)" % (t0, t1, len(ar)))
print("elbow moment |M|: mean %.2f, peak %.2f N.m\n" % (np.mean(M), np.max(M)))
print("%-10s %-9s %10s %10s %10s %10s" % ("muscle", "role", "act_mean%", "act_peak%", "frc_mean", "frc_peak"))
print("-" * 64)
for m in MUS:
    role = "FLEXOR" if m.startswith(("BIC", "BRA", "BRD")) else "extensor"
    a = np.array(P.col(ac, ar, m)) if m in ac else np.zeros(len(ar))
    f = np.array(P.col(fc, fr, m)) if m in fc else np.zeros(len(fr))
    print("%-10s %-9s %9.1f %9.1f %9.1f %9.1f" % (m, role, 100*np.mean(a), 100*np.max(a), np.mean(f), np.max(f)))
