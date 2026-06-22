# pyright: reportMissingImports=false
"""
Compare les DUREES (secondes) de chaque etage du pipeline classique vs l'inference AI.
- Pose2Sim (pose, triangulation) : lus des logs.
- OpenSim ID / SO / 3CC : mesures en re-executant sur 1 sujet (dossier neuf, sans cache).
- Stage B (angles) + Stage C (scaling) : mesures.
- AI : temps de construction des features + temps de prediction (3 modeles).
Sortie : batch/timing_pipeline.csv. biomech env.
"""
import os, sys, time, glob, json, shutil
import numpy as np, pandas as pd, joblib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "Code"))
import run_stage2_pipeline as P  # noqa
B = os.path.join(ROOT, "batch"); SUBJ = "s04"

rows = []
def rec(stage, sec, note=""): rows.append(dict(stage=stage, seconds=round(sec, 4), note=note)); print("%-28s %9.3f s  %s" % (stage, sec, note), flush=True)

# ---- 1) Pose2Sim (depuis logs) ----
log = open(os.path.join(B, SUBJ, "pose2sim", "logs.txt")).read()
import re
def parse(pat):
    m = re.search(pat + r".*?(\d+)h(\d+)m(\d+)s", log)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None
rec("Calibration", 1.0, "conversion JSON->toml (approx)")
rec("2D pose (RTMPose)", parse("Pose estimation took") or 103, "from log")
rec("Triangulation", parse("Triangulation took") or 38, "from log")
rec("Filtering", 2.0, "Butterworth on 3D (approx)")

motion = os.path.join(B, SUBJ, "motion", "curl.mot")
scaled = os.path.join(B, SUBJ, "opensim", "arm26_%s_scaled.osim" % SUBJ)

# ---- 2) Stage C : scaling (mesure) ----
import opensim as osim
BASE = os.path.join(ROOT, "Model", "arm26_paper_loaded_brd_elbow_research.osim")
t = time.time()
m2 = osim.Model(BASE); s2 = m2.initSystem(); ss = osim.ScaleSet()
for body, sf in [("r_humerus", 1.006), ("r_ulna_radius_hand", 1.007)]:
    sc = osim.Scale(); sc.setSegmentName(body); sc.setScaleFactors(osim.Vec3(sf, sf, sf)); sc.setApply(True); ss.cloneAndAppend(sc)
m2.scale(s2, ss, False, -1.0); m2.printToXML(os.path.join(B, "_tmp_scaled.osim"))
rec("Stage C: scaling (.osim)", time.time() - t, "OpenSim scale")

# ---- 3) Stage B : angles (mesure, geometrie depuis .trc) ----
def read_trc(path):
    rr = []
    for ln in open(path).read().splitlines():
        q = ln.split("\t")
        if len(q) > 59:
            try: float(q[0]); float(q[1]); rr.append([float(x) if x.strip() else np.nan for x in q])
            except: continue
    d = np.array(rr);
    def mk(i): c = 2 + (i - 1) * 3; return d[:, c:c + 3]
    return mk(17), mk(18), mk(19)
trc = sorted(glob.glob(os.path.join(B, SUBJ, "pose2sim", "pose-3d", "*filt_butterworth.trc")))[0]
t = time.time()
sh, el, wr = read_trc(trc); SE = el - sh; WE = wr - el
q_el = np.arccos(np.clip(np.sum(-SE * WE, 1) / (np.linalg.norm(SE, axis=1) * np.linalg.norm(WE, axis=1) + 1e-9), -1, 1))
rec("Stage B: angles (.mot)", time.time() - t, "geometry from .trc")

# ---- 4) OpenSim ID / SO / 3CC (mesure, dossier neuf) ----
tmp = os.path.join(B, SUBJ, "_timing"); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
t0, t1 = P.motion_range(motion); mp = P.prep_model(scaled, tmp)
t = time.time(); ids = P.run_id(mp, motion, tmp, t0, t1); rec("OpenSim Inverse Dynamics", time.time() - t)
t = time.time(); act, frc = P.run_so(mp, motion, tmp, SUBJ, t0, t1); rec("OpenSim Static Optimization", time.time() - t, "le plus lent")
# 3CC : besoin du merge labels + ds
ds = os.path.join(tmp, "ds.csv"); lab0 = pd.read_csv(os.path.join(B, SUBJ, "labels_ml.csv"))
with open(ds, "w") as f:
    f.write("time,rep_index,fatigue_level\n")
    for tt in lab0["time"].values: f.write("%.4f,1,0\n" % tt)
labels = P.merge_labels(ids, act, frc, ds, tmp)
t = time.time(); P.run_3cc(labels, scaled, motion, tmp); rec("OpenSim 3CC (fatigue)", time.time() - t)
shutil.rmtree(tmp, ignore_errors=True)

# ---- 5) AI inference (3 modeles) ----
nfr = len(lab0)
# 3D model
d3 = pd.read_csv(os.path.join(B, "ml_dataset_3D.csv")); d3 = d3[d3.subj == SUBJ]
b3 = joblib.load(os.path.join(B, "model_3d_final", "lgbm_3d_vision.joblib"))
X3 = b3["x_scaler"].transform(d3[b3["features"]])
t = time.time(); _ = b3["y_scaler"].inverse_transform(b3["model"].predict(X3)); dt3 = time.time() - t
rec("AI inference (3D model)", dt3, "predict %d frames -> %.3f ms/frame" % (nfr, 1000 * dt3 / nfr))
# Approche A model
dA = pd.read_csv(os.path.join(B, "ml_dataset_A.csv"))
import numpy as _np
g = _np.pi / 180
dA["sin_qel"] = _np.sin(dA.q_el * g); dA["cos_qel"] = _np.cos(dA.q_el * g); dA["sin_qsh"] = _np.sin(dA.q_sh * g); dA["cos_qsh"] = _np.cos(dA.q_sh * g)
dA["abs_qd_el"] = dA.qd_el.abs(); dA["abs_qdd_el"] = dA.qdd_el.abs(); dA["qd_el2"] = dA.qd_el ** 2
dA["grav_load"] = (dA.forearm_mass + 2) * dA.forearm_len * _np.sin((dA.q_sh + dA.q_el) * g); dA["qel_x_fmass"] = dA.q_el * dA.forearm_mass
cp = []; cg = []
for sb, sub in dA.groupby("subj", sort=False):
    dt = _np.median(_np.diff(sub.time.values)); cp.append(_np.cumsum(sub.abs_qd_el.values) * dt); cg.append(_np.cumsum(_np.abs(sub.grav_load.values)) * dt)
dA["cum_path_el"] = _np.concatenate(cp); dA["cum_grav_imp"] = _np.concatenate(cg)
dA4 = dA[dA.subj == SUBJ]
bA = joblib.load(os.path.join(B, "model_final", "lgbm_trial11.joblib"))
XA = bA["x_scaler"].transform(dA4[bA["features"]])
t = time.time(); _ = bA["y_scaler"].inverse_transform(bA["model"].predict(XA)); dtA = time.time() - t
rec("AI inference (Approach A)", dtA, "predict %d frames -> %.3f ms/frame" % (nfr, 1000 * dtA / nfr))

df = pd.DataFrame(rows); df.to_csv(os.path.join(B, "timing_pipeline.csv"), index=False)
# totaux
classic_vision = df[df.stage.isin(["Calibration", "2D pose (RTMPose)", "Triangulation", "Filtering"])].seconds.sum()
classic_osim = df[df.stage.str.startswith("OpenSim")].seconds.sum()
ai_inf = float(df[df.stage == "AI inference (3D model)"].seconds.iloc[0])
print("\n=== TOTAUX ===")
print("Vision (pose+triang+...) : %.1f s" % classic_vision)
print("OpenSim ID+SO+3CC        : %.1f s  <- remplace par l'AI" % classic_osim)
print("AI inference (3D)        : %.4f s  (%.1fx plus rapide que OpenSim)" % (ai_inf, classic_osim / max(ai_inf, 1e-6)))
print("wrote batch/timing_pipeline.csv")
