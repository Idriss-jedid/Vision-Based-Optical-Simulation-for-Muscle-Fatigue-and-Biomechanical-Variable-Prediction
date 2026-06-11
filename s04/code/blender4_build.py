# pyright: reportMissingImports=false
"""
4-CAMERA Blender build for s04. Produces, in D:\\p2s_blender\\s04_arm26_4cam\\ :
  model.osim            arm26_paper scaled to s04 (Vicon segment lengths)
  motion_world.csv      arm26 body transforms placed in the 4-cam .trc/world frame,
                        50 fps (synced to video + markers), z-up like markers.py
  arm_markers_world.trc the 3 arm markers in raw 4-cam world coords, 50 fps
Also writes build4/motion/curl_17s.mot + prints the angle accuracy vs Vicon.
Run with the biomech env.  (Calib_world.toml is built separately with cv2.)
"""
import csv
import glob
import os

import numpy as np
from scipy.signal import butter, filtfilt, medfilt
import opensim as osim

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
ROOT = os.path.dirname(S04)
BASE = os.path.join(ROOT, "Model", "arm26_paper_loaded_brd_elbow_research.osim")
GT_CSV = os.path.join(S04, "build2", "csv", "joints3d_25.csv")
TRC_DIR = os.path.join(S04, "build4", "pose2sim", "pose-3d")     # <-- 4-cam
OUTM = os.path.join(S04, "build4", "motion"); os.makedirs(OUTM, exist_ok=True)
OUTO = os.path.join(S04, "build4", "opensim"); os.makedirs(OUTO, exist_ok=True)
SCALED = os.path.join(OUTO, "arm26_s04_scaled.osim")
MOTION = os.path.join(OUTM, "curl_17s.mot")
DST = r"D:\p2s_blender\s04_arm26_4cam"
MK = [("RShoulder", "r_acromion"), ("RElbow", "r_humerus_epicondyle"), ("RWrist", "r_radius_styloid")]
RATE, ELB_MAX, SHOULDER_DEG, FPS = 100.0, 128.0, 20.0, 50.0
H_ZUP = np.array([[1., 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def flex(S, E, W):
    v1, v2 = S - E, W - E
    cs = np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)
    return 180.0 - np.degrees(np.arccos(cs))


def hampel(x, k=10, ns=3.0):
    x = x.astype(float).copy(); n = len(x)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1); w = x[lo:hi]; med = np.nanmedian(w)
        mad = 1.4826 * np.nanmedian(np.abs(w - med)) + 1e-9
        if not np.isnan(x[i]) and abs(x[i] - med) > ns * mad: x[i] = np.nan
    idx = np.arange(n); good = ~np.isnan(x); return np.interp(idx, idx[good], x[good])


def trc_xyz():
    trc = sorted(f for f in glob.glob(os.path.join(TRC_DIR, "*.trc")) if "LSTM" not in f)[-1]
    L = open(trc).read().splitlines()
    names = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = names.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return {r: g(r) for r, _ in MK}


def vicon_J():
    rows = list(csv.DictReader(open(GT_CSV)))
    return np.array([[float(r["J%d_%s" % (j, ax)]) for j in range(25) for ax in "xyz"] for r in rows]).reshape(-1, 25, 3)


def vicon():
    J = vicon_J()
    ua = float(np.median(np.linalg.norm(J[:, 14] - J[:, 15], axis=1)))
    fa = float(np.median(np.linalg.norm(J[:, 15] - J[:, 16], axis=1)))
    return ua, fa, flex(J[:, 14], J[:, 15], J[:, 16])


def elev_of(S, E, down):
    """Élévation du bras (S->E) par rapport à l'axe 'down', en degrés (array-safe)."""
    v = E - S
    vn = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip(vn @ np.asarray(down, float), -1, 1)))


def shoulder_lookup():
    """Table FK : r_shoulder_elev (deg) -> élévation du bras vs -Y (deg), modèle scalé.
    Monotone sur [0,130] -> on l'inverse pour mapper l'élévation mesurée vers la coordonnée."""
    m = osim.Model(SCALED); s = m.initSystem()
    csh = m.getCoordinateSet().get("r_shoulder_elev"); cel = m.getCoordinateSet().get("r_elbow_flex")
    cel.setValue(s, np.radians(20.0), False)
    def mk(n):
        loc = m.getMarkerSet().get(n).getLocationInGround(s); return np.array([loc.get(0), loc.get(1), loc.get(2)])
    sh_grid = np.arange(0.0, 131.0, 1.0); elev = []
    for sg in sh_grid:
        csh.setValue(s, np.radians(sg)); m.realizePosition(s)
        elev.append(float(elev_of(mk("r_acromion"), mk("r_humerus_epicondyle"), [0, -1, 0])))
    return sh_grid, np.array(elev)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, Dg, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s) / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt
    c = np.trace(np.diag(Dg) @ S) / (((src - mu_s) ** 2).sum() / len(src))
    return R, c, mu_d - c * R @ mu_s


def euler_xyz(Rm):
    sy = np.sqrt(Rm[1, 0] ** 2 + Rm[0, 0] ** 2)
    if sy > 1e-6:
        return np.arctan2(Rm[2, 1], Rm[2, 2]), np.arctan2(-Rm[2, 0], sy), np.arctan2(Rm[1, 0], Rm[0, 0])
    return np.arctan2(-Rm[1, 2], Rm[1, 1]), np.arctan2(-Rm[2, 0], sy), 0.0


def scale_and_motion():
    ua_gt, fa_gt, gt = vicon(); J = vicon_J()
    m = osim.Model(BASE); s = m.initSystem()
    def jc(j): return np.array([m.getJointSet().get(j).getChildFrame().getPositionInGround(s).get(i) for i in range(3)])
    def mk(n): return np.array([m.getMarkerSet().get(n).getLocationInGround(s).get(i) for i in range(3)])
    ua_d = np.linalg.norm(jc("r_shoulder") - jc("r_elbow")); fa_d = np.linalg.norm(jc("r_elbow") - mk("r_radius_styloid"))
    sf_h, sf_f = ua_gt / ua_d, fa_gt / fa_d
    m2 = osim.Model(BASE); s2 = m2.initSystem(); sset = osim.ScaleSet()
    for body, sf in [("r_humerus", sf_h), ("r_ulna_radius_hand", sf_f)]:
        sc = osim.Scale(); sc.setSegmentName(body); sc.setScaleFactors(osim.Vec3(sf, sf, sf)); sc.setApply(True); sset.cloneAndAppend(sc)
    m2.scale(s2, sset, False, -1.0); m2.printToXML(SCALED)
    print("scaled arm26 (4-cam): humerus x%.3f, forearm x%.3f -> %s" % (sf_h, sf_f, SCALED))

    raw = trc_xyz()
    n = min(len(raw["RElbow"]), len(gt), len(J))
    # --- ELBOW : despike -> alignement (lag) -> affine vs Vicon ---
    el_raw = hampel(medfilt(flex(raw["RShoulder"], raw["RElbow"], raw["RWrist"]), 7), 10)[:n]
    best = (-2, 0)
    for lag in range(0, 8):
        seg = gt[lag:lag + n]
        if len(seg) < n: continue
        r = np.corrcoef(el_raw, seg)[0, 1]
        if r > best[0]: best = (r, lag)
    lag = best[1]; g_el = gt[lag:lag + n]
    a_e, b_e = np.polyfit(el_raw, g_el, 1); el_cal = a_e * el_raw + b_e
    # --- SHOULDER (data-driven) : élévation markerless(-Y) -> affine vs Vicon élévation(-Z) -> inverse table FK ---
    ms_elev = hampel(medfilt(elev_of(raw["RShoulder"], raw["RElbow"], [0, -1, 0]), 7), 10)[:n]
    vs_elev = elev_of(J[:, 14], J[:, 15], [0, 0, -1])
    g_sh = vs_elev[lag:lag + n]
    a_s, b_s = np.polyfit(ms_elev, g_sh, 1); sh_elev_cal = a_s * ms_elev + b_s
    sh_grid, elev_grid = shoulder_lookup()
    sh_deg = np.interp(sh_elev_cal, elev_grid, sh_grid)        # élévation -> r_shoulder_elev
    # --- resample 100 Hz + 2 Hz low-pass (les deux DOF) ---
    t = np.arange(n) / FPS; tv = np.arange(0, t[-1] + 1e-9, 1.0 / RATE)
    bb, aa = butter(4, 2.0 / (RATE / 2), btype="low")
    fl = np.clip(filtfilt(bb, aa, np.interp(tv, t, el_cal)), 0.0, ELB_MAX)
    shv = np.clip(filtfilt(bb, aa, np.interp(tv, t, sh_deg)), -30.0, 130.0)
    with open(MOTION, "w", newline="\n") as f:
        f.write("curl_17s\nversion=1\nnRows=%d\nnColumns=3\ninDegrees=yes\nendheader\n" % len(tv))
        f.write("time\tr_shoulder_elev\tr_elbow_flex\n")
        for k in range(len(tv)): f.write("%.4f\t%.6f\t%.6f\n" % (tv[k], shv[k], fl[k]))
    gtv = np.interp(tv, t, g_el)
    sh_model_elev = np.interp(shv, sh_grid, elev_grid); g_sh_v = np.interp(tv, t, g_sh)
    print("4-cam ELBOW    : r=%.3f MAE=%.1f° ROM %.0f-%.0f°" % (best[0], np.mean(np.abs(fl - gtv)), fl.min(), fl.max()))
    print("4-cam SHOULDER : r=%.3f (élévation), r_shoulder_elev %.0f-%.0f°, élévation MAE vs Vicon %.1f° (était fixé 20°)"
          % (np.corrcoef(sh_elev_cal, g_sh)[0, 1], shv.min(), shv.max(), np.mean(np.abs(sh_model_elev - g_sh_v))))
    print("%d frames @100Hz -> %s" % (len(tv), MOTION))


def world_outputs():
    os.makedirs(DST, exist_ok=True)
    m = osim.Model(SCALED); s = m.initSystem()
    L = open(MOTION).read().splitlines(); i = next(k for k, l in enumerate(L) if l.strip().lower() == "endheader")
    cols = L[i + 1].split("\t"); d = np.array([[float(x) for x in r.split("\t")] for r in L[i + 2:] if r.strip()])
    tm = d[:, 0]; sh = d[:, cols.index("r_shoulder_elev")]; el = d[:, cols.index("r_elbow_flex")]
    csh = m.getCoordinateSet().get("r_shoulder_elev"); cel = m.getCoordinateSet().get("r_elbow_flex")
    bodies = [m.getBodySet().get(j) for j in range(m.getBodySet().getSize())]; bodyNames = [b.getName() for b in bodies]
    real = trc_xyz(); n_r = len(real["RElbow"]); tr = np.arange(n_r) / FPS
    # model markers at real times for the similarity fit
    mdl = np.zeros((n_r, 3, 3))
    for k, tt in enumerate(tr):
        csh.setValue(s, np.radians(np.interp(tt, tm, sh)), False); cel.setValue(s, np.radians(np.interp(tt, tm, el))); m.realizePosition(s)
        for j, (_, mn) in enumerate(MK):
            loc = m.getMarkerSet().get(mn).getLocationInGround(s); mdl[k, j] = [loc.get(0), loc.get(1), loc.get(2)]
    realA = np.stack([real[r] for r, _ in MK], axis=1); good = ~np.isnan(realA).any(axis=(1, 2))
    R, c, tt0 = umeyama(realA[good].reshape(-1, 3), mdl[good].reshape(-1, 3))
    Minv = np.eye(4); Minv[:3, :3] = R.T / c; Minv[:3, 3] = -(R.T @ tt0) / c
    # body transforms @50 fps (synced to video/markers)
    rows = []
    for tt in tr:
        csh.setValue(s, np.radians(np.interp(tt, tm, sh)), False); cel.setValue(s, np.radians(np.interp(tt, tm, el))); m.realizePosition(s)
        row = [tt]
        for b in bodies:
            Hs = b.getTransformInGround(s); T = Hs.T().to_numpy(); Rs = Hs.R()
            Rm = np.array([[Rs.get(r, cc) for cc in range(3)] for r in range(3)])
            Hg = np.eye(4); Hg[:3, :3] = Rm; Hg[:3, 3] = T
            Hb = H_ZUP @ (Minv @ Hg); loc = Hb[:3, 3]; rx, ry, rz = euler_xyz(Hb[:3, :3])
            row += [loc[0], loc[1], loc[2], rx, ry, rz]
        rows.append(row)
    arr = np.array(rows)
    header = "times, " + "".join(["%s_x, %s_y, %s_z, %s_rotx, %s_roty, %s_rotz, " % (b, b, b, b, b, b) for b in bodyNames])[:-2]
    np.savetxt(os.path.join(DST, "motion_world.csv"), arr, delimiter=",", header=header)
    print("wrote motion_world.csv (%d frames @50fps, bodies: %s)" % (len(arr), ", ".join(bodyNames)))
    # arm markers (raw world coords)
    M = len(MK); names = [r for r, _ in MK]
    with open(os.path.join(DST, "arm_markers_world.trc"), "w", newline="\n") as f:
        f.write("PathFileType\t4\t(X/Y/Z)\tarm_markers_world.trc\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write("%g\t%g\t%d\t%d\tm\t%g\t1\t%d\n" % (FPS, FPS, n_r, M, FPS, n_r))
        f.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\t\n")
        f.write("\t\t" + "\t".join("X%d\tY%d\tZ%d" % (k + 1, k + 1, k + 1) for k in range(M)) + "\t\n\n")
        for k in range(n_r):
            vals = "\t".join("%.6f\t%.6f\t%.6f" % tuple(realA[k, j]) for j in range(M)); f.write("%d\t%.5f\t%s\n" % (k + 1, tr[k], vals))
    print("wrote arm_markers_world.trc (%d frames @50fps)" % n_r)
    import shutil as sh_
    sh_.copy(SCALED, os.path.join(DST, "model.osim"))
    print("copied model.osim")


if __name__ == "__main__":
    scale_and_motion()
    world_outputs()
