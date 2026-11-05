# pyright: reportMissingImports=false
"""
Place the arm26 model INTO the triangulation/world frame so that, in Blender, it
overlays the real person in the video at the same scale, with the markers on it.

Pose2Sim_Blender's motion.py has a .csv branch that sets each body's location/euler
directly (no extra axis conversion). So we:
  1) fit ONE similarity real(world)->model (Umeyama) from the 3 arm markers;
  2) for every motion frame, FK arm26 -> each body's transform in the model ground;
  3) map it to the world frame (inverse similarity), then apply the SAME y-up->z-up
     H_zup that markers.py/motion.py use, and dump loc+euler to motion_world.csv;
  4) also write arm_markers_world.trc = the 3 real markers in raw world coords.
In Blender: Import Model (model.osim) -> Import Motion (motion_world.csv) ->
Import Markers (arm_markers_world.trc) -> Import cameras (Calib.toml) -> Show videos.
Run with biomech env.
"""
import glob
import os

import numpy as np
import opensim as osim

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
SCALED = os.path.join(S04, "build2", "opensim", "arm26_s04_scaled.osim")
MOTION = os.path.join(S04, "build2", "motion", "curl_17s.mot")
TRC = sorted(f for f in glob.glob(os.path.join(S04, "build2", "pose2sim", "pose-3d", "*.trc")) if "LSTM" not in f)[-1]
OUT_CSV = r"D:\p2s_blender\s04_arm26\motion_world.csv"
OUT_TRC = r"D:\p2s_blender\s04_arm26\arm_markers_world.trc"
MK = [("RShoulder", "r_acromion"), ("RElbow", "r_humerus_epicondyle"), ("RWrist", "r_radius_styloid")]
FPS = 50.0
H_ZUP = np.array([[1., 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def read_mot(path):
    L = open(path).read().splitlines()
    i = next(k for k, l in enumerate(L) if l.strip().lower() == "endheader")
    cols = L[i + 1].split("\t")
    d = np.array([[float(x) for x in r.split("\t")] for r in L[i + 2:] if r.strip()])
    return cols, d


def trc_xyz():
    L = open(TRC).read().splitlines()
    names = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = names.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return {r: g(r) for r, _ in MK}


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    U, Dg, Vt = np.linalg.svd(Dc.T @ Sc / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    c = np.trace(np.diag(Dg) @ S) / ((Sc ** 2).sum() / len(src))
    t = mu_d - c * R @ mu_s
    return R, c, t


def euler_xyz(Rm):
    sy = np.sqrt(Rm[1, 0] ** 2 + Rm[0, 0] ** 2)
    if sy > 1e-6:
        return np.arctan2(Rm[2, 1], Rm[2, 2]), np.arctan2(-Rm[2, 0], sy), np.arctan2(Rm[1, 0], Rm[0, 0])
    return np.arctan2(-Rm[1, 2], Rm[1, 1]), np.arctan2(-Rm[2, 0], sy), 0.0


def main():
    m = osim.Model(SCALED); s = m.initSystem()
    cols, d = read_mot(MOTION)
    tm = d[:, 0]; sh = d[:, cols.index("r_shoulder_elev")]; el = d[:, cols.index("r_elbow_flex")]
    csh = m.getCoordinateSet().get("r_shoulder_elev"); cel = m.getCoordinateSet().get("r_elbow_flex")
    bodies = [m.getBodySet().get(i) for i in range(m.getBodySet().getSize())]
    bodyNames = [b.getName() for b in bodies]

    # --- similarity real(world) -> model, from the 3 arm markers ---
    real = trc_xyz(); n_r = len(real["RElbow"]); tr = np.arange(n_r) / FPS
    # model marker positions at the real-frame times (FK)
    mdl = np.zeros((n_r, 3, 3))
    for i, t in enumerate(tr):
        csh.setValue(s, np.radians(np.interp(t, tm, sh)), False)
        cel.setValue(s, np.radians(np.interp(t, tm, el)))
        m.realizePosition(s)
        for j, (_, mn) in enumerate(MK):
            loc = m.getMarkerSet().get(mn).getLocationInGround(s)
            mdl[i, j] = [loc.get(0), loc.get(1), loc.get(2)]
    realA = np.stack([real[r] for r, _ in MK], axis=1)
    good = ~np.isnan(realA).any(axis=(1, 2))
    R, c, t = umeyama(realA[good].reshape(-1, 3), mdl[good].reshape(-1, 3))   # model = c R world + t
    Minv = np.eye(4); Minv[:3, :3] = R.T / c; Minv[:3, 3] = -(R.T @ t) / c    # world = Minv @ model

    # --- body transforms at 50 fps (SAME rate/grid as the video + markers, so they
    #     stay frame-synced in Blender; the .mot was 100 Hz which desynced the video) ---
    tv50 = np.arange(n_r) / FPS
    rows = []
    for tt in tv50:
        csh.setValue(s, np.radians(np.interp(tt, tm, sh)), False)
        cel.setValue(s, np.radians(np.interp(tt, tm, el)))
        m.realizePosition(s)
        row = [tt]
        for b in bodies:
            Hs = b.getTransformInGround(s)
            T = Hs.T().to_numpy(); Rs = Hs.R()
            Rm = np.array([[Rs.get(r, cc) for cc in range(3)] for r in range(3)])
            Hg = np.eye(4); Hg[:3, :3] = Rm; Hg[:3, 3] = T
            Hb = H_ZUP @ (Minv @ Hg)
            loc = Hb[:3, 3]; rx, ry, rz = euler_xyz(Hb[:3, :3])
            row += [loc[0], loc[1], loc[2], rx, ry, rz]
        rows.append(row)
    arr = np.array(rows)
    header = "times, " + "".join(["%s_x, %s_y, %s_z, %s_rotx, %s_roty, %s_rotz, " % (b, b, b, b, b, b) for b in bodyNames])[:-2]
    np.savetxt(OUT_CSV, arr, delimiter=",", header=header)
    print("wrote %s  (%d frames, %d bodies: %s)" % (OUT_CSV, len(arr), len(bodyNames), ", ".join(bodyNames)))

    # --- arm markers in raw world coords (markers.py applies z-up itself) ---
    M = len(MK); names = [r for r, _ in MK]
    with open(OUT_TRC, "w", newline="\n") as f:
        f.write("PathFileType\t4\t(X/Y/Z)\t%s\n" % os.path.basename(OUT_TRC))
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write("%g\t%g\t%d\t%d\tm\t%g\t1\t%d\n" % (FPS, FPS, n_r, M, FPS, n_r))
        f.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\t\n")
        f.write("\t\t" + "\t".join("X%d\tY%d\tZ%d" % (k + 1, k + 1, k + 1) for k in range(M)) + "\t\n\n")
        for i in range(n_r):
            vals = "\t".join("%.6f\t%.6f\t%.6f" % tuple(realA[i, j]) for j in range(M))
            f.write("%d\t%.5f\t%s\n" % (i + 1, tr[i], vals))
    print("wrote %s  (%d frames, %d markers, raw world coords)" % (OUT_TRC, n_r, M))


if __name__ == "__main__":
    main()
