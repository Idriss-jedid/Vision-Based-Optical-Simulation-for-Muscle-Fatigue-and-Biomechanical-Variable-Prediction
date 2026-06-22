# pyright: reportMissingImports=false
"""
Build an arm-only .trc (RShoulder/RElbow/RWrist) expressed in the arm26 model frame,
so the REAL markerless markers overlay on the moving arm26 in Blender.

The full-body Pose2Sim markers.trc lives in the lab/triangulation frame and is the
wrong marker set for the arm26 model, so it floats away. Here we:
  1) drive arm26 with curl_17s.mot and read its 3 arm markers (acromion/epicondyle/
     styloid) in ground at each real-frame time (forward kinematics);
  2) read the real RShoulder/RElbow/RWrist from the filtered .trc;
  3) fit ONE similarity transform (scale+R+t, Umeyama) real->model over all frames;
  4) apply it to the real markers and write arm_markers.trc (model frame, 50 Hz).
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
OUT = r"D:\p2s_blender\s04_arm26\arm_markers.trc"
MK = [("RShoulder", "r_acromion"), ("RElbow", "r_humerus_epicondyle"), ("RWrist", "r_radius_styloid")]
FPS = 50.0


def read_mot(path):
    L = open(path).read().splitlines()
    i = next(k for k, l in enumerate(L) if l.strip().lower() == "endheader")
    cols = L[i + 1].split("\t")
    d = np.array([[float(x) for x in r.split("\t")] for r in L[i + 2:] if r.strip()])
    return cols, d


def real_markers():
    L = open(TRC).read().splitlines()
    names = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = names.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return np.stack([g(r) for r, _ in MK], axis=1)        # (T,3,3)


def model_markers_at(times):
    m = osim.Model(SCALED); s = m.initSystem()
    cols, d = read_mot(MOTION)
    tm = d[:, 0]; sh = d[:, cols.index("r_shoulder_elev")]; el = d[:, cols.index("r_elbow_flex")]
    csh = m.getCoordinateSet().get("r_shoulder_elev"); cel = m.getCoordinateSet().get("r_elbow_flex")
    out = np.zeros((len(times), 3, 3))
    for i, t in enumerate(times):
        csh.setValue(s, np.radians(np.interp(t, tm, sh)), False)
        cel.setValue(s, np.radians(np.interp(t, tm, el)))
        m.realizePosition(s)
        for j, (_, mn) in enumerate(MK):
            loc = m.getMarkerSet().get(mn).getLocationInGround(s)
            out[i, j] = [loc.get(0), loc.get(1), loc.get(2)]
    return out


def umeyama(src, dst):
    """similarity transform mapping src->dst: dst ~= c*R@src + t. src,dst (N,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = Dc.T @ Sc / len(src)
    U, Dg, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    c = np.trace(np.diag(Dg) @ S) / ((Sc ** 2).sum() / len(src))
    t = mu_d - c * R @ mu_s
    return R, c, t


def write_trc(path, P, times):
    T, M, _ = P.shape
    names = [r for r, _ in MK]
    with open(path, "w", newline="\n") as f:
        f.write("PathFileType\t4\t(X/Y/Z)\t%s\n" % os.path.basename(path))
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write("%g\t%g\t%d\t%d\tm\t%g\t1\t%d\n" % (FPS, FPS, T, M, FPS, T))
        f.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\t\n")
        f.write("\t\t" + "\t".join("X%d\tY%d\tZ%d" % (k + 1, k + 1, k + 1) for k in range(M)) + "\t\n")
        f.write("\n")
        for i in range(T):
            row = "\t".join("%.6f\t%.6f\t%.6f" % tuple(P[i, j]) for j in range(M))
            f.write("%d\t%.5f\t%s\n" % (i + 1, times[i], row))


def main():
    real = real_markers()
    n = len(real); times = np.arange(n) / FPS
    mdl = model_markers_at(times)
    # pair valid frames, fit ONE global similarity real->model
    good = ~np.isnan(real).any(axis=(1, 2))
    src = real[good].reshape(-1, 3); dst = mdl[good].reshape(-1, 3)
    R, c, t = umeyama(src, dst)
    aligned = (c * (real.reshape(-1, 3) @ R.T) + t).reshape(n, 3, 3)
    res = np.linalg.norm(aligned[good] - mdl[good], axis=2)
    write_trc(OUT, aligned, times)
    print("similarity real->arm26: scale=%.3f" % c)
    print("overlay residual (real vs model markers): mean=%.1f mm, max=%.1f mm" % (1000 * res.mean(), 1000 * res.max()))
    print("wrote %s  (%d frames, 3 markers, model frame, m)" % (OUT, n))


if __name__ == "__main__":
    main()
