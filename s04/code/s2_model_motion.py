# pyright: reportMissingImports=false
"""
s04 — STAGE 2: scale arm26_paper_loaded_brd_elbow_research.osim to subject s04 and
build the elbow-curl motion from the 2-camera Pose2Sim output. Run with biomech env.

Base model  : Model/arm26_paper_loaded_brd_elbow_research.osim  (2 DOF: r_shoulder_elev,
              r_elbow_flex; 7 muscles; 2 kg dumbbell welded). THIS is the model used
              everywhere downstream (biomechanics + Blender).
Scaling     : OpenSim ScaleSet on r_humerus + r_ulna_radius_hand, factors = Vicon
              segment length / model default length.
Motion      : raw triangulated RShoulder/RElbow/RWrist elbow angle -> median(7)+Hampel
              de-spike -> affine de-bias vs Vicon -> 2 Hz low-pass -> clip to model
              range (0..128 deg). Shoulder held at 20 deg (curl is ~planar at the elbow).
Output      : s04/build2/opensim/arm26_s04_scaled.osim , s04/build2/motion/curl_17s.mot
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt
import opensim as osim

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
ROOT = os.path.dirname(S04)
BASE = os.path.join(ROOT, "Model", "arm26_paper_loaded_brd_elbow_research.osim")
GT_CSV = os.path.join(S04, "build2", "csv", "joints3d_25.csv")
TRC_DIR = os.path.join(S04, "build2", "pose2sim", "pose-3d")
OUT = os.path.join(S04, "build2", "opensim")
MDIR = os.path.join(S04, "build2", "motion")
SCALED = os.path.join(OUT, "arm26_s04_scaled.osim")
MOTION = os.path.join(MDIR, "curl_17s.mot")
RATE, ELB_MAX, SHOULDER_DEG, FPS = 100.0, 128.0, 20.0, 50.0


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


def vicon():
    rows = list(csv.DictReader(open(GT_CSV)))
    J = np.array([[float(r["J%d_%s" % (j, ax)]) for j in range(25) for ax in "xyz"] for r in rows]).reshape(-1, 25, 3)
    ua = float(np.median(np.linalg.norm(J[:, 14] - J[:, 15], axis=1)))
    fa = float(np.median(np.linalg.norm(J[:, 15] - J[:, 16], axis=1)))
    return ua, fa, flex(J[:, 14], J[:, 15], J[:, 16])


def default_lengths():
    m = osim.Model(BASE); s = m.initSystem()
    def jc(j): return np.array([m.getJointSet().get(j).getChildFrame().getPositionInGround(s).get(i) for i in range(3)])
    def mk(n): return np.array([m.getMarkerSet().get(n).getLocationInGround(s).get(i) for i in range(3)])
    return float(np.linalg.norm(jc("r_shoulder") - jc("r_elbow"))), float(np.linalg.norm(jc("r_elbow") - mk("r_radius_styloid")))


def scale_model(sf_h, sf_f):
    m = osim.Model(BASE); s = m.initSystem()
    sset = osim.ScaleSet()
    for body, sf in [("r_humerus", sf_h), ("r_ulna_radius_hand", sf_f)]:
        sc = osim.Scale(); sc.setSegmentName(body); sc.setScaleFactors(osim.Vec3(sf, sf, sf)); sc.setApply(True)
        sset.cloneAndAppend(sc)
    m.scale(s, sset, False, -1.0)
    os.makedirs(OUT, exist_ok=True); m.printToXML(SCALED)


def trc_elbow():
    trc = sorted(f for f in glob.glob(os.path.join(TRC_DIR, "*.trc")) if "LSTM" not in f)[-1]
    L = open(trc).read().splitlines()
    mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return flex(g("RShoulder"), g("RElbow"), g("RWrist"))


def main():
    print("s04 STAGE 2 — scale arm26_paper + build motion")
    ua_gt, fa_gt, gt = vicon()
    ua_d, fa_d = default_lengths()
    sf_h, sf_f = ua_gt / ua_d, fa_gt / fa_d
    scale_model(sf_h, sf_f)
    print("  scaled: humerus x%.3f (Vicon %.3f / def %.3f), forearm x%.3f (Vicon %.3f / def %.3f)"
          % (sf_h, ua_gt, ua_d, sf_f, fa_gt, fa_d))
    print("  -> %s" % os.path.relpath(SCALED, S04))

    raw = hampel(medfilt(trc_elbow(), 7), 10)
    n = min(len(raw), len(gt)); raw = raw[:n]
    best = (-2, 0)
    for lag in range(0, 8):
        seg = gt[lag:lag + n]
        if len(seg) < n: continue
        r = np.corrcoef(raw, seg)[0, 1]
        if r > best[0]: best = (r, lag)
    lag = best[1]; g = gt[lag:lag + n]
    a, b = np.polyfit(raw, g, 1); cal = a * raw + b
    t = np.arange(n) / FPS; tv = np.arange(0, t[-1] + 1e-9, 1.0 / RATE)
    fl = np.interp(tv, t, cal)
    bb, aa = butter(4, 2.0 / (RATE / 2), btype="low"); fl = np.clip(filtfilt(bb, aa, fl), 0.0, ELB_MAX)
    os.makedirs(MDIR, exist_ok=True)
    with open(MOTION, "w", newline="\n") as f:
        f.write("curl_17s\nversion=1\nnRows=%d\nnColumns=3\ninDegrees=yes\nendheader\n" % len(tv))
        f.write("time\tr_shoulder_elev\tr_elbow_flex\n")
        for k in range(len(tv)): f.write("%.4f\t%.6f\t%.6f\n" % (tv[k], SHOULDER_DEG, fl[k]))
    gtv = np.interp(tv, t, g); mae = float(np.mean(np.abs(fl - gtv)))
    print("  affine de-bias a=%.2f b=%.0f | r=%.3f MAE=%.1f deg | ROM %.0f-%.0f deg, %d frames"
          % (a, b, best[0], mae, fl.min(), fl.max(), len(tv)))
    print("  -> %s" % os.path.relpath(MOTION, S04))

    plt.figure(figsize=(13, 4.5))
    plt.plot(tv, gtv, color="#2ca02c", lw=2, label="Vicon GT")
    plt.plot(tv, fl, color="#1f77b4", lw=1.5, label="arm26 motion (2-cam, r=%.3f MAE %.0f°)" % (best[0], mae))
    plt.xlabel("time (s)"); plt.ylabel("r_elbow_flex (deg)")
    plt.title("s04 — arm26_paper curl motion (2-cam 60457274+65906101, full 17 s)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(S04, "build2", "motion_s04.png"), dpi=130); plt.close()
    print("  plot -> build2/motion_s04.png")
    print("DONE s04 stage 2")


if __name__ == "__main__":
    main()
