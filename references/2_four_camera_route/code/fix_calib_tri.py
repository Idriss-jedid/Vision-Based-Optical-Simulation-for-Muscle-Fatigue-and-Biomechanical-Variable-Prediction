# -*- coding: utf-8 -*-
"""
FIX: Pose2Sim computeP() builds P = K[R | translation], i.e. translation must be the
OpenCV tvec (t in X_cam = R*X + t). Fit3D's extrinsic T is the camera CENTRE, so the
correct value is tvec = -R*T. Rewrite Calib.toml accordingly, then re-triangulate
(reusing the cached pose) and check the elbow angle vs Vicon. pose2sim_env.
"""
import csv
import glob
import json
import os
import re

import numpy as np
import cv2
import toml

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
EXERCISE = "dumbbell_biceps_curls"
PROJ = os.path.join(S04, "build2", "pose2sim")
CALIB = os.path.join(PROJ, "calibration", "Calib.toml")
GT_CSV = os.path.join(S04, "build2", "csv", "joints3d_25.csv")
CAMS = ["60457274", "65906101"]
SIZE = (900, 900)


def num(x): return repr(float(x))
def arr(a): return "[ " + ", ".join(num(v) for v in a) + ",]"
def mat(M): return "[ " + ", ".join("[ %s,]" % ", ".join(num(v) for v in r) for r in M) + ",]"


def rewrite_calib():
    txt = ""
    for cam in CAMS:
        d = json.load(open(os.path.join(S04, "camera_parameters", cam, EXERCISE + ".json")))
        R = np.array(d["extrinsics"]["R"], float).reshape(3, 3)
        T = np.array(d["extrinsics"]["T"], float).reshape(3)
        tvec = (-R @ T)                                          # <-- the fix
        f = np.array(d["intrinsics_w_distortion"]["f"], float).reshape(-1)
        c = np.array(d["intrinsics_w_distortion"]["c"], float).reshape(-1)
        k = np.array(d["intrinsics_w_distortion"]["k"], float).reshape(-1)
        p = np.array(d["intrinsics_w_distortion"]["p"], float).reshape(-1)
        K = [[f[0], 0.0, c[0]], [0.0, f[1], c[1]], [0.0, 0.0, 1.0]]
        rvec = cv2.Rodrigues(R)[0].reshape(-1)
        txt += ("[cam_%s]\nname = \"cam_%s\"\nsize = %s\nmatrix = %s\n"
                "distortions = %s\nrotation = %s\ntranslation = %s\nfisheye = false\n\n"
                % (cam, cam, arr([SIZE[0], SIZE[1]]), mat(K),
                   arr([k[0], k[1], p[0], p[1]]), arr(rvec), arr(tvec)))
    txt += "[metadata]\nadjusted = false\nerror = 0.0\n"
    open(CALIB, "w", newline="\n").write(txt)
    print("Calib.toml rewritten with translation = -R*T (OpenCV tvec convention)")


def run():
    cfg = toml.load(os.path.join(PROJ, "Config.toml"))
    cfg["pose"]["overwrite_pose"] = False
    cfg.setdefault("triangulation", {}).update(dict(
        min_cameras_for_triangulation=2, reproj_error_threshold_triangulation=50,
        likelihood_threshold_triangulation=0.2, interp_if_gap_smaller_than=30))
    toml.dump(cfg, open(os.path.join(PROJ, "Config.toml"), "w"))
    os.chdir(PROJ)
    from Pose2Sim import Pose2Sim
    print(">>> triangulation");      Pose2Sim.triangulation()
    print(">>> filtering");          Pose2Sim.filtering()
    print(">>> markerAugmentation"); Pose2Sim.markerAugmentation()
    print(">>> kinematics");         Pose2Sim.kinematics()


def flex(S, E, W):
    v1, v2 = S - E, W - E
    cs = np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)
    return 180.0 - np.degrees(np.arccos(cs))


def check():
    cands = [f for f in glob.glob(os.path.join(PROJ, "pose-3d", "*.trc")) if "LSTM" not in f]
    if not cands:
        print("  STILL no TRC."); return
    L = open(sorted(cands)[-1]).read().splitlines()
    mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    ps = flex(g("RShoulder"), g("RElbow"), g("RWrist"))
    rows = list(csv.DictReader(open(GT_CSV)))
    J = np.array([[float(r["J%d_%s" % (j, ax)]) for j in range(25) for ax in "xyz"] for r in rows]).reshape(-1, 25, 3)
    gt = flex(J[:, 14], J[:, 15], J[:, 16])
    n = min(len(ps), len(gt)); m = ~np.isnan(ps[:n]); best = -2
    for lag in range(0, 8):
        seg = gt[lag:lag + n]
        if len(seg) < n: continue
        best = max(best, np.corrcoef(ps[:n][m], seg[m])[0, 1])
    print("  TRC: %d frames, %d valid (%.0f%%), markerless ROM %.0f-%.0f, r vs Vicon=%.3f"
          % (n, m.sum(), 100 * m.sum() / n, np.nanmin(ps), np.nanmax(ps), best))


if __name__ == "__main__":
    rewrite_calib()
    run()
    check()
