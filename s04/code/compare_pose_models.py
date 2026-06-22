# -*- coding: utf-8 -*-
"""
Pose-model comparison (lightweight vs balanced) on s04, 4 cameras, same window
[150,450] (~6 s, 2 reps), same fixed calib. Reports elbow-angle accuracy vs Vicon
(r, MAE, ROM) + pose-estimation time. We have Vicon GT, so the comparison is valid.
pose2sim_env.
"""
import csv
import glob
import json
import os
import shutil
import time

import numpy as np
import cv2
import toml

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
EXERCISE = "dumbbell_biceps_curls"
CAMS = ["50591643", "58860488", "60457274", "65906101"]
SIZE = (900, 900); FPS = 50
F0, F1 = 150, 450
GT = os.path.join(S04, "joints3d_25", EXERCISE + ".json")
DEMO = r"D:/Download/pose2sim_env/Lib/site-packages/Pose2Sim/Demo_SinglePerson/Config.toml"


def num(x): return repr(float(x))
def arr(a): return "[ " + ", ".join(num(v) for v in a) + ",]"
def mat(M): return "[ " + ", ".join("[ %s,]" % ", ".join(num(v) for v in r) for r in M) + ",]"


def setup(proj, mode):
    cal = os.path.join(proj, "calibration"); vid = os.path.join(proj, "videos")
    os.makedirs(cal, exist_ok=True); os.makedirs(vid, exist_ok=True)
    txt = ""
    for cam in CAMS:
        d = json.load(open(os.path.join(S04, "camera_parameters", cam, EXERCISE + ".json")))
        R = np.array(d["extrinsics"]["R"], float).reshape(3, 3); T = np.array(d["extrinsics"]["T"], float).reshape(3)
        tvec = -R @ T
        f = np.array(d["intrinsics_w_distortion"]["f"], float).reshape(-1); c = np.array(d["intrinsics_w_distortion"]["c"], float).reshape(-1)
        k = np.array(d["intrinsics_w_distortion"]["k"], float).reshape(-1); p = np.array(d["intrinsics_w_distortion"]["p"], float).reshape(-1)
        K = [[f[0], 0.0, c[0]], [0.0, f[1], c[1]], [0.0, 0.0, 1.0]]
        txt += ("[cam_%s]\nname = \"cam_%s\"\nsize = %s\nmatrix = %s\ndistortions = %s\nrotation = %s\ntranslation = %s\nfisheye = false\n\n"
                % (cam, cam, arr([SIZE[0], SIZE[1]]), mat(K), arr([k[0], k[1], p[0], p[1]]), arr(cv2.Rodrigues(R)[0].reshape(-1)), arr(tvec)))
    txt += "[metadata]\nadjusted = false\nerror = 0.0\n"
    open(os.path.join(cal, "Calib.toml"), "w", newline="\n").write(txt)
    for cam in CAMS:
        shutil.copy(os.path.join(S04, "videos", cam, EXERCISE + ".mp4"), os.path.join(vid, "cam_%s.mp4" % cam))
    cfg = toml.load(DEMO)
    cfg["project"].update(dict(project_dir=proj, multi_person=False, participant_height="auto",
                               participant_mass=70.0, frame_rate=FPS, frame_range=[F0, F1]))
    cfg.setdefault("pose", {}).update(dict(pose_model="Body_with_feet", mode=mode, det_frequency=4,
                                           save_video="none", display_detection=False, overwrite_pose=True, tracking_mode="none"))
    cfg.setdefault("filtering", {})["display_figures"] = False
    cfg.setdefault("triangulation", {}).update(dict(min_cameras_for_triangulation=2, reproj_error_threshold_triangulation=30,
                                                    likelihood_threshold_triangulation=0.3, interp_if_gap_smaller_than=30))
    toml.dump(cfg, open(os.path.join(proj, "Config.toml"), "w"))


def flex(S, E, W):
    v1, v2 = S - E, W - E
    cs = np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)
    return 180.0 - np.degrees(np.arccos(cs))


def angle_from_trc(proj):
    cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f and "filt" not in f]
    if not cands: cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f]
    L = open(sorted(cands)[-1]).read().splitlines()
    mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); cc = 2 + 3 * j; return D[:, cc:cc + 3]
    return flex(g("RShoulder"), g("RElbow"), g("RWrist"))


def run(mode):
    proj = os.path.join(S04, "build_cmp_%s" % mode, "pose2sim")
    setup(proj, mode)
    os.chdir(proj)
    from Pose2Sim import Pose2Sim
    t0 = time.time(); Pose2Sim.poseEstimation(); pose_t = time.time() - t0
    Pose2Sim.triangulation()
    return angle_from_trc(proj), pose_t


def main():
    J = np.array(json.load(open(GT))["joints3d_25"])
    gt = flex(J[:, 14], J[:, 15], J[:, 16])[F0:F1 + 1]
    print("window frames [%d,%d] (%d frames, ~%.1f s)\n" % (F0, F1, F1 - F0 + 1, (F1 - F0 + 1) / FPS))
    res = {}
    for mode in ["lightweight", "balanced"]:
        print(">>> running %s ..." % mode)
        ang, pt = run(mode); res[mode] = (ang, pt)
    print("\n=== POSE MODEL COMPARISON (4-cam, vs Vicon) ===")
    print("%-12s %9s %8s %14s %12s" % ("mode", "r", "MAE", "ROM", "pose_time"))
    for mode in ["lightweight", "balanced"]:
        ang, pt = res[mode]; n = min(len(ang), len(gt)); m = ~np.isnan(ang[:n])
        best = -2
        for lag in range(-3, 4):
            a = ang[:n]; g = np.roll(gt[:n], lag)
            r = np.corrcoef(a[m], g[m])[0, 1]; best = max(best, r)
        mae = np.mean(np.abs(ang[:n][m] - gt[:n][m]))
        print("%-12s %9.3f %6.1f° %5.0f-%-5.0f %8.1f s" % (mode, best, mae, np.nanmin(ang), np.nanmax(ang), pt))


if __name__ == "__main__":
    main()
