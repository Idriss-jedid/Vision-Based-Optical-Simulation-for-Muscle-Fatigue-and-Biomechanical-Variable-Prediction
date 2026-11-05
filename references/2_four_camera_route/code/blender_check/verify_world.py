# -*- coding: utf-8 -*-
"""Verify (1) motion_world.csv parses exactly like Pose2Sim_Blender motion.py, and
(2) the world-placed arm26 overlays the video: reproject the world arm markers into
camera 65906101 and compare to the detected 2D keypoints. pose2sim_env (needs cv2)."""
import glob
import json
import os
import re

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
CSV = r"D:\p2s_blender\s04_arm26\motion_world.csv"
TRC = r"D:\p2s_blender\s04_arm26\arm_markers_world.trc"
POSE = os.path.join(S04, "build2", "pose2sim", "pose")
CAM = "65906101"
EXERCISE = "dumbbell_biceps_curls"
CORR = {"RShoulder": 6, "RElbow": 8, "RWrist": 10}


def check_csv():
    arr = np.loadtxt(CSV, delimiter=",", skiprows=1)
    with open(CSV) as f:
        h = f.readline()
    names = [b[1:-2] for b in h.split(",")[1::6]]
    print("CSV: %d frames, %d cols, bodies parsed = %s" % (arr.shape[0], arr.shape[1], names))
    assert arr.shape[1] == 1 + 6 * len(names), "column/body mismatch"
    print("  -> parses correctly (1 + 6*%d = %d cols) [OK]" % (len(names), arr.shape[1]))


def read_trc_world():
    L = open(TRC).read().splitlines()
    names = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 4])
    out = {}
    for k, nm in enumerate(names):
        c = 2 + 3 * k
        out[nm] = D[:, c:c + 3]
    return out


def reproject_check():
    d = json.load(open(os.path.join(S04, "camera_parameters", CAM, EXERCISE + ".json")))
    R = np.array(d["extrinsics"]["R"], float).reshape(3, 3)
    T = np.array(d["extrinsics"]["T"], float).reshape(3)
    f = np.array(d["intrinsics_w_distortion"]["f"], float).reshape(-1)
    c = np.array(d["intrinsics_w_distortion"]["c"], float).reshape(-1)
    k = np.array(d["intrinsics_w_distortion"]["k"], float).reshape(-1)
    p = np.array(d["intrinsics_w_distortion"]["p"], float).reshape(-1)
    K = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]])
    dist = np.array([k[0], k[1], p[0], p[1], k[2]])
    rvec = cv2.Rodrigues(R)[0]; tvec = -R @ T
    # detected 2D
    det = {}
    for fpath in sorted(glob.glob(os.path.join(POSE, "cam_%s_json" % CAM, "*.json"))):
        fr = int(re.findall(r"_(\d+)\.json$", fpath)[0])
        ppl = json.load(open(fpath)).get("people", [])
        det[fr] = np.array(ppl[0]["pose_keypoints_2d"]).reshape(-1, 3) if ppl else None
    world = read_trc_world()
    errs = []
    n = len(world["RElbow"])
    for fr in range(n):
        if fr not in det or det[fr] is None:
            continue
        for nm, hi in CORR.items():
            P3 = world[nm][fr]
            if np.isnan(P3).any() or det[fr][hi, 2] < 0.3:
                continue
            pr = cv2.projectPoints(P3.reshape(1, 1, 3), rvec, tvec, K, dist)[0].reshape(2)
            errs.append(np.linalg.norm(pr - det[fr][hi, :2]))
    errs = np.array(errs)
    print("Reprojection of WORLD arm markers into cam %s vs detected 2D: median %.1f px, mean %.1f px (n=%d)"
          % (CAM, np.median(errs), errs.mean(), len(errs)))
    print("  -> small px = the world-placed model/markers overlay the video correctly")


if __name__ == "__main__":
    check_csv()
    reproject_check()
