# -*- coding: utf-8 -*-
"""Is the 60457274+65906101 pair geometrically usable? Compute the baseline angle
the two cameras subtend at the subject, and MANUALLY triangulate the arm joints
(cv2) to compare against Vicon. pose2sim_env."""
import glob
import json
import os
import re

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
EXERCISE = "dumbbell_biceps_curls"
POSE = os.path.join(S04, "build2", "pose2sim", "pose")
CAMS = ["60457274", "65906101"]
CORR = [(14, 6), (15, 8), (16, 10)]   # R shoulder/elbow/wrist


def cam_params(cam):
    d = json.load(open(os.path.join(S04, "camera_parameters", cam, EXERCISE + ".json")))
    R = np.array(d["extrinsics"]["R"], float).reshape(3, 3)
    T = np.array(d["extrinsics"]["T"], float).reshape(3)
    f = np.array(d["intrinsics_w_distortion"]["f"], float).reshape(-1)
    c = np.array(d["intrinsics_w_distortion"]["c"], float).reshape(-1)
    k = np.array(d["intrinsics_w_distortion"]["k"], float).reshape(-1)
    p = np.array(d["intrinsics_w_distortion"]["p"], float).reshape(-1)
    K = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]])
    dist = np.array([k[0], k[1], p[0], p[1], k[2]])
    return dict(R=R, T=T, K=K, dist=dist, tvec=-R @ T, P=K @ np.hstack([R, (-R @ T).reshape(3, 1)]))


def load2d(cam):
    out = {}
    for f in sorted(glob.glob(os.path.join(POSE, "cam_%s_json" % cam, "*.json"))):
        fr = int(re.findall(r"_(\d+)\.json$", f)[0])
        ppl = json.load(open(f)).get("people", [])
        out[fr] = np.array(ppl[0]["pose_keypoints_2d"]).reshape(-1, 3) if ppl else None
    return out


def main():
    J = np.array(json.load(open(os.path.join(S04, "joints3d_25", EXERCISE + ".json")))["joints3d_25"])
    subj = np.nanmean(J.reshape(-1, 3), axis=0)
    c0, c1 = cam_params(CAMS[0]), cam_params(CAMS[1])
    v0, v1 = c0["T"] - subj, c1["T"] - subj
    ang = np.degrees(np.arccos(np.clip(v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)), -1, 1)))
    print("camera centres:  %s  dist=%.2fm" % (CAMS[0], np.linalg.norm(c0["T"] - subj)))
    print("                 %s  dist=%.2fm" % (CAMS[1], np.linalg.norm(c1["T"] - subj)))
    print(">>> BASELINE ANGLE subtended at subject = %.1f deg  (want >25-30 for good triangulation)" % ang)

    k0, k1 = load2d(CAMS[0]), load2d(CAMS[1])
    errs = []
    for fr in sorted(set(k0) & set(k1)):
        a, b = k0[fr], k1[fr]
        if a is None or b is None or fr >= len(J):
            continue
        for vi, hi in CORR:
            if a[hi, 2] < 0.3 or b[hi, 2] < 0.3:
                continue
            p0 = cv2.undistortPoints(a[hi, :2].reshape(1, 1, 2), c0["K"], c0["dist"], P=c0["K"]).reshape(2)
            p1 = cv2.undistortPoints(b[hi, :2].reshape(1, 1, 2), c1["K"], c1["dist"], P=c1["K"]).reshape(2)
            X = cv2.triangulatePoints(c0["P"], c1["P"], p0.reshape(2, 1), p1.reshape(2, 1))
            X = (X[:3] / X[3]).reshape(3)
            errs.append(np.linalg.norm(X - J[fr, vi]))
    errs = np.array(errs)
    print(">>> MANUAL triangulation of arm joints vs Vicon: median 3D error = %.3f m  (n=%d)" % (np.median(errs), len(errs)))
    print("    (good ~<0.03m; >0.1m = ill-conditioned pair)")


if __name__ == "__main__":
    main()
