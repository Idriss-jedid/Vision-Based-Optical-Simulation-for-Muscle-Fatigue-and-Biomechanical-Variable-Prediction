# -*- coding: utf-8 -*-
"""Diagnose which of s04's cameras are usable: per-camera arm-keypoint confidence
and reprojection error of the Vicon arm joints (using the GT calibration). p2s env."""
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
# (Vicon idx, HALPE26 idx) for R shoulder/elbow/wrist
CORR = [(14, 6), (15, 8), (16, 10)]


def load2d(cam):
    out = {}
    for f in sorted(glob.glob(os.path.join(POSE, "cam_%s_json" % cam, "*.json"))):
        fr = int(re.findall(r"_(\d+)\.json$", f)[0])
        ppl = json.load(open(f)).get("people", [])
        out[fr] = np.array(ppl[0]["pose_keypoints_2d"]).reshape(-1, 3) if ppl else None
    return out


def main():
    J = np.array(json.load(open(os.path.join(S04, "joints3d_25", EXERCISE + ".json")))["joints3d_25"])
    print("%-12s %8s %10s %12s" % ("camera", "frames", "armConf", "reproj(px)"))
    print("-" * 46)
    for cam in CAMS:
        d = json.load(open(os.path.join(S04, "camera_parameters", cam, EXERCISE + ".json")))
        R = np.array(d["extrinsics"]["R"], float).reshape(3, 3)
        T = np.array(d["extrinsics"]["T"], float).reshape(3)
        f = np.array(d["intrinsics_w_distortion"]["f"], float).reshape(-1)
        c = np.array(d["intrinsics_w_distortion"]["c"], float).reshape(-1)
        k = np.array(d["intrinsics_w_distortion"]["k"], float).reshape(-1)
        p = np.array(d["intrinsics_w_distortion"]["p"], float).reshape(-1)
        K = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]])
        dist = np.array([k[0], k[1], p[0], p[1], k[2]])
        rvec = cv2.Rodrigues(R)[0]; tvec = -R @ T
        kp = load2d(cam)
        conf, reproj, ndet = [], [], 0
        for fr, k2 in kp.items():
            if k2 is None or fr >= len(J):
                continue
            ndet += 1
            for vi, hi in CORR:
                conf.append(k2[hi, 2])
                if k2[hi, 2] < 0.3:
                    continue
                pr = cv2.projectPoints(J[fr, vi].reshape(1, 1, 3), rvec, tvec, K, dist)[0].reshape(2)
                reproj.append(np.linalg.norm(pr - k2[hi, :2]))
        print("%-12s %8d %10.2f %12.1f" % (cam, ndet, np.median(conf), np.median(reproj) if reproj else -1))


if __name__ == "__main__":
    main()
