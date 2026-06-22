# -*- coding: utf-8 -*-
"""
Pose2Sim writes the .trc in its own (Y-up) frame, a fixed rotation G of the
camera_parameters/Vicon (Z-up) world. So the Calib.toml cameras (Vicon frame) do NOT
sit in the same frame as the .trc markers / our world-placed arm26. This rebuilds a
Calib_world.toml whose cameras live in the .trc frame, so in Blender the cameras +
video overlay the markers and the arm26 model.

G is found by Procrustes between the .trc arm markers (Pose2Sim frame) and the Vicon
joints 14/15/16 (camera frame). New extrinsics: R' = R @ G.T , C_trc = G @ C.
Verifies by reprojecting the .trc markers with the new calib (should be a few px).
pose2sim_env (cv2).
"""
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
OUT = r"D:\p2s_blender\s04_arm26\Calib_world.toml"
SIZE = (900, 900)
VIDX = {"RShoulder": 14, "RElbow": 15, "RWrist": 16}
HIDX = {"RShoulder": 6, "RElbow": 8, "RWrist": 10}


def num(x): return repr(float(x))
def arr(a): return "[ " + ", ".join(num(v) for v in a) + ",]"
def mat(M): return "[ " + ", ".join("[ %s,]" % ", ".join(num(v) for v in r) for r in M) + ",]"


def trc_arm():
    trc = sorted(f for f in glob.glob(os.path.join(S04, "build2", "pose2sim", "pose-3d", "*.trc")) if "LSTM" not in f)[-1]
    L = open(trc).read().splitlines()
    names = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = names.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return {k: g(k) for k in VIDX}


def procrustes(src, dst):
    """rotation+translation mapping src->dst (no scale): dst ~= G@src + g0."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    G = U @ S @ Vt
    return G, mu_d - G @ mu_s


def main():
    J = np.array(json.load(open(os.path.join(S04, "joints3d_25", EXERCISE + ".json")))["joints3d_25"])
    trc = trc_arm()
    n = min(len(trc["RElbow"]), len(J))
    src, dst = [], []   # src = vicon(camera frame), dst = trc(pose2sim frame)
    for k in VIDX:
        v = J[:n, VIDX[k]]; t = trc[k][:n]
        good = ~np.isnan(t).any(1)
        src.append(v[good]); dst.append(t[good])
    src = np.vstack(src); dst = np.vstack(dst)
    G, g0 = procrustes(src, dst)          # trc = G@vicon + g0
    res = np.linalg.norm((src @ G.T + g0) - dst, axis=1)
    print("Procrustes vicon->trc: residual mean=%.1f mm  |G| det=%.3f  |g0|=%.3f m" % (1000 * res.mean(), np.linalg.det(G), np.linalg.norm(g0)))
    print("G =\n", np.round(G, 3))

    txt = ""
    rep_all = []
    for cam in CAMS:
        d = json.load(open(os.path.join(S04, "camera_parameters", cam, EXERCISE + ".json")))
        R = np.array(d["extrinsics"]["R"], float).reshape(3, 3)
        T = np.array(d["extrinsics"]["T"], float).reshape(3)         # camera centre (vicon)
        f = np.array(d["intrinsics_w_distortion"]["f"], float).reshape(-1)
        c = np.array(d["intrinsics_w_distortion"]["c"], float).reshape(-1)
        k = np.array(d["intrinsics_w_distortion"]["k"], float).reshape(-1)
        p = np.array(d["intrinsics_w_distortion"]["p"], float).reshape(-1)
        # new extrinsics in trc frame:  X_cam = R(X_vicon - T); X_trc = G X_vicon + g0
        #  => X_vicon = G.T (X_trc - g0); X_cam = R G.T (X_trc - g0) - R T + R T ... derive centre
        Rp = R @ G.T
        C_trc = G @ T + g0                                            # camera centre in trc frame
        tvec = -Rp @ C_trc
        K = [[f[0], 0.0, c[0]], [0.0, f[1], c[1]], [0.0, 0.0, 1.0]]
        Km = np.array(K); dist = np.array([k[0], k[1], p[0], p[1], k[2]])
        # verify reprojection of trc markers
        kp = {}
        for fp in sorted(glob.glob(os.path.join(POSE, "cam_%s_json" % cam, "*.json"))):
            fr = int(re.findall(r"_(\d+)\.json$", fp)[0]); ppl = json.load(open(fp)).get("people", [])
            kp[fr] = np.array(ppl[0]["pose_keypoints_2d"]).reshape(-1, 3) if ppl else None
        rv = cv2.Rodrigues(Rp)[0]
        for key in VIDX:
            for fr in range(n):
                if fr not in kp or kp[fr] is None or kp[fr][HIDX[key], 2] < 0.3 or np.isnan(trc[key][fr]).any():
                    continue
                pr = cv2.projectPoints(trc[key][fr].reshape(1, 1, 3), rv, tvec, Km, dist)[0].reshape(2)
                rep_all.append(np.linalg.norm(pr - kp[fr][HIDX[key], :2]))
        txt += ("[cam_%s]\nname = \"cam_%s\"\nsize = %s\nmatrix = %s\n"
                "distortions = %s\nrotation = %s\ntranslation = %s\nfisheye = false\n\n"
                % (cam, cam, arr([SIZE[0], SIZE[1]]), mat(K),
                   arr([k[0], k[1], p[0], p[1]]), arr(cv2.Rodrigues(Rp)[0].reshape(-1)), arr(tvec)))
    txt += "[metadata]\nadjusted = false\nerror = 0.0\n"
    open(OUT, "w", newline="\n").write(txt)
    rep_all = np.array(rep_all)
    print("Reproject .trc markers with Calib_world: median=%.1f px, mean=%.1f px (n=%d)" % (np.median(rep_all), rep_all.mean(), len(rep_all)))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
