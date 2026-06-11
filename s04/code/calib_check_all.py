# -*- coding: utf-8 -*-
"""Vérifie la CALIBRATION de chaque sujet (s03..s11) indépendamment de l'épaule :
on reprojette les joints Vicon (épaule/coude/poignet) dans chaque caméra avec la calib
(-R*T) et on compare aux keypoints 2D détectés. Reproj faible (~quelques px) = calib OK.
pose2sim_env (cv2)."""
import glob, json, os, re
import numpy as np
import cv2

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
SRC = r"D:\Download\fit3d_train\train"; EX = "dumbbell_biceps_curls"
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
CAMS = ["50591643", "58860488", "60457274", "65906101"]
CORR = [(14, 6), (15, 8), (16, 10)]   # (Vicon idx, HALPE26 idx) shoulder/elbow/wrist


def load2d(subj, cam):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "batch", subj, "pose2sim", "pose", "cam_%s_json" % cam, "*.json"))):
        fr = int(re.findall(r"_(\d+)\.json$", f)[0]); ppl = json.load(open(f)).get("people", [])
        out[fr] = np.array(ppl[0]["pose_keypoints_2d"]).reshape(-1, 3) if ppl else None
    return out


def main():
    print("%-5s %8s %8s %8s %8s %8s" % ("subj", "cam1", "cam2", "cam3", "cam4", "moy(px)"))
    for subj in SUBJECTS:
        J = np.array(json.load(open(os.path.join(SRC, subj, "joints3d_25", EX + ".json")))["joints3d_25"])
        permcam = []
        for cam in CAMS:
            d = json.load(open(os.path.join(SRC, subj, "camera_parameters", cam, EX + ".json")))
            R = np.array(d["extrinsics"]["R"], float).reshape(3, 3); T = np.array(d["extrinsics"]["T"], float).reshape(3)
            f = np.array(d["intrinsics_w_distortion"]["f"], float).reshape(-1); c = np.array(d["intrinsics_w_distortion"]["c"], float).reshape(-1)
            k = np.array(d["intrinsics_w_distortion"]["k"], float).reshape(-1); p = np.array(d["intrinsics_w_distortion"]["p"], float).reshape(-1)
            K = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]]); dist = np.array([k[0], k[1], p[0], p[1], k[2]])
            rvec = cv2.Rodrigues(R)[0]; tvec = -R @ T
            kp = load2d(subj, cam); reps = []
            for fr, k2 in kp.items():
                if k2 is None or fr >= len(J): continue
                for vi, hi in CORR:
                    if k2[hi, 2] < 0.5: continue
                    pr = cv2.projectPoints(J[fr, vi].reshape(1, 1, 3), rvec, tvec, K, dist)[0].reshape(2)
                    reps.append(np.linalg.norm(pr - k2[hi, :2]))
            permcam.append(np.median(reps) if reps else -1)
        print("%-5s %7.1f %8.1f %8.1f %8.1f %8.1f" % (subj, permcam[0], permcam[1], permcam[2], permcam[3], np.mean(permcam)))


if __name__ == "__main__":
    main()
