# -*- coding: utf-8 -*-
"""Diagnostic des dossiers Blender (8 sujets) : pour chacun, reprojette les marqueurs
arm_markers_world.trc dans les caméras Calib_world.toml et compare aux 2D détectés.
Reproj faible = caméras bien placées. Compare s04 (bon) aux autres. pose2sim_env."""
import glob, json, os, re
import numpy as np
import cv2, toml

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
P2S = r"D:\p2s_blender"
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
CAMS = ["50591643", "58860488", "60457274", "65906101"]
HIDX = {"RShoulder": 6, "RElbow": 8, "RWrist": 10}
NAMES = ["RShoulder", "RElbow", "RWrist"]


def read_world_trc(p):
    L = open(p).read().splitlines()
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 4])
    return {NAMES[k]: D[:, 2 + 3 * k:2 + 3 * k + 3] for k in range(3)}


def load2d(subj, cam):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "batch", subj, "pose2sim", "pose", "cam_%s_json" % cam, "*.json"))):
        fr = int(re.findall(r"_(\d+)\.json$", f)[0]); ppl = json.load(open(f)).get("people", [])
        out[fr] = np.array(ppl[0]["pose_keypoints_2d"]).reshape(-1, 3) if ppl else None
    return out


print("%-5s %8s %8s %8s %8s %8s" % ("subj", "cam1", "cam2", "cam3", "cam4", "moy(px)"))
for subj in SUBJECTS:
    folder = os.path.join(P2S, "%s_arm26_4cam" % subj)
    trc = read_world_trc(os.path.join(folder, "arm_markers_world.trc"))
    cal = toml.load(os.path.join(folder, "Calib_world.toml"))
    per = []
    for cam in CAMS:
        c = cal["cam_%s" % cam]
        K = np.array(c["matrix"]); dist = np.array(c["distortions"])
        rvec = np.array(c["rotation"]).reshape(3, 1); tvec = np.array(c["translation"]).reshape(3, 1)
        kp = load2d(subj, cam); errs = []
        n = len(trc["RElbow"])
        for fr in range(n):
            if fr not in kp or kp[fr] is None: continue
            for nm in NAMES:
                P3 = trc[nm][fr]
                if np.isnan(P3).any() or kp[fr][HIDX[nm], 2] < 0.3: continue
                pr = cv2.projectPoints(P3.reshape(1, 1, 3), rvec, tvec, K, dist)[0].reshape(2)
                errs.append(np.linalg.norm(pr - kp[fr][HIDX[nm], :2]))
        per.append(np.median(errs) if errs else -1)
    print("%-5s %8.1f %8.1f %8.1f %8.1f %8.1f" % (subj, per[0], per[1], per[2], per[3], np.mean(per)))
