# -*- coding: utf-8 -*-
"""Re-vérification APRÈS le fix upright : pour chaque sujet, reprojette les marqueurs
MODÈLE (upright, model_mk_world.csv) ET les marqueurs RÉELS (arm_markers_world.trc)
dans Calib_world, et compare aux 2D détectés. Si reproj modèle ≈ reproj réel ->
le modèle (debout) tombe toujours sur la personne. pose2sim_env."""
import glob, json, os, re
import numpy as np
import cv2, toml

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
P2S = r"D:\p2s_blender"; BATCH = os.path.join(ROOT, "batch")
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
CAMS = ["50591643", "58860488", "60457274", "65906101"]
HIDX = {"RShoulder": 6, "RElbow": 8, "RWrist": 10}; NAMES = ["RShoulder", "RElbow", "RWrist"]


def read_model_mk(p):
    rows = [l.split(",") for l in open(p).read().splitlines()[1:]]
    arr = np.array([[float(x) for x in r] for r in rows])
    return arr[:, 2:].reshape(-1, 3, 3)


def read_real(p):
    L = open(p).read().splitlines()
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 4])
    return np.stack([D[:, 2:5], D[:, 5:8], D[:, 8:11]], 1)


def load2d(subj, cam):
    out = {}
    for f in sorted(glob.glob(os.path.join(BATCH, subj, "pose2sim", "pose", "cam_%s_json" % cam, "*.json"))):
        fr = int(re.findall(r"_(\d+)\.json$", f)[0]); ppl = json.load(open(f)).get("people", [])
        out[fr] = np.array(ppl[0]["pose_keypoints_2d"]).reshape(-1, 3) if ppl else None
    return out


def reproj(markers, subj, cal):
    """median reproj (px) of markers(N,3,3) into all cams vs detected 2D."""
    errs = []
    for cam in CAMS:
        c = cal["cam_%s" % cam]; K = np.array(c["matrix"]); dist = np.array(c["distortions"])
        rvec = np.array(c["rotation"]).reshape(3, 1); tvec = np.array(c["translation"]).reshape(3, 1)
        kp = load2d(subj, cam); n = len(markers)
        for fr in range(n):
            if fr not in kp or kp[fr] is None: continue
            for j, nm in enumerate(NAMES):
                P3 = markers[fr, j]
                if np.isnan(P3).any() or kp[fr][HIDX[nm], 2] < 0.3: continue
                pr = cv2.projectPoints(P3.reshape(1, 1, 3), rvec, tvec, K, dist)[0].reshape(2)
                errs.append(np.linalg.norm(pr - kp[fr][HIDX[nm], :2]))
    return np.median(errs) if errs else -1


print("%-5s %14s %14s %10s" % ("subj", "reproj MODÈLE", "reproj RÉEL", "verdict"))
for subj in SUBJECTS:
    folder = os.path.join(P2S, "%s_arm26_4cam" % subj)
    cal = toml.load(os.path.join(folder, "Calib_world.toml"))
    mr = reproj(read_model_mk(os.path.join(BATCH, subj, "model_mk_world.csv")), subj, cal)
    rr = reproj(read_real(os.path.join(folder, "arm_markers_world.trc")), subj, cal)
    print("%-5s %11.1f px %11.1f px %10s" % (subj, mr, rr, "OK" if mr < 25 else "CHECK"))
