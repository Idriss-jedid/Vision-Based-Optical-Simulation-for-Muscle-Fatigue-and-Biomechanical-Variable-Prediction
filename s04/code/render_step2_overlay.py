# -*- coding: utf-8 -*-
"""Step 2 (pose2sim_env) : overlay du modèle sur la vidéo, par sujet. Projette les
marqueurs MODÈLE (rouge, avec le bras shoulder-elbow-wrist) et les marqueurs RÉELS
(vert) dans la caméra Calib_world, sur la frame de flexion max. -> batch/overlays/<subj>.png
Si le bras rouge (modèle) tombe sur le bras de la personne -> modèle bien placé."""
import os
import numpy as np
import cv2, toml

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
P2S = r"D:\p2s_blender"; BATCH = os.path.join(ROOT, "batch")
OUT = os.path.join(BATCH, "overlays"); os.makedirs(OUT, exist_ok=True)
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
CAM = "65906101"


def read_model_mk(p):
    rows = [l.split(",") for l in open(p).read().splitlines()[1:]]
    arr = np.array([[float(x) for x in r] for r in rows])
    frame = arr[:, 0].astype(int); elbow = arr[:, 1]; mk = arr[:, 2:].reshape(-1, 3, 3)
    return frame, elbow, mk


def read_real_trc(p):
    L = open(p).read().splitlines()
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 4])
    return {0: D[:, 2:5], 1: D[:, 5:8], 2: D[:, 8:11]}  # RSho, RElb, RWri


def proj(X, rvec, tvec, K, dist):
    return cv2.projectPoints(X.reshape(1, 1, 3), rvec, tvec, K, dist)[0].reshape(2)


for subj in SUBJECTS:
    folder = os.path.join(P2S, "%s_arm26_4cam" % subj)
    frame, elbow, mk = read_model_mk(os.path.join(BATCH, subj, "model_mk_world.csv"))
    real = read_real_trc(os.path.join(folder, "arm_markers_world.trc"))
    cal = toml.load(os.path.join(folder, "Calib_world.toml"))["cam_%s" % CAM]
    K = np.array(cal["matrix"]); dist = np.array(cal["distortions"])
    rvec = np.array(cal["rotation"]).reshape(3, 1); tvec = np.array(cal["translation"]).reshape(3, 1)
    fidx = int(np.argmax(elbow))
    cap = cv2.VideoCapture(os.path.join(folder, "videos", "cam_%s.mp4" % CAM))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx); ok, img = cap.read(); cap.release()
    if not ok: print("%s: video frame read FAIL" % subj); continue
    # model markers (red) + arm lines
    mp = [proj(mk[fidx, j], rvec, tvec, K, dist) for j in range(3)]
    cv2.line(img, tuple(mp[0].astype(int)), tuple(mp[1].astype(int)), (0, 0, 255), 2)
    cv2.line(img, tuple(mp[1].astype(int)), tuple(mp[2].astype(int)), (0, 0, 255), 2)
    for p in mp: cv2.circle(img, tuple(p.astype(int)), 6, (0, 0, 255), -1)
    # real markers (green)
    for j in range(3):
        rp = real[j][fidx]
        if not np.isnan(rp).any():
            q = proj(rp, rvec, tvec, K, dist); cv2.circle(img, tuple(q.astype(int)), 5, (0, 255, 0), 2)
    cv2.putText(img, "%s frame %d elbow %.0f deg (red=model, green=real)" % (subj, fidx, elbow[fidx]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imwrite(os.path.join(OUT, "%s.png" % subj), img)
    print("%s: overlay saved (frame %d)" % (subj, fidx))
