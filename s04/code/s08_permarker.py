# -*- coding: utf-8 -*-
"""s08 : erreur 3D PAR marqueur (markerless vs Vicon, après alignement rigide) pour voir
si RShoulder a un vrai problème de triangulation, et test d'une métrique d'agrément
robuste pour l'épaule (corrélation des VECTEURS 3D du bras, pas l'angle scalaire). p2s env."""
import glob, json, os
import numpy as np

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
SRC = r"D:\Download\fit3d\fit3d_train\train"; EX = "dumbbell_biceps_curls"
KMAP = {"RShoulder": 14, "RElbow": 15, "RWrist": 16}


def trc(subj):
    proj = os.path.join(ROOT, "batch", subj, "pose2sim")
    cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f and "filt" not in f]
    if not cands: cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f]
    L = open(sorted(cands)[-1]).read().splitlines(); mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return {n: g(n) for n in KMAP}


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, Dg, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s) / len(src)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt; c = np.trace(np.diag(Dg) @ S) / (((src - mu_s) ** 2).sum() / len(src))
    return R, c, mu_d - c * R @ mu_s


for subj in ["s08", "s04"]:
    R = trc(subj); J = np.array(json.load(open(os.path.join(SRC, subj, "joints3d_25", EX + ".json")))["joints3d_25"])
    n = min(len(R["RElbow"]), len(J))
    A = np.stack([R[k][:n] for k in KMAP], 1); B = np.stack([J[:n, KMAP[k]] for k in KMAP], 1)
    good = ~np.isnan(A).any((1, 2))
    Rr, c, t = umeyama(A[good].reshape(-1, 3), B[good].reshape(-1, 3))
    P = (c * (A.reshape(-1, 3) @ Rr.T) + t).reshape(n, 3, 3)
    print("\n=== %s : erreur 3D par marqueur (mm) ===" % subj)
    for i, k in enumerate(KMAP):
        e = np.linalg.norm(P[good, i] - B[good, i], axis=1) * 1000
        print("  %-10s : moy %.1f  med %.1f  max %.1f mm" % (k, e.mean(), np.median(e), e.max()))
    # métrique robuste épaule : corrélation des composantes du vecteur bras (3D) markerless vs Vicon
    ua_ml = (A[:, 1] - A[:, 0]); ua_ml = ua_ml / (np.linalg.norm(ua_ml, axis=1, keepdims=True) + 1e-9)
    ua_ml_v = (ua_ml @ Rr.T)                       # markerless bras -> frame Vicon
    ua_v = (B[:, 1] - B[:, 0]); ua_v = ua_v / (np.linalg.norm(ua_v, axis=1, keepdims=True) + 1e-9)
    ang = np.degrees(np.arccos(np.clip((ua_ml_v[good] * ua_v[good]).sum(1), -1, 1)))
    print("  angle 3D bras (markerless vs Vicon) : moy %.1f° med %.1f°  <- agrément direction" % (ang.mean(), np.median(ang)))
