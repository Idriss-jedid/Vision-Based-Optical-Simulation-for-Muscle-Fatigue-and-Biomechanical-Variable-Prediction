# -*- coding: utf-8 -*-
"""Teste si le faible r d'épaule (s08...) vient d'un MISMATCH DE FRAME : l'élévation
markerless est calculée vs -Y (frame Pose2Sim) et la Vicon vs -Z (frame Vicon). On
recalcule les DEUX dans le MÊME frame (markerless aligné sur Vicon via Procrustes) et
on compare le r. p2s env."""
import glob, json, os
import numpy as np
from scipy.signal import medfilt

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
SRC = r"D:\Download\fit3d\fit3d_train\train"; EX = "dumbbell_biceps_curls"
KMAP = {"RShoulder": 14, "RElbow": 15, "RWrist": 16}


def hampel(x, k=10, ns=3.0):
    x = x.astype(float).copy(); n = len(x)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1); w = x[lo:hi]; med = np.nanmedian(w)
        mad = 1.4826 * np.nanmedian(np.abs(w - med)) + 1e-9
        if not np.isnan(x[i]) and abs(x[i] - med) > ns * mad: x[i] = np.nan
    idx = np.arange(n); good = ~np.isnan(x); return np.interp(idx, idx[good], x[good])


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
    R = U @ S @ Vt; return R


def elev(v, d):
    vn = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip(vn @ np.asarray(d, float), -1, 1)))


def best_r(x, gt):
    n = min(len(x), len(gt)); x = x[:n]; m = ~np.isnan(x); best = -2
    for lag in range(0, 8):
        seg = gt[lag:lag + n]
        if len(seg) < n: continue
        best = max(best, np.corrcoef(x[m], seg[m])[0, 1])
    return best


print("%-5s %8s %14s %14s" % ("subj", "stdVic", "r (frames diff)", "r (même frame)"))
for subj in ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]:
    R = trc(subj); J = np.array(json.load(open(os.path.join(SRC, subj, "joints3d_25", EX + ".json")))["joints3d_25"])
    n = min(len(R["RElbow"]), len(J))
    A = np.stack([R[k][:n] for k in KMAP], 1); B = np.stack([J[:n, KMAP[k]] for k in KMAP], 1)
    good = ~np.isnan(A).any((1, 2)); Rr = umeyama(A[good].reshape(-1, 3), B[good].reshape(-1, 3))
    # markerless upper-arm transformé dans le frame Vicon
    ua_ml = A[:, 1] - A[:, 0]; ua_ml_v = ua_ml @ Rr.T
    ua_vic = B[:, 1] - B[:, 0]
    # ancienne méthode (frames différents)
    e_ml_old = hampel(medfilt(elev(ua_ml, [0, -1, 0]), 7), 10)
    e_vic = elev(ua_vic, [0, 0, -1])
    r_old = best_r(e_ml_old, e_vic)
    # nouvelle (même frame Vicon, -Z)
    e_ml_new = hampel(medfilt(elev(ua_ml_v, [0, 0, -1]), 7), 10)
    r_new = best_r(e_ml_new, e_vic)
    print("%-5s %7.1f° %14.3f %14.3f" % (subj, np.std(e_vic), r_old, r_new))
