# -*- coding: utf-8 -*-
"""Investigue les métriques les plus faibles (s09 coude, s08 3D) : cause = lag ? bruit ?
quel marqueur ? Pour s09/s08 (+ s04 contrôle) : meilleur lag (0-15), MAE après affine,
MAE avec filtre plus fort (1 Hz), et erreur 3D PAR marqueur. pose2sim_env."""
import glob, json, os
import numpy as np
from scipy.signal import butter, filtfilt, medfilt

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
SRC = r"D:\Download\fit3d\fit3d_train\train"; EX = "dumbbell_biceps_curls"; FPS = 50.0
KMAP = {"RShoulder": 14, "RElbow": 15, "RWrist": 16}


def trc(subj):
    proj = os.path.join(ROOT, "batch", subj, "pose2sim")
    cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f and "filt" not in f]
    L = open(sorted(cands)[-1]).read().splitlines(); mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return {n: g(n) for n in KMAP}


def flex(S, E, W):
    v1, v2 = S - E, W - E
    cs = np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)
    return 180.0 - np.degrees(np.arccos(cs))


def hampel(x, k=10, ns=3.0):
    x = x.astype(float).copy(); n = len(x)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1); w = x[lo:hi]; med = np.nanmedian(w)
        mad = 1.4826 * np.nanmedian(np.abs(w - med)) + 1e-9
        if not np.isnan(x[i]) and abs(x[i] - med) > ns * mad: x[i] = np.nan
    idx = np.arange(n); good = ~np.isnan(x); return np.interp(idx, idx[good], x[good])


def lp(x, hz):
    b, a = butter(4, hz / (FPS / 2), btype="low"); return filtfilt(b, a, x)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, Dg, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s) / len(src)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    return U @ S @ Vt, mu_s, mu_d


def mae_at(x, gt, lag, n):
    g = gt[lag:lag + n]; m = ~np.isnan(x[:n])
    a, b = np.polyfit(x[:n][m], g[m], 1)
    return float(np.mean(np.abs((a * x[:n] + b)[m] - g[m])))


for subj in ["s09", "s08", "s04"]:
    R = trc(subj); J = np.array(json.load(open(os.path.join(SRC, subj, "joints3d_25", EX + ".json")))["joints3d_25"])
    el = hampel(medfilt(flex(R["RShoulder"], R["RElbow"], R["RWrist"]), 7), 10)
    gt = flex(J[:, 14], J[:, 15], J[:, 16]); n = min(len(el), len(gt))
    # best lag 0-15
    best = min(range(0, 16), key=lambda L: mae_at(el, gt, L, n - 16))
    # MAE avec filtre courant (2Hz) vs plus fort (1Hz)
    mae2 = mae_at(lp(el, 2.0), gt, best, n - 16)
    mae1 = mae_at(lp(el, 1.0), gt, best, n - 16)
    raw_mae = mae_at(el, gt, best, n - 16)
    print("\n=== %s COUDE ===  best lag=%d frames" % (subj, best))
    print("  MAE: brut %.1f° | 2Hz %.1f° | 1Hz %.1f°" % (raw_mae, mae2, mae1))
    # erreur 3D par marqueur
    A = np.stack([R[k][:n] for k in KMAP], 1); B = np.stack([J[:n, KMAP[k]] for k in KMAP], 1)
    good = ~np.isnan(A).any((1, 2)); Rr, mus, mud = umeyama(A[good].reshape(-1, 3), B[good].reshape(-1, 3))
    P = ((A.reshape(-1, 3) - mus) @ Rr.T + mud).reshape(n, 3, 3)
    for i, k in enumerate(KMAP):
        e = np.linalg.norm(P[good, i] - B[good, i], axis=1) * 1000
        print("  %-10s 3D err: moy %.1f mm" % (k, e.mean()))
