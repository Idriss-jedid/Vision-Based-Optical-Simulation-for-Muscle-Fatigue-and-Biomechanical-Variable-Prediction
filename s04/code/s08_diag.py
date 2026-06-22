# -*- coding: utf-8 -*-
"""Diagnostic profond de l'épaule s08 (r=0.32). On teste : lag propre à l'épaule (≠ coude),
lissage adapté (le vrai mouvement d'épaule est lent), et le SNR. But : trouver pourquoi r
est bas et le corriger proprement. pose2sim_env (numpy only)."""
import glob, json, os
import numpy as np
from scipy.signal import butter, filtfilt, medfilt

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
SRC = r"D:\Download\fit3d\fit3d_train\train"; EXERCISE = "dumbbell_biceps_curls"; FPS = 50.0


def elev(S, E, d):
    v = E - S; vn = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip(vn @ np.asarray(d, float), -1, 1)))


def hampel(x, k=10, ns=3.0):
    x = x.astype(float).copy(); n = len(x)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1); w = x[lo:hi]; med = np.nanmedian(w)
        mad = 1.4826 * np.nanmedian(np.abs(w - med)) + 1e-9
        if not np.isnan(x[i]) and abs(x[i] - med) > ns * mad: x[i] = np.nan
    idx = np.arange(n); good = ~np.isnan(x); return np.interp(idx, idx[good], x[good])


def lp(x, hz):
    if hz is None: return x
    b, a = butter(4, hz / (FPS / 2), btype="low"); return filtfilt(b, a, x)


def trc(subj):
    proj = os.path.join(ROOT, "batch", subj, "pose2sim")
    cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f and "filt" not in f]
    if not cands: cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f]
    L = open(sorted(cands)[-1]).read().splitlines(); mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return {n: g(n) for n in ["RShoulder", "RElbow", "RWrist"]}


def corr_at(x, gt, lag, n):
    seg = gt[lag:lag + n]
    if len(seg) < n: return -2
    m = ~np.isnan(x[:n])
    return np.corrcoef(x[:n][m], seg[m])[0, 1]


def analyze(subj):
    R = trc(subj)
    J = np.array(json.load(open(os.path.join(SRC, subj, "joints3d_25", EXERCISE + ".json")))["joints3d_25"])
    ms = hampel(medfilt(elev(R["RShoulder"], R["RElbow"], [0, -1, 0]), 7), 10)
    vs = elev(J[:, 14], J[:, 15], [0, 0, -1])
    n = min(len(ms), len(vs)); ms = ms[:n]
    print("\n=== %s épaule ===  std markerless=%.1f° / Vicon=%.1f°" % (subj, np.std(ms), np.std(vs[:n])))
    # (a) lag propre épaule
    best = max(range(0, 8), key=lambda L: corr_at(ms, vs, L, n - 7))
    print("  lag propre épaule = %d (vs elbow lag souvent 0-2)" % best)
    g = vs[best:best + n]
    # (b) lissages
    print("  %-22s %8s" % ("traitement", "r"))
    for hz, lab in [(None, "brut"), (4.0, "lowpass 4 Hz"), (2.0, "lowpass 2 Hz"), (1.0, "lowpass 1 Hz"), (0.5, "lowpass 0.5 Hz")]:
        a = lp(ms, hz); b = lp(g[:len(a)], hz); m = min(len(a), len(b))
        r = np.corrcoef(a[:m], b[:m])[0, 1]
        print("  %-22s %8.3f" % (lab, r))


for subj in ["s08", "s04", "s09"]:
    analyze(subj)
