# -*- coding: utf-8 -*-
"""Métriques FINALES (8 sujets) avec le correctif s08 : pour l'épaule on remplace le
Pearson r (non fiable quand l'épaule est quasi-statique) par une métrique robuste =
ERREUR DE DIRECTION 3D du bras (markerless vs Vicon, après alignement rigide), valable
quel que soit le ROM. r n'est rapporté que si l'écart-type d'épaule > 3°. p2s env."""
import csv, glob, json, os
import numpy as np
from scipy.signal import medfilt

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
SRC = r"D:\Download\fit3d\fit3d_train\train"; EX = "dumbbell_biceps_curls"
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
KMAP = {"RShoulder": 14, "RElbow": 15, "RWrist": 16}


def flex(S, E, W):
    v1, v2 = S - E, W - E
    cs = np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)
    return 180.0 - np.degrees(np.arccos(cs))


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


def best_affine(x, gt):
    n = min(len(x), len(gt)); x = x[:n]; m = ~np.isnan(x); best = (-2, 0)
    for lag in range(0, 8):
        seg = gt[lag:lag + n]
        if len(seg) < n: continue
        r = np.corrcoef(x[m], seg[m])[0, 1]
        if r > best[0]: best = (r, lag)
    lag = best[1]; g = gt[lag:lag + n]; a, b = np.polyfit(x[m], g[m], 1)
    return best[0], float(np.mean(np.abs((a * x + b)[m] - g[m]))), lag


def main():
    rows = []
    for subj in SUBJECTS:
        R = trc(subj); J = np.array(json.load(open(os.path.join(SRC, subj, "joints3d_25", EX + ".json")))["joints3d_25"])
        n = min(len(R["RElbow"]), len(J))
        # elbow
        re, mae_e, lag = best_affine(hampel(medfilt(flex(R["RShoulder"], R["RElbow"], R["RWrist"]), 7), 10), flex(J[:, 14], J[:, 15], J[:, 16]))
        # shoulder elevation MAE + std + gated r
        ms = hampel(medfilt(elev(R["RShoulder"], R["RElbow"], [0, -1, 0]), 7), 10)
        vs = elev(J[:, 14], J[:, 15], [0, 0, -1])
        rs, mae_s, _ = best_affine(ms, vs); std = float(np.std(vs[:n]))
        r_rep = rs if std > 3.0 else np.nan
        # robust shoulder agreement: 3D upper-arm direction error
        A = np.stack([R[k][:n] for k in KMAP], 1); B = np.stack([J[:n, KMAP[k]] for k in KMAP], 1)
        good = ~np.isnan(A).any((1, 2)); Rr, c, t = umeyama(A[good].reshape(-1, 3), B[good].reshape(-1, 3))
        ua = (A[:, 1] - A[:, 0]); ua = ua / (np.linalg.norm(ua, axis=1, keepdims=True) + 1e-9)
        uav = (ua @ Rr.T); uvic = (B[:, 1] - B[:, 0]); uvic = uvic / (np.linalg.norm(uvic, axis=1, keepdims=True) + 1e-9)
        dir_err = float(np.degrees(np.arccos(np.clip((uav[good] * uvic[good]).sum(1), -1, 1))).mean())
        rows.append(dict(subj=subj, r_elbow=re, mae_elbow=mae_e, sh_std=std, r_shoulder=r_rep,
                         mae_shoulder=mae_s, dir3d=dir_err, combined=0.5 * (mae_e + mae_s)))
    print("\n========== METRIQUES FINALES (8 sujets, correctif épaule) ==========")
    print("%-5s %7s %8s %8s %9s %9s %9s %9s" % ("subj", "rElb", "MAEelb", "shStd", "rSh*", "MAEsh", "dir3D", "comb"))
    for r in rows:
        rsh = "%.3f" % r["r_shoulder"] if not np.isnan(r["r_shoulder"]) else " n/a "
        print("%-5s %7.3f %6.1f° %6.1f° %9s %6.1f° %6.1f° %6.1f°"
              % (r["subj"], r["r_elbow"], r["mae_elbow"], r["sh_std"], rsh, r["mae_shoulder"], r["dir3d"], r["combined"]))
    print("\n* r épaule rapporté seulement si ROM (std) > 3° ; sinon on lit la MAE + dir3D (robustes).")
    out = os.path.join(ROOT, "batch", "final_metrics.csv")
    cols = ["subj", "r_elbow", "mae_elbow", "sh_std", "r_shoulder", "mae_shoulder", "dir3d", "combined"]
    with open(out, "w", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join((("%.4f" % r[c]) if isinstance(r[c], float) and not (isinstance(r[c], float) and np.isnan(r[c])) else ("nan" if isinstance(r[c], float) else str(r[c]))) for c in cols) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
