# pyright: reportMissingImports=false
"""Résidu d'overlay (marqueurs modèle FK vs marqueurs réels) avec la nouvelle motion
shoulder data-driven, comparé à l'ancien (shoulder fixé 20°). biomech env."""
import glob, os
import numpy as np
import opensim as osim

HERE = os.path.dirname(os.path.abspath(__file__)); S04 = os.path.dirname(HERE)
SCALED = os.path.join(S04, "build4", "opensim", "arm26_s04_scaled.osim")
MOTION = os.path.join(S04, "build4", "motion", "curl_17s.mot")
TRC = sorted(f for f in glob.glob(os.path.join(S04, "build4", "pose2sim", "pose-3d", "*.trc")) if "LSTM" not in f)[-1]
MK = [("RShoulder", "r_acromion"), ("RElbow", "r_humerus_epicondyle"), ("RWrist", "r_radius_styloid")]
FPS = 50.0


def real():
    L = open(TRC).read().splitlines(); names = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = names.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return np.stack([g(r) for r, _ in MK], 1)


def model_markers(sh_mode):
    m = osim.Model(SCALED); s = m.initSystem()
    L = open(MOTION).read().splitlines(); i = next(k for k, l in enumerate(L) if l.strip().lower() == "endheader")
    cols = L[i + 1].split("\t"); d = np.array([[float(x) for x in r.split("\t")] for r in L[i + 2:] if r.strip()])
    tm = d[:, 0]; sh = d[:, cols.index("r_shoulder_elev")]; el = d[:, cols.index("r_elbow_flex")]
    csh = m.getCoordinateSet().get("r_shoulder_elev"); cel = m.getCoordinateSet().get("r_elbow_flex")
    R = real(); n = len(R); tr = np.arange(n) / FPS; out = np.zeros((n, 3, 3))
    for k, tt in enumerate(tr):
        shval = 20.0 if sh_mode == "fixed" else np.interp(tt, tm, sh)
        csh.setValue(s, np.radians(shval), False); cel.setValue(s, np.radians(np.interp(tt, tm, el))); m.realizePosition(s)
        for j, (_, mn) in enumerate(MK):
            loc = m.getMarkerSet().get(mn).getLocationInGround(s); out[k, j] = [loc.get(0), loc.get(1), loc.get(2)]
    return R, out


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, Dg, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s) / len(src)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    Rr = U @ S @ Vt; c = np.trace(np.diag(Dg) @ S) / (((src - mu_s) ** 2).sum() / len(src))
    return Rr, c, mu_d - c * Rr @ mu_s


for mode in ["fixed", "data"]:
    R, mdl = model_markers(mode)
    good = ~np.isnan(R).any((1, 2))
    Rr, c, t0 = umeyama(R[good].reshape(-1, 3), mdl[good].reshape(-1, 3))
    placed = (c * (R.reshape(-1, 3) @ Rr.T) + t0).reshape(len(R), 3, 3)
    res = np.linalg.norm(placed[good] - mdl[good], axis=2)
    print("shoulder %-6s : résidu overlay (modèle vs réel)  moy %.1f mm,  max %.1f mm"
          % (mode, 1000 * res.mean(), 1000 * res.max()))
