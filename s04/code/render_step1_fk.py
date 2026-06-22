# pyright: reportMissingImports=false
"""Step 1 (biomech) : pour chaque sujet, FK du modèle arm26 (curl.mot) -> marqueurs
modèle (acromion/epicondyle/styloid) placés dans le frame .trc (via Umeyama real->model
inverse). Sauvegarde batch/<subj>/model_mk_world.csv pour l'overlay (step 2)."""
import glob, os
import numpy as np
import opensim as osim

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
BATCH = os.path.join(ROOT, "batch")
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
MK = [("RShoulder", "r_acromion"), ("RElbow", "r_humerus_epicondyle"), ("RWrist", "r_radius_styloid")]
FPS = 50.0


def trc_xyz(proj):
    cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f and "filt" not in f]
    if not cands: cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f]
    L = open(sorted(cands)[-1]).read().splitlines(); mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return {n: g(n) for n, _ in MK}


def read_mot(p):
    L = open(p).read().splitlines(); i = next(k for k, l in enumerate(L) if l.strip().lower() == "endheader")
    cols = L[i + 1].split("\t"); d = np.array([[float(x) for x in r.split("\t")] for r in L[i + 2:] if r.strip()])
    return cols, d


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, Dg, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s) / len(src)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt; c = np.trace(np.diag(Dg) @ S) / (((src - mu_s) ** 2).sum() / len(src))
    return R, c, mu_d - c * R @ mu_s


for subj in SUBJECTS:
    scaled = os.path.join(BATCH, subj, "opensim", "arm26_%s_scaled.osim" % subj)
    motion = os.path.join(BATCH, subj, "motion", "curl.mot")
    m = osim.Model(scaled); s = m.initSystem()
    cols, d = read_mot(motion); tm = d[:, 0]; sh = d[:, cols.index("r_shoulder_elev")]; el = d[:, cols.index("r_elbow_flex")]
    csh = m.getCoordinateSet().get("r_shoulder_elev"); cel = m.getCoordinateSet().get("r_elbow_flex")
    real = trc_xyz(os.path.join(BATCH, subj, "pose2sim")); n = len(real["RElbow"]); tr = np.arange(n) / FPS
    mdl = np.zeros((n, 3, 3)); elang = np.zeros(n)
    for k, tt in enumerate(tr):
        ev = np.interp(tt, tm, el); csh.setValue(s, np.radians(np.interp(tt, tm, sh)), False); cel.setValue(s, np.radians(ev)); m.realizePosition(s)
        elang[k] = ev
        for j, (_, mn) in enumerate(MK):
            loc = m.getMarkerSet().get(mn).getLocationInGround(s); mdl[k, j] = [loc.get(0), loc.get(1), loc.get(2)]
    realA = np.stack([real[nm] for nm, _ in MK], 1); good = ~np.isnan(realA).any((1, 2))
    R, c, t = umeyama(realA[good].reshape(-1, 3), mdl[good].reshape(-1, 3))
    Minv = np.eye(4); Minv[:3, :3] = R.T / c; Minv[:3, 3] = -(R.T @ t) / c
    world = (Minv[:3, :3] @ mdl.reshape(-1, 3).T).T + Minv[:3, 3]
    world = world.reshape(n, 3, 3)
    out = os.path.join(BATCH, subj, "model_mk_world.csv")
    with open(out, "w", newline="\n") as f:
        f.write("frame,elbow_deg,msho_x,msho_y,msho_z,melb_x,melb_y,melb_z,mwri_x,mwri_y,mwri_z\n")
        for k in range(n):
            f.write("%d,%.2f,%s\n" % (k, elang[k], ",".join("%.5f" % v for v in world[k].reshape(-1))))
    print("%s: model markers world -> %s (frame max-flex=%d, %.0f deg)" % (subj, os.path.basename(out), int(np.argmax(elang)), elang.max()))
