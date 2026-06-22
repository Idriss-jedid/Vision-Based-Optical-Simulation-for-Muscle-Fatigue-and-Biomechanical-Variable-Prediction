# pyright: reportMissingImports=false
"""FIX : place le modèle DEBOUT (upright) au lieu de l'Umeyama complet qui penche.
Rotation = YAW seul (autour de la verticale world Y) -> le modèle reste vertical ;
le yaw oriente le facing. Translation = centroïde réel. Réécrit motion_world.csv
(et garde arm_markers_world.trc). biomech env."""
import glob, os
import numpy as np
import opensim as osim

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
BATCH = os.path.join(ROOT, "batch")
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
MK = [("RShoulder", "r_acromion"), ("RElbow", "r_humerus_epicondyle"), ("RWrist", "r_radius_styloid")]
FPS = 50.0
H_ZUP = np.array([[1., 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


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


def euler_xyz(Rm):
    sy = np.sqrt(Rm[1, 0] ** 2 + Rm[0, 0] ** 2)
    if sy > 1e-6:
        return np.arctan2(Rm[2, 1], Rm[2, 2]), np.arctan2(-Rm[2, 0], sy), np.arctan2(Rm[1, 0], Rm[0, 0])
    return np.arctan2(-Rm[1, 2], Rm[1, 1]), np.arctan2(-Rm[2, 0], sy), 0.0


for subj in SUBJECTS:
    scaled = os.path.join(BATCH, subj, "opensim", "arm26_%s_scaled.osim" % subj)
    motion = os.path.join(BATCH, subj, "motion", "curl.mot")
    DST = os.path.join("D:\\", "p2s_blender", "%s_arm26_4cam" % subj)
    m = osim.Model(scaled); s = m.initSystem()
    cols, d = read_mot(motion); tm = d[:, 0]; sh = d[:, cols.index("r_shoulder_elev")]; el = d[:, cols.index("r_elbow_flex")]
    csh = m.getCoordinateSet().get("r_shoulder_elev"); cel = m.getCoordinateSet().get("r_elbow_flex")
    bodies = [m.getBodySet().get(j) for j in range(m.getBodySet().getSize())]; bn = [b.getName() for b in bodies]
    real = trc_xyz(os.path.join(BATCH, subj, "pose2sim")); n = len(real["RElbow"]); tr = np.arange(n) / FPS
    mdl = np.zeros((n, 3, 3))
    for k, tt in enumerate(tr):
        csh.setValue(s, np.radians(np.interp(tt, tm, sh)), False); cel.setValue(s, np.radians(np.interp(tt, tm, el))); m.realizePosition(s)
        for j, (_, mn) in enumerate(MK):
            loc = m.getMarkerSet().get(mn).getLocationInGround(s); mdl[k, j] = [loc.get(0), loc.get(1), loc.get(2)]
    realA = np.stack([real[nm] for nm, _ in MK], 1); good = ~np.isnan(realA).any((1, 2))
    R, c, t = umeyama(realA[good].reshape(-1, 3), mdl[good].reshape(-1, 3))
    Rinv = R.T
    # --- YAW seul autour de world-Y (garde le modèle vertical) ---
    yaw = np.arctan2(Rinv[0, 2] - Rinv[2, 0], Rinv[0, 0] + Rinv[2, 2])
    cy, sy = np.cos(yaw), np.sin(yaw)
    A = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])      # rotation pure, vertical préservé
    mu_model = mdl[good].reshape(-1, 3).mean(0); mu_real = realA[good].reshape(-1, 3).mean(0)
    b = mu_real - A @ mu_model
    Minv = np.eye(4); Minv[:3, :3] = A; Minv[:3, 3] = b
    # body transforms upright -> z-up -> csv
    rows = []
    for tt in tr:
        csh.setValue(s, np.radians(np.interp(tt, tm, sh)), False); cel.setValue(s, np.radians(np.interp(tt, tm, el))); m.realizePosition(s)
        row = [tt]
        for bdy in bodies:
            Hs = bdy.getTransformInGround(s); T = Hs.T().to_numpy(); Rs = Hs.R()
            Rm = np.array([[Rs.get(r, cc) for cc in range(3)] for r in range(3)])
            Hg = np.eye(4); Hg[:3, :3] = Rm; Hg[:3, 3] = T
            Hb = H_ZUP @ (Minv @ Hg); loc = Hb[:3, 3]; rx, ry, rz = euler_xyz(Hb[:3, :3])
            row += [loc[0], loc[1], loc[2], rx, ry, rz]
        rows.append(row)
    header = "times, " + "".join(["%s_x, %s_y, %s_z, %s_rotx, %s_roty, %s_rotz, " % (bb, bb, bb, bb, bb, bb) for bb in bn])[:-2]
    np.savetxt(os.path.join(DST, "motion_world.csv"), np.array(rows), delimiter=",", header=header)
    # save model markers world (upright) for re-render
    worldmk = (A @ mdl.reshape(-1, 3).T).T + b; worldmk = worldmk.reshape(n, 3, 3)
    elang = np.array([np.interp(tt, tm, el) for tt in tr])
    with open(os.path.join(BATCH, subj, "model_mk_world.csv"), "w", newline="\n") as f:
        f.write("frame,elbow_deg,msho_x,msho_y,msho_z,melb_x,melb_y,melb_z,mwri_x,mwri_y,mwri_z\n")
        for k in range(n): f.write("%d,%.2f,%s\n" % (k, elang[k], ",".join("%.5f" % v for v in worldmk[k].reshape(-1))))
    up_b = H_ZUP[:3, :3] @ (A @ np.array([0., 1., 0.]))
    lean = np.degrees(np.arccos(np.clip(up_b @ np.array([0., 0., 1.]), -1, 1)))
    print("%s: upright placement (yaw=%.0f°), nouveau lean=%.1f°" % (subj, np.degrees(yaw), lean))
