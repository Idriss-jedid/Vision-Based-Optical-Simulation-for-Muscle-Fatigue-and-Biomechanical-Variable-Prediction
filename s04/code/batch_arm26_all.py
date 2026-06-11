# pyright: reportMissingImports=false
"""
BATCH arm26 sur tous les sujets (s03..s11), réutilise les .trc de batch/<subj>/pose2sim.
Pour chacun : scaling arm26 (Vicon) + motion (épaule+coude data-driven, de-bias vs Vicon,
2 Hz) + métriques (coude, épaule, métrique COMBINÉE, RMSE 3D, résidu overlay) + diagnostic
épaule (std) + sorties Blender world (motion_world.csv, arm_markers_world.trc, model.osim)
dans D:\\p2s_blender\\<subj>_arm26_4cam. Écrit batch/results_arm26_all.csv. biomech env.
"""
import csv, glob, json, os
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
import opensim as osim

HERE = os.path.dirname(os.path.abspath(__file__)); S04 = os.path.dirname(HERE); ROOT = os.path.dirname(S04)
BASE = os.path.join(ROOT, "Model", "arm26_paper_loaded_brd_elbow_research.osim")
SRC = r"D:\Download\fit3d_train\train"; BATCH = os.path.join(ROOT, "batch")
EXERCISE = "dumbbell_biceps_curls"
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
MK = [("RShoulder", "r_acromion"), ("RElbow", "r_humerus_epicondyle"), ("RWrist", "r_radius_styloid")]
RATE, ELB_MAX, FPS = 100.0, 128.0, 50.0
H_ZUP = np.array([[1., 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def flex(S, E, W):
    v1, v2 = S - E, W - E
    cs = np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)
    return 180.0 - np.degrees(np.arccos(cs))


def elev_of(S, E, down):
    v = E - S; vn = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip(vn @ np.asarray(down, float), -1, 1)))


def hampel(x, k=10, ns=3.0):
    x = x.astype(float).copy(); n = len(x)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1); w = x[lo:hi]; med = np.nanmedian(w)
        mad = 1.4826 * np.nanmedian(np.abs(w - med)) + 1e-9
        if not np.isnan(x[i]) and abs(x[i] - med) > ns * mad: x[i] = np.nan
    idx = np.arange(n); good = ~np.isnan(x); return np.interp(idx, idx[good], x[good])


def trc_xyz(proj):
    cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f and "filt" not in f]
    if not cands: cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f]
    L = open(sorted(cands)[-1]).read().splitlines(); mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return {n: g(n) for n in ["RShoulder", "RElbow", "RWrist"]}


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


def shoulder_lookup(scaled):
    m = osim.Model(scaled); s = m.initSystem()
    csh = m.getCoordinateSet().get("r_shoulder_elev"); cel = m.getCoordinateSet().get("r_elbow_flex")
    cel.setValue(s, np.radians(20.0), False)
    def mk(n):
        loc = m.getMarkerSet().get(n).getLocationInGround(s); return np.array([loc.get(0), loc.get(1), loc.get(2)])
    sg = np.arange(0.0, 131.0, 1.0); ev = []
    for v in sg:
        csh.setValue(s, np.radians(v)); m.realizePosition(s)
        ev.append(float(elev_of(mk("r_acromion"), mk("r_humerus_epicondyle"), [0, -1, 0])))
    return sg, np.array(ev)


def best_lag(x, gt):
    n = min(len(x), len(gt)); x = x[:n]; m = ~np.isnan(x); best = (-2, 0)
    for lag in range(0, 8):
        seg = gt[lag:lag + n]
        if len(seg) < n: continue
        r = np.corrcoef(x[m], seg[m])[0, 1]
        if r > best[0]: best = (r, lag)
    return best[0], best[1], n


def proc_rmse(ml, vic):
    """RMSE 3D des 3 marqueurs (markerless vs Vicon) après alignement rigide global."""
    n = min(len(ml["RElbow"]), len(vic));
    A = np.stack([ml[k][:n] for k in ["RShoulder", "RElbow", "RWrist"]], 1)
    B = np.stack([vic[:n, 14], vic[:n, 15], vic[:n, 16]], 1)
    good = ~np.isnan(A).any((1, 2))
    src = A[good].reshape(-1, 3); dst = B[good].reshape(-1, 3)
    R, c, t = umeyama(src, dst)
    res = np.linalg.norm((c * (src @ R.T) + t) - dst, axis=1)
    return float(np.sqrt(np.mean(res ** 2)) * 1000)   # mm


def process(subj):
    proj = os.path.join(BATCH, subj, "pose2sim")
    out_o = os.path.join(BATCH, subj, "opensim"); out_m = os.path.join(BATCH, subj, "motion")
    os.makedirs(out_o, exist_ok=True); os.makedirs(out_m, exist_ok=True)
    scaled = os.path.join(out_o, "arm26_%s_scaled.osim" % subj)
    motion = os.path.join(out_m, "curl.mot")
    DST = os.path.join("D:\\", "p2s_blender", "%s_arm26_4cam" % subj); os.makedirs(DST, exist_ok=True)

    J = np.array(json.load(open(os.path.join(SRC, subj, "joints3d_25", EXERCISE + ".json")))["joints3d_25"])
    raw = trc_xyz(proj)
    # scaling (Vicon)
    ua_gt = float(np.median(np.linalg.norm(J[:, 14] - J[:, 15], axis=1)))
    fa_gt = float(np.median(np.linalg.norm(J[:, 15] - J[:, 16], axis=1)))
    m = osim.Model(BASE); s = m.initSystem()
    def jc(j): return np.array([m.getJointSet().get(j).getChildFrame().getPositionInGround(s).get(i) for i in range(3)])
    def mk0(n): return np.array([m.getMarkerSet().get(n).getLocationInGround(s).get(i) for i in range(3)])
    ua_d = np.linalg.norm(jc("r_shoulder") - jc("r_elbow")); fa_d = np.linalg.norm(jc("r_elbow") - mk0("r_radius_styloid"))
    sfh, sff = ua_gt / ua_d, fa_gt / fa_d
    m2 = osim.Model(BASE); s2 = m2.initSystem(); ss = osim.ScaleSet()
    for body, sf in [("r_humerus", sfh), ("r_ulna_radius_hand", sff)]:
        sc = osim.Scale(); sc.setSegmentName(body); sc.setScaleFactors(osim.Vec3(sf, sf, sf)); sc.setApply(True); ss.cloneAndAppend(sc)
    m2.scale(s2, ss, False, -1.0); m2.printToXML(scaled)

    # elbow
    el_raw = hampel(medfilt(flex(raw["RShoulder"], raw["RElbow"], raw["RWrist"]), 7), 10)
    el_gt = flex(J[:, 14], J[:, 15], J[:, 16])
    r_el, lag, n = best_lag(el_raw, el_gt); el_raw = el_raw[:n]; g_el = el_gt[lag:lag + n]
    ae, be = np.polyfit(el_raw, g_el, 1); el_cal = ae * el_raw + be
    mae_el = float(np.mean(np.abs(el_cal - g_el)))
    # shoulder
    ms = hampel(medfilt(elev_of(raw["RShoulder"], raw["RElbow"], [0, -1, 0]), 7), 10)[:n]
    vs = elev_of(J[:, 14], J[:, 15], [0, 0, -1]); g_sh = vs[lag:lag + n]
    r_sh = np.corrcoef(ms, g_sh)[0, 1]
    asb, bsb = np.polyfit(ms, g_sh, 1); sh_cal = asb * ms + bsb
    sh_grid, ev_grid = shoulder_lookup(scaled); sh_deg = np.interp(sh_cal, ev_grid, sh_grid)
    sh_model_ev = np.interp(sh_deg, sh_grid, ev_grid); mae_sh = float(np.mean(np.abs(sh_model_ev - g_sh)))
    sh_std_ml, sh_std_vic = float(np.std(ms)), float(np.std(g_sh))
    # write motion (both DOF, 2 Hz)
    t = np.arange(n) / FPS; tv = np.arange(0, t[-1] + 1e-9, 1.0 / RATE)
    bb, aa = butter(4, 2.0 / (RATE / 2), btype="low")
    fl = np.clip(filtfilt(bb, aa, np.interp(tv, t, el_cal)), 0.0, ELB_MAX)
    shv = np.clip(filtfilt(bb, aa, np.interp(tv, t, sh_deg)), -30.0, 130.0)
    with open(motion, "w", newline="\n") as f:
        f.write("curl\nversion=1\nnRows=%d\nnColumns=3\ninDegrees=yes\nendheader\ntime\tr_shoulder_elev\tr_elbow_flex\n" % len(tv))
        for k in range(len(tv)): f.write("%.4f\t%.6f\t%.6f\n" % (tv[k], shv[k], fl[k]))
    # combined metrics
    combined_mae = 0.5 * (mae_el + mae_sh)
    rmse3d = proc_rmse(raw, J)

    # world outputs (FK with data-driven shoulder) + overlay residual (data vs fixed)
    mm = osim.Model(scaled); sm = mm.initSystem()
    csh = mm.getCoordinateSet().get("r_shoulder_elev"); cel = mm.getCoordinateSet().get("r_elbow_flex")
    bodies = [mm.getBodySet().get(j) for j in range(mm.getBodySet().getSize())]; bn = [b.getName() for b in bodies]
    realA = np.stack([raw[k] for k in ["RShoulder", "RElbow", "RWrist"]], 1)
    tr = np.arange(len(realA)) / FPS
    def fk_markers(use_data):
        out = np.zeros((len(tr), 3, 3))
        for k, tt in enumerate(tr):
            shval = np.interp(tt, tv, shv) if use_data else 20.0
            csh.setValue(sm, np.radians(shval), False); cel.setValue(sm, np.radians(np.interp(tt, tv, fl))); mm.realizePosition(sm)
            for j, (_, mn) in enumerate(MK):
                loc = mm.getMarkerSet().get(mn).getLocationInGround(sm); out[k, j] = [loc.get(0), loc.get(1), loc.get(2)]
        return out
    mdl = fk_markers(True); good = ~np.isnan(realA).any((1, 2))
    R, c, t0 = umeyama(realA[good].reshape(-1, 3), mdl[good].reshape(-1, 3))
    placed = (c * (realA.reshape(-1, 3) @ R.T) + t0).reshape(len(realA), 3, 3)
    resid = float(np.mean(np.linalg.norm(placed[good] - mdl[good], axis=2)) * 1000)
    Minv = np.eye(4); Minv[:3, :3] = R.T / c; Minv[:3, 3] = -(R.T @ t0) / c
    rows = []
    for tt in tr:
        csh.setValue(sm, np.radians(np.interp(tt, tv, shv)), False); cel.setValue(sm, np.radians(np.interp(tt, tv, fl))); mm.realizePosition(sm)
        row = [tt]
        for b in bodies:
            Hs = b.getTransformInGround(sm); T = Hs.T().to_numpy(); Rs = Hs.R()
            Rm = np.array([[Rs.get(r, cc) for cc in range(3)] for r in range(3)])
            Hg = np.eye(4); Hg[:3, :3] = Rm; Hg[:3, 3] = T
            Hb = H_ZUP @ (Minv @ Hg); loc = Hb[:3, 3]; rx, ry, rz = euler_xyz(Hb[:3, :3])
            row += [loc[0], loc[1], loc[2], rx, ry, rz]
        rows.append(row)
    header = "times, " + "".join(["%s_x, %s_y, %s_z, %s_rotx, %s_roty, %s_rotz, " % (b, b, b, b, b, b) for b in bn])[:-2]
    np.savetxt(os.path.join(DST, "motion_world.csv"), np.array(rows), delimiter=",", header=header)
    Mn = len(MK); names = [r for r, _ in MK]
    with open(os.path.join(DST, "arm_markers_world.trc"), "w", newline="\n") as f:
        f.write("PathFileType\t4\t(X/Y/Z)\tarm_markers_world.trc\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write("%g\t%g\t%d\t%d\tm\t%g\t1\t%d\n" % (FPS, FPS, len(realA), Mn, FPS, len(realA)))
        f.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\t\n")
        f.write("\t\t" + "\t".join("X%d\tY%d\tZ%d" % (k + 1, k + 1, k + 1) for k in range(Mn)) + "\t\n\n")
        for k in range(len(realA)):
            f.write("%d\t%.5f\t%s\n" % (k + 1, tr[k], "\t".join("%.6f\t%.6f\t%.6f" % tuple(realA[k, j]) for j in range(Mn))))
    import shutil as sh_; sh_.copy(scaled, os.path.join(DST, "model.osim"))

    return dict(subj=subj, sf_humerus=sfh, sf_forearm=sff, ua=ua_gt, fa=fa_gt,
                r_elbow=r_el, mae_elbow=mae_el, rom_lo=float(fl.min()), rom_hi=float(fl.max()),
                r_shoulder=r_sh, mae_shoulder=mae_sh, sh_std_ml=sh_std_ml, sh_std_vic=sh_std_vic,
                sh_range=float(shv.max() - shv.min()), combined_mae=combined_mae, rmse3d_mm=rmse3d, overlay_mm=resid)


def main():
    rows = []
    for subj in SUBJECTS:
        try:
            r = process(subj); rows.append(r)
            print("%s: scale H×%.3f F×%.3f | elbow r=%.3f MAE=%.1f° | shoulder r=%.3f MAE=%.1f° (std ml=%.1f vic=%.1f) | comb=%.1f° 3D=%.1fmm overlay=%.1fmm"
                  % (subj, r["sf_humerus"], r["sf_forearm"], r["r_elbow"], r["mae_elbow"], r["r_shoulder"], r["mae_shoulder"], r["sh_std_ml"], r["sh_std_vic"], r["combined_mae"], r["rmse3d_mm"], r["overlay_mm"]))
        except Exception as e:
            print("%s FAILED: %s" % (subj, e))
    cols = ["subj", "sf_humerus", "sf_forearm", "ua", "fa", "r_elbow", "mae_elbow", "rom_lo", "rom_hi",
            "r_shoulder", "mae_shoulder", "sh_std_ml", "sh_std_vic", "sh_range", "combined_mae", "rmse3d_mm", "overlay_mm"]
    out = os.path.join(BATCH, "results_arm26_all.csv")
    with open(out, "w", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("%.4f" % r[c] if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
