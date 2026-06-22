# -*- coding: utf-8 -*-
"""
BATCH : même pipeline 4-caméras (calib fix -R*T) sur TOUS les sujets Fit3D train
(s03..s11), exercice dumbbell_biceps_curls. Pour chacun : setup -> Pose2Sim
(pose->triangulation->filtering) -> métriques de l'angle markerless vs Vicon
(coude + épaule). Écrit batch/metrics_all.csv. pose2sim_env.
"""
import csv
import glob
import json
import os
import shutil
import time

import numpy as np
import cv2
import toml

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
ROOT = os.path.dirname(S04)
SRC = r"D:\Download\fit3d\fit3d_train\train"
BATCH = os.path.join(ROOT, "batch")
EXERCISE = "dumbbell_biceps_curls"
import sys
SUBJECTS = sys.argv[1:] if len(sys.argv) > 1 else ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
CAMS = ["50591643", "58860488", "60457274", "65906101"]
SIZE = (900, 900); FPS = 50
DEMO = r"D:/Download/pose2sim_env/Lib/site-packages/Pose2Sim/Demo_SinglePerson/Config.toml"


def num(x): return repr(float(x))
def arr(a): return "[ " + ", ".join(num(v) for v in a) + ",]"
def mat(M): return "[ " + ", ".join("[ %s,]" % ", ".join(num(v) for v in r) for r in M) + ",]"


def setup(subj, proj):
    cal = os.path.join(proj, "calibration"); vid = os.path.join(proj, "videos")
    os.makedirs(cal, exist_ok=True); os.makedirs(vid, exist_ok=True)
    txt = ""
    for cam in CAMS:
        d = json.load(open(os.path.join(SRC, subj, "camera_parameters", cam, EXERCISE + ".json")))
        R = np.array(d["extrinsics"]["R"], float).reshape(3, 3); T = np.array(d["extrinsics"]["T"], float).reshape(3)
        tvec = -R @ T
        f = np.array(d["intrinsics_w_distortion"]["f"], float).reshape(-1); c = np.array(d["intrinsics_w_distortion"]["c"], float).reshape(-1)
        k = np.array(d["intrinsics_w_distortion"]["k"], float).reshape(-1); p = np.array(d["intrinsics_w_distortion"]["p"], float).reshape(-1)
        K = [[f[0], 0.0, c[0]], [0.0, f[1], c[1]], [0.0, 0.0, 1.0]]
        txt += ("[cam_%s]\nname = \"cam_%s\"\nsize = %s\nmatrix = %s\ndistortions = %s\nrotation = %s\ntranslation = %s\nfisheye = false\n\n"
                % (cam, cam, arr([SIZE[0], SIZE[1]]), mat(K), arr([k[0], k[1], p[0], p[1]]), arr(cv2.Rodrigues(R)[0].reshape(-1)), arr(tvec)))
    txt += "[metadata]\nadjusted = false\nerror = 0.0\n"
    open(os.path.join(cal, "Calib.toml"), "w", newline="\n").write(txt)
    for cam in CAMS:
        shutil.copy(os.path.join(SRC, subj, "videos", cam, EXERCISE + ".mp4"), os.path.join(vid, "cam_%s.mp4" % cam))
    cfg = toml.load(DEMO)
    cfg["project"].update(dict(project_dir=proj, multi_person=False, participant_height="auto",
                               participant_mass=70.0, frame_rate=FPS, frame_range="all"))
    cfg.setdefault("pose", {}).update(dict(pose_model="Body_with_feet", mode="lightweight", det_frequency=4,
                                           save_video="none", display_detection=False, overwrite_pose=True, tracking_mode="none"))
    cfg.setdefault("filtering", {})["display_figures"] = False
    cfg.setdefault("triangulation", {}).update(dict(min_cameras_for_triangulation=2, reproj_error_threshold_triangulation=30,
                                                    likelihood_threshold_triangulation=0.3, interp_if_gap_smaller_than=30))
    toml.dump(cfg, open(os.path.join(proj, "Config.toml"), "w"))


def flex(S, E, W):
    v1, v2 = S - E, W - E
    cs = np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)
    return 180.0 - np.degrees(np.arccos(cs))


def elev(S, E, down):
    v = E - S; vn = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip(vn @ np.asarray(down, float), -1, 1)))


def trc_xyz(proj):
    cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f and "filt" not in f]
    if not cands: cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f]
    L = open(sorted(cands)[-1]).read().splitlines()
    mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return {n: g(n) for n in ["RShoulder", "RElbow", "RWrist"]}


def best_affine(x, gt):
    n = min(len(x), len(gt)); x = x[:n]; m = ~np.isnan(x)
    best = (-2, 0)
    for lag in range(0, 8):
        seg = gt[lag:lag + n]
        if len(seg) < n: continue
        r = np.corrcoef(x[m], seg[m])[0, 1]
        if r > best[0]: best = (r, lag)
    lag = best[1]; g = gt[lag:lag + n]
    a, b = np.polyfit(x[m], g[m], 1); cal = a * x + b
    mae = float(np.mean(np.abs(cal[m] - g[m])))
    return best[0], mae, cal, g, m


def metrics(subj, proj):
    raw = trc_xyz(proj)
    J = np.array(json.load(open(os.path.join(SRC, subj, "joints3d_25", EXERCISE + ".json")))["joints3d_25"])
    el_ml = flex(raw["RShoulder"], raw["RElbow"], raw["RWrist"])
    el_gt = flex(J[:, 14], J[:, 15], J[:, 16])
    sh_ml = elev(raw["RShoulder"], raw["RElbow"], [0, -1, 0])
    sh_gt = elev(J[:, 14], J[:, 15], [0, 0, -1])
    n = min(len(el_ml), len(el_gt)); valid = 100 * np.mean(~np.isnan(el_ml[:n]))
    re, mae_e, cal_e, g_e, m = best_affine(el_ml, el_gt)
    rs, mae_s, cal_s, g_s, _ = best_affine(sh_ml, sh_gt)
    ua = float(np.median(np.linalg.norm(J[:, 14] - J[:, 15], axis=1)))
    fa = float(np.median(np.linalg.norm(J[:, 15] - J[:, 16], axis=1)))
    return dict(subj=subj, frames=n, valid=valid, r_elbow=re, mae_elbow=mae_e,
                rom_lo=float(np.nanmin(cal_e)), rom_hi=float(np.nanmax(cal_e)),
                vic_lo=float(np.nanmin(g_e)), vic_hi=float(np.nanmax(g_e)),
                r_shoulder=rs, mae_shoulder=mae_s, ua=ua, fa=fa)


def main():
    os.makedirs(BATCH, exist_ok=True)
    rows = []
    for subj in SUBJECTS:
        proj = os.path.join(BATCH, subj, "pose2sim")
        print("\n========== %s ==========" % subj)
        try:
            t0 = time.time()
            setup(subj, proj)
            os.chdir(proj)
            from Pose2Sim import Pose2Sim
            Pose2Sim.poseEstimation(); Pose2Sim.triangulation(); Pose2Sim.filtering()
            r = metrics(subj, proj); r["sec"] = time.time() - t0
            rows.append(r)
            print("  %s: valid %.0f%%  elbow r=%.3f MAE=%.1f°  shoulder r=%.3f MAE=%.1f°  (%.0fs)"
                  % (subj, r["valid"], r["r_elbow"], r["mae_elbow"], r["r_shoulder"], r["mae_shoulder"], r["sec"]))
        except Exception as e:
            print("  %s FAILED: %s" % (subj, e))
            rows.append(dict(subj=subj, frames=0, valid=0, r_elbow=-1, mae_elbow=-1, rom_lo=0, rom_hi=0,
                             vic_lo=0, vic_hi=0, r_shoulder=-1, mae_shoulder=-1, ua=0, fa=0, sec=0))

    cols = ["subj", "frames", "valid", "r_elbow", "mae_elbow", "rom_lo", "rom_hi", "vic_lo", "vic_hi",
            "r_shoulder", "mae_shoulder", "ua", "fa", "sec"]
    out = os.path.join(BATCH, "metrics_all.csv")
    with open(out, "w", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("%.4f" % r[c] if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print("\n=================== METRICS (4-cam, vs Vicon) ===================")
    print("%-5s %6s %6s %8s %7s %12s %9s %7s" % ("subj", "valid", "rElb", "MAEelb", "rSh", "ROMelb", "MAEsh", "frames"))
    for r in rows:
        print("%-5s %5.0f%% %6.3f %6.1f° %7.3f %5.0f-%-5.0f %6.1f° %7d"
              % (r["subj"], r["valid"], r["r_elbow"], r["mae_elbow"], r["r_shoulder"], r["rom_lo"], r["rom_hi"], r["mae_shoulder"], r["frames"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
