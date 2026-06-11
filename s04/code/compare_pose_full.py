# -*- coding: utf-8 -*-
"""Comparison pose models sur la VIDEO COMPLETE (s04, 4-cam, calib -R*T) :
lightweight (déjà dans build4) vs balanced (build4_bal). Métriques coude + épaule
vs Vicon sur tout le clip. Choisit le meilleur. pose2sim_env."""
import csv, glob, json, os, shutil, time
import numpy as np
import cv2, toml

HERE = os.path.dirname(os.path.abspath(__file__)); S04 = os.path.dirname(HERE)
EXERCISE = "dumbbell_biceps_curls"; CAMS = ["50591643", "58860488", "60457274", "65906101"]
SIZE = (900, 900); FPS = 50
DEMO = r"D:/Download/pose2sim_env/Lib/site-packages/Pose2Sim/Demo_SinglePerson/Config.toml"
GT = os.path.join(S04, "build2", "csv", "joints3d_25.csv")
LIGHT = os.path.join(S04, "build4", "pose2sim")          # lightweight déjà fait
BAL = os.path.join(S04, "build4_bal", "pose2sim")


def num(x): return repr(float(x))
def arr(a): return "[ " + ", ".join(num(v) for v in a) + ",]"
def mat(M): return "[ " + ", ".join("[ %s,]" % ", ".join(num(v) for v in r) for r in M) + ",]"


def setup_bal():
    cal = os.path.join(BAL, "calibration"); vid = os.path.join(BAL, "videos")
    os.makedirs(cal, exist_ok=True); os.makedirs(vid, exist_ok=True)
    shutil.copy(os.path.join(LIGHT, "calibration", "Calib.toml"), os.path.join(cal, "Calib.toml"))
    for cam in CAMS:
        shutil.copy(os.path.join(LIGHT, "videos", "cam_%s.mp4" % cam), os.path.join(vid, "cam_%s.mp4" % cam))
    cfg = toml.load(DEMO)
    cfg["project"].update(dict(project_dir=BAL, multi_person=False, participant_height="auto",
                               participant_mass=70.0, frame_rate=FPS, frame_range="all"))
    cfg.setdefault("pose", {}).update(dict(pose_model="Body_with_feet", mode="balanced", det_frequency=4,
                                           save_video="none", display_detection=False, overwrite_pose=True, tracking_mode="none"))
    cfg.setdefault("filtering", {})["display_figures"] = False
    cfg.setdefault("triangulation", {}).update(dict(min_cameras_for_triangulation=2, reproj_error_threshold_triangulation=30,
                                                    likelihood_threshold_triangulation=0.3, interp_if_gap_smaller_than=30))
    toml.dump(cfg, open(os.path.join(BAL, "Config.toml"), "w"))


def flex(S, E, W):
    v1, v2 = S - E, W - E
    cs = np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)
    return 180.0 - np.degrees(np.arccos(cs))


def elevd(S, E, d):
    v = E - S; vn = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip(vn @ np.asarray(d, float), -1, 1)))


def trc(proj):
    cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f and "filt" not in f]
    if not cands: cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f]
    L = open(sorted(cands)[-1]).read().splitlines(); mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return {n: g(n) for n in ["RShoulder", "RElbow", "RWrist"]}


def vicon():
    rows = list(csv.DictReader(open(GT)))
    J = np.array([[float(r["J%d_%s" % (j, ax)]) for j in range(25) for ax in "xyz"] for r in rows]).reshape(-1, 25, 3)
    return J


def rmae(x, gt):
    n = min(len(x), len(gt)); x = x[:n]; m = ~np.isnan(x); best = (-2, 0)
    for lag in range(0, 8):
        seg = gt[lag:lag + n]
        if len(seg) < n: continue
        r = np.corrcoef(x[m], seg[m])[0, 1]
        if r > best[0]: best = (r, lag)
    g = gt[best[1]:best[1] + n]; a, b = np.polyfit(x[m], g[m], 1)
    return best[0], float(np.mean(np.abs((a * x + b)[m] - g[m])))


def metrics(proj, J):
    R = trc(proj)
    re, mae_e = rmae(flex(R["RShoulder"], R["RElbow"], R["RWrist"]), flex(J[:, 14], J[:, 15], J[:, 16]))
    rs, mae_s = rmae(elevd(R["RShoulder"], R["RElbow"], [0, -1, 0]), elevd(J[:, 14], J[:, 15], [0, 0, -1]))
    return re, mae_e, rs, mae_s


def main():
    setup_bal(); os.chdir(BAL)
    from Pose2Sim import Pose2Sim
    t0 = time.time(); Pose2Sim.poseEstimation(); pt = time.time() - t0
    Pose2Sim.triangulation(); Pose2Sim.filtering()
    J = vicon()
    print("\n===== POSE MODELS — VIDEO COMPLETE s04 4-cam (vs Vicon) =====")
    print("%-12s %8s %8s %10s %8s %10s" % ("mode", "rElb", "MAEelb", "rShoulder", "MAEsh", "poseTime"))
    le, lme, ls, lms = metrics(LIGHT, J)
    print("%-12s %8.3f %6.1f° %10.3f %6.1f° %s" % ("lightweight", le, lme, ls, lms, "(déjà fait)"))
    be, bme, bs, bms = metrics(BAL, J)
    print("%-12s %8.3f %6.1f° %10.3f %6.1f° %8.0fs" % ("balanced", be, bme, bs, bms, pt))
    win = "balanced" if bme < lme else "lightweight"
    print("\n-> MEILLEUR (MAE coude): %s  (light %.1f° vs bal %.1f°)" % (win, lme, bme))


if __name__ == "__main__":
    main()
