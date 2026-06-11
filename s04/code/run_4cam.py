# -*- coding: utf-8 -*-
"""
s04 — 4-CAMERA pipeline with the CORRECT calibration (translation = -R*T), full video.
Then compare the elbow angle (2-cam vs 4-cam) against Vicon. The old S3 "2 beats 4"
finding used the BROKEN calib, so we re-test here on equal footing. pose2sim_env.
"""
import csv
import glob
import json
import os
import shutil

import numpy as np
import cv2
import toml

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
EXERCISE = "dumbbell_biceps_curls"
CAMS = ["50591643", "58860488", "60457274", "65906101"]    # all four
SIZE = (900, 900); FPS = 50
PROJ = os.path.join(S04, "build4", "pose2sim")
GT_CSV = os.path.join(S04, "build2", "csv", "joints3d_25.csv")
DEMO = r"D:/Download/pose2sim_env/Lib/site-packages/Pose2Sim/Demo_SinglePerson/Config.toml"


def num(x): return repr(float(x))
def arr(a): return "[ " + ", ".join(num(v) for v in a) + ",]"
def mat(M): return "[ " + ", ".join("[ %s,]" % ", ".join(num(v) for v in r) for r in M) + ",]"


def build_calib():
    cal = os.path.join(PROJ, "calibration"); os.makedirs(cal, exist_ok=True)
    txt = ""
    for cam in CAMS:
        d = json.load(open(os.path.join(S04, "camera_parameters", cam, EXERCISE + ".json")))
        R = np.array(d["extrinsics"]["R"], float).reshape(3, 3)
        T = np.array(d["extrinsics"]["T"], float).reshape(3)
        tvec = (-R @ T)                                        # the calibration fix
        f = np.array(d["intrinsics_w_distortion"]["f"], float).reshape(-1)
        c = np.array(d["intrinsics_w_distortion"]["c"], float).reshape(-1)
        k = np.array(d["intrinsics_w_distortion"]["k"], float).reshape(-1)
        p = np.array(d["intrinsics_w_distortion"]["p"], float).reshape(-1)
        K = [[f[0], 0.0, c[0]], [0.0, f[1], c[1]], [0.0, 0.0, 1.0]]
        rvec = cv2.Rodrigues(R)[0].reshape(-1)
        txt += ("[cam_%s]\nname = \"cam_%s\"\nsize = %s\nmatrix = %s\n"
                "distortions = %s\nrotation = %s\ntranslation = %s\nfisheye = false\n\n"
                % (cam, cam, arr([SIZE[0], SIZE[1]]), mat(K), arr([k[0], k[1], p[0], p[1]]), arr(rvec), arr(tvec)))
    txt += "[metadata]\nadjusted = false\nerror = 0.0\n"
    open(os.path.join(cal, "Calib.toml"), "w", newline="\n").write(txt)
    print("  Calib.toml (4 cams, translation=-R*T)")


def build_videos():
    vid = os.path.join(PROJ, "videos"); os.makedirs(vid, exist_ok=True)
    for cam in CAMS:
        shutil.copy(os.path.join(S04, "videos", cam, EXERCISE + ".mp4"), os.path.join(vid, "cam_%s.mp4" % cam))


def build_config():
    cfg = toml.load(DEMO)
    cfg["project"].update(dict(project_dir=PROJ, multi_person=False, participant_height="auto",
                               participant_mass=70.0, frame_rate=FPS, frame_range="all"))
    cfg.setdefault("pose", {}).update(dict(pose_model="Body_with_feet", mode="lightweight",
                                           save_video="none", display_detection=False,
                                           overwrite_pose=True, tracking_mode="none"))
    cfg.setdefault("filtering", {})["display_figures"] = False
    cfg.setdefault("kinematics", {}).update(dict(use_simple_model=True, use_augmentation=True))
    # 4 cams: can drop the worst views per frame -> stricter reproj, min 2 to triangulate
    cfg.setdefault("triangulation", {}).update(dict(
        min_cameras_for_triangulation=2, reproj_error_threshold_triangulation=30,
        likelihood_threshold_triangulation=0.3, interp_if_gap_smaller_than=30))
    toml.dump(cfg, open(os.path.join(PROJ, "Config.toml"), "w"))
    print("  Config.toml (full video, 4 cams)")


def flex(S, E, W):
    v1, v2 = S - E, W - E
    cs = np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)
    return 180.0 - np.degrees(np.arccos(cs))


def trc_elbow(proj):
    cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f and "filt" not in f]
    if not cands:
        cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f]
    L = open(sorted(cands)[-1]).read().splitlines()
    mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return flex(g("RShoulder"), g("RElbow"), g("RWrist"))


def vicon():
    rows = list(csv.DictReader(open(GT_CSV)))
    J = np.array([[float(r["J%d_%s" % (j, ax)]) for j in range(25) for ax in "xyz"] for r in rows]).reshape(-1, 25, 3)
    return flex(J[:, 14], J[:, 15], J[:, 16])


def metrics(name, ps, gt):
    n = min(len(ps), len(gt)); m = ~np.isnan(ps[:n])
    best = (-2, 0)
    for lag in range(0, 8):
        seg = gt[lag:lag + n]
        if len(seg) < n: continue
        r = np.corrcoef(ps[:n][m], seg[m])[0, 1]
        if r > best[0]: best = (r, lag)
    g = gt[best[1]:best[1] + n]; e = ps[:n][m] - g[m]
    print("  %-6s: valid %3.0f%%  r=%.3f  MAE=%4.1f deg  ROM %.0f-%.0f (Vicon %.0f-%.0f)"
          % (name, 100 * m.sum() / n, best[0], np.mean(np.abs(e)), np.nanmin(ps), np.nanmax(ps), np.nanmin(g), np.nanmax(g)))


def main():
    print("s04 4-CAM setup (correct calib)")
    os.makedirs(PROJ, exist_ok=True)
    build_calib(); build_videos(); build_config()
    os.chdir(PROJ)
    from Pose2Sim import Pose2Sim
    print(">>> poseEstimation");     Pose2Sim.poseEstimation()
    print(">>> triangulation");      Pose2Sim.triangulation()
    print(">>> filtering");          Pose2Sim.filtering()
    print(">>> markerAugmentation"); Pose2Sim.markerAugmentation()
    print(">>> kinematics");         Pose2Sim.kinematics()
    print("\n=== s04: 2-CAM vs 4-CAM vs Vicon (both with the corrected calib) ===")
    gt = vicon()
    metrics("2-cam", trc_elbow(os.path.join(S04, "build2", "pose2sim")), gt)
    metrics("4-cam", trc_elbow(PROJ), gt)


if __name__ == "__main__":
    main()
