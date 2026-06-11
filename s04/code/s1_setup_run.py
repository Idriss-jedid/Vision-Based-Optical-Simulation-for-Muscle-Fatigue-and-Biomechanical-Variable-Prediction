# -*- coding: utf-8 -*-
"""
s04 — STAGE 1: build the 2-camera Pose2Sim project (60457274 + 65906101) on the
FULL ~17 s video and run the whole Pose2Sim chain.  Run with the pose2sim_env.

Fit3D layout for s04:
  s04/camera_parameters/<id>/dumbbell_biceps_curls.json   (IMAR calib)
  s04/joints3d_25/dumbbell_biceps_curls.json              (Vicon GT)
  s04/videos/<id>/dumbbell_biceps_curls.mp4

Output: s04/build2/pose2sim/ {calibration/Calib.toml, videos, Config.toml,
        pose, pose-3d/*.trc, kinematics/*.osim+*.mot}  +  s04/build2/csv/joints3d_25.csv
"""
import json
import os
import shutil

import numpy as np
import cv2
import toml

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
EXERCISE = "dumbbell_biceps_curls"
CAMS = ["60457274", "65906101"]          # the two cameras the user asked for
SIZE = (900, 900)
FPS = 50
PROJ = os.path.join(S04, "build2", "pose2sim")
DEMO = r"D:/Download/pose2sim_env/Lib/site-packages/Pose2Sim/Demo_SinglePerson/Config.toml"


def num(x): return repr(float(x))
def arr(a): return "[ " + ", ".join(num(v) for v in a) + ",]"
def mat(M): return "[ " + ", ".join("[ %s,]" % ", ".join(num(v) for v in r) for r in M) + ",]"


def build_calib():
    cal = os.path.join(PROJ, "calibration"); os.makedirs(cal, exist_ok=True)
    toml_txt = ""
    for cam in CAMS:
        d = json.load(open(os.path.join(S04, "camera_parameters", cam, EXERCISE + ".json")))
        R = np.array(d["extrinsics"]["R"], float).reshape(3, 3)
        T = np.array(d["extrinsics"]["T"], float).reshape(3)          # camera centre (IMAR)
        f = np.array(d["intrinsics_w_distortion"]["f"], float).reshape(-1)
        c = np.array(d["intrinsics_w_distortion"]["c"], float).reshape(-1)
        k = np.array(d["intrinsics_w_distortion"]["k"], float).reshape(-1)
        p = np.array(d["intrinsics_w_distortion"]["p"], float).reshape(-1)
        K = [[f[0], 0.0, c[0]], [0.0, f[1], c[1]], [0.0, 0.0, 1.0]]
        rvec = cv2.Rodrigues(R)[0].reshape(-1)
        toml_txt += ("[cam_%s]\nname = \"cam_%s\"\nsize = %s\nmatrix = %s\n"
                     "distortions = %s\nrotation = %s\ntranslation = %s\nfisheye = false\n\n"
                     % (cam, cam, arr([SIZE[0], SIZE[1]]), mat(K),
                        arr([k[0], k[1], p[0], p[1]]), arr(rvec), arr(T)))
    toml_txt += "[metadata]\nadjusted = false\nerror = 0.0\n"
    open(os.path.join(cal, "Calib.toml"), "w", newline="\n").write(toml_txt)
    assert len([x for x in toml.load(os.path.join(cal, "Calib.toml")) if x != "metadata"]) == 2
    print("  [PASS] Calib.toml (2 cams: %s)" % ", ".join(CAMS))


def build_gt():
    d = json.load(open(os.path.join(S04, "joints3d_25", EXERCISE + ".json")))
    J = np.array(d["joints3d_25"], float)
    out = os.path.join(S04, "build2", "csv"); os.makedirs(out, exist_ok=True)
    cols = ["frame"] + ["J%d_%s" % (j, ax) for j in range(J.shape[1]) for ax in "xyz"]
    with open(os.path.join(out, "joints3d_25.csv"), "w", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for t in range(J.shape[0]):
            f.write("%d," % t + ",".join("%.6f" % v for v in J[t].reshape(-1)) + "\n")
    print("  [PASS] joints3d_25.csv (%d frames)" % J.shape[0])


def build_videos():
    vid = os.path.join(PROJ, "videos"); os.makedirs(vid, exist_ok=True)
    for cam in CAMS:
        shutil.copy(os.path.join(S04, "videos", cam, EXERCISE + ".mp4"),
                    os.path.join(vid, "cam_%s.mp4" % cam))
    print("  [PASS] %d videos copied" % len(CAMS))


def build_config():
    cfg = toml.load(DEMO)
    cfg["project"].update(dict(project_dir=PROJ, multi_person=False, participant_height="auto",
                               participant_mass=70.0, frame_rate=FPS, frame_range="all"))
    cfg.setdefault("pose", {}).update(dict(pose_model="Body_with_feet", mode="lightweight",
                                           save_video="none", display_detection=False,
                                           overwrite_pose=True, tracking_mode="none"))
    cfg.setdefault("filtering", {})["display_figures"] = False
    cfg.setdefault("kinematics", {}).update(dict(use_simple_model=True, use_augmentation=True))
    cfg.setdefault("triangulation", {}).update(dict(
        min_cameras_for_triangulation=2,
        reproj_error_threshold_triangulation=50,
        likelihood_threshold_triangulation=0.2,
        interp_if_gap_smaller_than=30))
    toml.dump(cfg, open(os.path.join(PROJ, "Config.toml"), "w"))
    print("  [PASS] Config.toml (full video, 2-cam relaxed thresholds)")


def run_pose2sim():
    os.chdir(PROJ)
    from Pose2Sim import Pose2Sim
    print(">>> poseEstimation");     Pose2Sim.poseEstimation()
    print(">>> triangulation");      Pose2Sim.triangulation()
    print(">>> filtering");          Pose2Sim.filtering()
    print(">>> markerAugmentation"); Pose2Sim.markerAugmentation()
    print(">>> kinematics");         Pose2Sim.kinematics()


def main():
    print("s04 STAGE 1 — 2-cam project setup (%s)" % ", ".join(CAMS))
    os.makedirs(PROJ, exist_ok=True)
    build_calib(); build_gt(); build_videos(); build_config()
    run_pose2sim()
    print("DONE s04 stage 1")


if __name__ == "__main__":
    main()
