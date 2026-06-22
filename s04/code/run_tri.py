# -*- coding: utf-8 -*-
"""
Retry triangulation->kinematics on s04 REUSING the cached pose (overwrite_pose=False),
with very relaxed thresholds, to see whether the user's pair (60457274+65906101) can
be triangulated at all, and how well the elbow angle matches Vicon. pose2sim_env.
"""
import csv
import glob
import os

import numpy as np
import toml

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
PROJ = os.path.join(S04, "build2", "pose2sim")
GT_CSV = os.path.join(S04, "build2", "csv", "joints3d_25.csv")


def relax_and_run():
    cfgp = os.path.join(PROJ, "Config.toml")
    cfg = toml.load(cfgp)
    cfg["pose"]["overwrite_pose"] = False                      # REUSE cached lightweight pose
    cfg.setdefault("triangulation", {}).update(dict(
        min_cameras_for_triangulation=2,
        reproj_error_threshold_triangulation=200,              # very relaxed (poor view)
        likelihood_threshold_triangulation=0.1,
        interp_if_gap_smaller_than=60,
        min_frames_per_seq=3))
    toml.dump(cfg, open(cfgp, "w"))
    os.chdir(PROJ)
    from Pose2Sim import Pose2Sim
    print(">>> triangulation (relaxed)"); Pose2Sim.triangulation()
    print(">>> filtering");               Pose2Sim.filtering()


def flex(S, E, W):
    v1, v2 = S - E, W - E
    cs = np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)
    return 180.0 - np.degrees(np.arccos(cs))


def check():
    cands = [f for f in glob.glob(os.path.join(PROJ, "pose-3d", "*.trc")) if "LSTM" not in f]
    if not cands:
        print("  STILL no TRC produced."); return
    trc = sorted(cands)[-1]
    L = open(trc).read().splitlines()
    mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    ps = flex(g("RShoulder"), g("RElbow"), g("RWrist"))
    rows = list(csv.DictReader(open(GT_CSV)))
    J = np.array([[float(r["J%d_%s" % (j, ax)]) for j in range(25) for ax in "xyz"] for r in rows]).reshape(-1, 25, 3)
    gt = flex(J[:, 14], J[:, 15], J[:, 16])
    n = min(len(ps), len(gt)); m = ~np.isnan(ps[:n])
    best = -2
    for lag in range(0, 8):
        seg = gt[lag:lag + n]
        if len(seg) < n: continue
        r = np.corrcoef(ps[:n][m], seg[m])[0, 1]; best = max(best, r)
    print("  TRC produced: %d frames, %d valid (%.0f%%), correlation vs Vicon r=%.3f"
          % (n, m.sum(), 100 * m.sum() / n, best))


if __name__ == "__main__":
    relax_and_run()
    check()
