# -*- coding: utf-8 -*-
"""Résidu overlay APRÈS le fix upright : distance moyenne entre marqueurs modèle (upright,
model_mk_world.csv) et marqueurs réels (arm_markers_world.trc), par sujet. pose2sim_env."""
import os
import numpy as np

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
P2S = r"D:\p2s_blender"; BATCH = os.path.join(ROOT, "batch")
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]


def model_mk(p):
    rows = [l.split(",") for l in open(p).read().splitlines()[1:]]
    return np.array([[float(x) for x in r] for r in rows])[:, 2:].reshape(-1, 3, 3)


def real_mk(p):
    L = open(p).read().splitlines()
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 4])
    return np.stack([D[:, 2:5], D[:, 5:8], D[:, 8:11]], 1)


print("%-5s %14s" % ("subj", "overlay (mm)"))
for subj in SUBJECTS:
    m = model_mk(os.path.join(BATCH, subj, "model_mk_world.csv"))
    r = real_mk(os.path.join(P2S, "%s_arm26_4cam" % subj, "arm_markers_world.trc"))
    n = min(len(m), len(r)); good = ~np.isnan(r[:n]).any((1, 2))
    resid = np.linalg.norm(m[:n][good] - r[:n][good], axis=2).mean() * 1000
    print("%-5s %11.1f" % (subj, resid))
