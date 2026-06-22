# -*- coding: utf-8 -*-
"""Pour CHAQUE sujet : copie curl.mot (angles) -> D:\\p2s_blender\\<subj>_arm26_4cam\\motion.mot
et convertit motion_world.csv -> motion_world.mot (mêmes données, format Storage).
Comme ça les 8 dossiers ont les mêmes fichiers que s04."""
import os, shutil

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
BATCH = os.path.join(ROOT, "batch")
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]


def csv_to_mot(csv_path, mot_path):
    with open(csv_path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    hdr = lines[0].lstrip("# ").replace(", ", "\t").split("\t"); hdr[0] = "time"
    data = [l.split(",") for l in lines[1:]]
    n, ncol = len(data), len(hdr)
    with open(mot_path, "w", newline="\n") as f:
        f.write("motion_world\nversion=1\nnRows=%d\nnColumns=%d\ninDegrees=no\n" % (n, ncol))
        f.write("# body transforms (world/.trc frame): loc x,y,z (m), rot x,y,z (rad, euler XYZ)\nendheader\n")
        f.write("\t".join(hdr) + "\n")
        for row in data:
            f.write("\t".join("%.6f" % float(x) for x in row) + "\n")
    return n, ncol


for subj in SUBJECTS:
    dst = os.path.join("D:\\", "p2s_blender", "%s_arm26_4cam" % subj)
    src_mot = os.path.join(BATCH, subj, "motion", "curl.mot")
    msg = []
    # 1) motion.mot (joint angles)
    if os.path.exists(src_mot):
        shutil.copy(src_mot, os.path.join(dst, "motion.mot")); msg.append("motion.mot")
    else:
        msg.append("curl.mot MANQUANT")
    # 2) motion_world.mot (body transforms)
    csvp = os.path.join(dst, "motion_world.csv")
    if os.path.exists(csvp):
        n, c = csv_to_mot(csvp, os.path.join(dst, "motion_world.mot")); msg.append("motion_world.mot (%d×%d)" % (n, c))
    print("%s: %s" % (subj, ", ".join(msg)))
