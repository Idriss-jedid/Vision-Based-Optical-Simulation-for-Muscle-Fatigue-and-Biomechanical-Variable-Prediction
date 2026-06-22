# -*- coding: utf-8 -*-
"""Convert motion_world.csv (body transforms loc+rot, world frame) -> motion_world.mot
(OpenSim Storage format, mêmes données). loc en mètres, rot en radians (euler XYZ)."""
import csv
import os

DST = r"D:\p2s_blender\s04_arm26_4cam"
CSV = os.path.join(DST, "motion_world.csv")
MOT = os.path.join(DST, "motion_world.mot")

# read csv (header line starts with "# times, base_x, ...")
with open(CSV) as f:
    lines = [l.rstrip("\n") for l in f if l.strip()]
hdr = lines[0].lstrip("# ").replace(", ", "\t").split("\t")   # times, base_x, ...
hdr[0] = "time"
data = [l.split(",") for l in lines[1:]]

n = len(data); ncol = len(hdr)
with open(MOT, "w", newline="\n") as f:
    f.write("motion_world\nversion=1\nnRows=%d\nnColumns=%d\ninDegrees=no\n" % (n, ncol))
    f.write("# body transforms in the world/.trc frame (loc x,y,z en m ; rot x,y,z en rad, euler XYZ)\n")
    f.write("endheader\n")
    f.write("\t".join(hdr) + "\n")
    for row in data:
        f.write("\t".join("%.6f" % float(x) for x in row) + "\n")
print("wrote %s  (%d frames, %d colonnes, bodies: %s)"
      % (MOT, n, ncol, ", ".join(set(c.rsplit("_", 1)[0] for c in hdr[1:]))))
