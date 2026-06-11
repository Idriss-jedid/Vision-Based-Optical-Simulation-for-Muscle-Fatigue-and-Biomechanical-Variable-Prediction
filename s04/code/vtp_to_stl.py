# -*- coding: utf-8 -*-
"""
Convert OpenSim ASCII-XML .vtp meshes to binary .stl WITHOUT VTK, so the
Pose2Sim_Blender add-on can import them directly (it crashes at model.py:178 when
its own vtp2stl fails because VTK is not installed in Blender's Python).

Writes <name>.stl next to each <name>.vtp in the given folder. Pure stdlib + numpy.
Run with any python (e.g. biomech).  Usage: vtp_to_stl.py [geometry_dir]
"""
import os
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_DIR = r"D:\p2s_blender\s04_arm26\Geometry"


def _floats(text):
    return np.array(text.split(), dtype=np.float64)


def vtp_to_stl(vtp_path, stl_path):
    root = ET.parse(vtp_path).getroot()
    piece = root.find(".//Piece")
    # points
    pts_da = piece.find("./Points/DataArray")
    if pts_da is None or (pts_da.get("format", "ascii") != "ascii"):
        raise ValueError("non-ascii or missing Points")
    pts = _floats(pts_da.text).reshape(-1, 3)
    # polys: connectivity + offsets
    polys = piece.find("./Polys")
    if polys is None:
        raise ValueError("no Polys")
    conn = off = None
    for da in polys.findall("DataArray"):
        if da.get("format", "ascii") != "ascii":
            raise ValueError("non-ascii Polys")
        if da.get("Name") == "connectivity":
            conn = _floats(da.text).astype(np.int64)
        elif da.get("Name") == "offsets":
            off = _floats(da.text).astype(np.int64)
    if conn is None or off is None:
        raise ValueError("missing connectivity/offsets")
    # build triangles (fan-triangulate any polygon)
    tris = []
    start = 0
    for end in off:
        face = conn[start:end]
        for i in range(1, len(face) - 1):
            tris.append((face[0], face[i], face[i + 1]))
        start = end
    tris = np.array(tris, dtype=np.int64)
    v0 = pts[tris[:, 0]]; v1 = pts[tris[:, 1]]; v2 = pts[tris[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1.0
    n = n / ln
    # binary STL
    with open(stl_path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        buf = bytearray()
        for k in range(len(tris)):
            buf += struct.pack("<12fH", n[k, 0], n[k, 1], n[k, 2],
                               v0[k, 0], v0[k, 1], v0[k, 2],
                               v1[k, 0], v1[k, 1], v1[k, 2],
                               v2[k, 0], v2[k, 1], v2[k, 2], 0)
        f.write(buf)
    return len(tris)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    vtps = [f for f in os.listdir(d) if f.lower().endswith(".vtp")]
    ok = skip = fail = 0
    for v in vtps:
        vp = os.path.join(d, v); sp = os.path.splitext(vp)[0] + ".stl"
        if os.path.exists(sp):
            skip += 1; continue
        try:
            vtp_to_stl(vp, sp); ok += 1
        except Exception as e:
            fail += 1
            if fail <= 10:
                print("  FAIL %s: %s" % (v, e))
    print("converted %d, skipped %d, failed %d  (of %d .vtp in %s)" % (ok, skip, fail, len(vtps), d))


if __name__ == "__main__":
    main()
