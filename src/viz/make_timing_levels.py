# pyright: reportMissingImports=false
"""Copie du pipeline avec timeline des temps cumules par niveau AI -> batch/report_figs/pipeline_diagram_timed_levels.png"""
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Patch

FIG = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation\batch\report_figs"

# durees (s)
calib, pose, triang, filt, ang, scal = 1.0, 103.0, 38.0, 2.0, 0.04, 0.05
osim = 88.66
AI = {"L1": 0.36, "L2": 0.10, "L3": 0.17}
to_stageC = calib + pose + triang + filt + ang + scal          # 144.09
to_3d = calib + pose + triang + filt                            # 144.0
to_2d = calib + pose                                            # 104.0
classic = to_stageC + osim                                      # 232.8

BL = "#3a76c2"; RD = "#d6492f"; GR = "#1a9a5a"
fig, ax = plt.subplots(figsize=(13, 6.2))
rows = [
    ("CLASSICAL (videos -> OpenSim labels)", [(0, to_stageC, BL, "vision + angles + scaling"), (to_stageC, osim, RD, "OpenSim ID+SO+3CC")], classic, None),
    ("L1 : AI from .mot + .osim", [(0, to_stageC, BL, ""), (to_stageC, AI["L1"], GR, "AI")], to_stageC + AI["L1"], "saves OpenSim (~%.0f s)" % osim),
    ("L2 : AI from 3D joints (.trc)", [(0, to_3d, BL, ""), (to_3d, AI["L2"], GR, "AI")], to_3d + AI["L2"], "saves OpenSim + angles/scaling (~%.0f s)" % osim),
    ("L3 : AI from 2D keypoints", [(0, to_2d, BL, ""), (to_2d, AI["L3"], GR, "AI")], to_2d + AI["L3"], "saves OpenSim + triangulation + ... (~%.0f s)" % (classic - to_2d)),
]
H = 0.62; ys = [3.6, 2.5, 1.5, 0.5]
for (name, segs, total, note), y in zip(rows, ys):
    for x0, w, c, lab in segs:
        ax.add_patch(FancyBboxPatch((x0, y - H/2), w, H, boxstyle="round,pad=0.0,rounding_size=0.02", fc=c, ec="white", lw=0.5))
        if lab and w > 12: ax.text(x0 + w/2, y, lab, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.text(-3, y, name, ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.text(total + 3, y, "%.1f s" % total, ha="left", va="center", fontsize=10, fontweight="bold",
            color=RD if total > 200 else GR)
    if note: ax.text(total + 22, y, note, ha="left", va="center", fontsize=8, color="#555", style="italic")
ax.axvline(classic, ls="--", color=RD, lw=1.2, alpha=.6)
ax.text(classic, 4.15, "classical full = %.0f s" % classic, ha="center", fontsize=8.5, color=RD)
ax.set_xlim(-95, 300); ax.set_ylim(0, 4.4); ax.set_yticks([]); ax.set_xlabel("cumulative time per subject (s)")
ax.set_title("Cumulative time per AI level vs the classical pipeline\n(each level taps earlier and replaces OpenSim -> huge time saving)", fontsize=12, fontweight="bold")
ax.legend(handles=[Patch(color=BL, label="vision pipeline (to tap point)"), Patch(color=RD, label="OpenSim ID+SO+3CC"), Patch(color=GR, label="AI inference")],
          loc="lower right", fontsize=9)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "pipeline_diagram_timed_levels.png"), dpi=145, bbox_inches="tight"); plt.close()
print("wrote pipeline_diagram_timed_levels.png")
print("Classical=%.1fs | L1=%.1fs | L2=%.1fs | L3=%.1fs" % (classic, to_stageC+AI["L1"], to_3d+AI["L2"], to_2d+AI["L3"]))
