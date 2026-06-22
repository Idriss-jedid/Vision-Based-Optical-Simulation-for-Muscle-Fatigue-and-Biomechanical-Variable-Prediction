# pyright: reportMissingImports=false
"""Figures de timing + pipeline annote des durees. -> batch/report_figs/fig_timing*.png, pipeline_diagram_timed.png"""
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd, numpy as np

B = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation\batch"; FIG = os.path.join(B, "report_figs")
df = pd.read_csv(os.path.join(B, "timing_pipeline.csv")).set_index("stage")["seconds"]

# ===== Fig 1: barres log de toutes les etapes =====
order = ["2D pose (RTMPose)", "Triangulation", "OpenSim Static Optimization", "OpenSim 3CC (fatigue)",
         "Filtering", "Calibration", "OpenSim Inverse Dynamics", "AI inference (Approach A)", "AI inference (3D model)"]
vals = [df.get(s, 0.01) for s in order]
grp = ["vision", "vision", "opensim", "opensim", "vision", "vision", "opensim", "ai", "ai"]
col = {"vision": "#3a76c2", "opensim": "#d6492f", "ai": "#1a9a5a"}
fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(order))[::-1]
ax.barh(y, vals, color=[col[g] for g in grp])
for yi, v in zip(y, vals): ax.text(v * 1.15, yi, ("%.2f s" % v) if v >= 0.1 else ("%.0f ms" % (v * 1000)), va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9); ax.set_xscale("log"); ax.set_xlabel("duration per subject (s, log scale)")
ax.set_title("Per-stage duration: vision (blue) · OpenSim (red) · AI inference (green)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=col[k], label=k) for k in col], loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_timing_stages.png"), dpi=140); plt.close()

# ===== Fig 2: OpenSim vs AI (speedup) =====
osim = df["OpenSim Inverse Dynamics"] + df["OpenSim Static Optimization"] + df["OpenSim 3CC (fatigue)"]
ai = df["AI inference (3D model)"]
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.bar(["OpenSim\nID+SO+3CC", "AI surrogate\n(3D)"], [osim, ai], color=["#d6492f", "#1a9a5a"], width=0.55)
ax.text(0, osim + 2, "%.1f s" % osim, ha="center", fontsize=12, fontweight="bold")
ax.text(1, ai + 2, "%.2f s\n(%.0f ms)" % (ai, ai * 1000), ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("seconds per subject"); ax.set_title("Replacing OpenSim by the AI surrogate")
ax.annotate("~%.0fx faster" % (osim / ai), xy=(1, ai), xytext=(0.5, osim * 0.6),
            fontsize=13, color="#1a9a5a", fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color="#1a9a5a", lw=2))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_timing_speedup.png"), dpi=140); plt.close()

# ===== Fig 3: pipeline annote des durees =====
fig, ax = plt.subplots(figsize=(13, 10)); ax.set_xlim(0, 13); ax.set_ylim(0, 11); ax.axis("off")
CL = "#e8f0fb"; CLe = "#3a76c2"; OSc = "#fdecea"; OSe = "#d6492f"; AIc = "#e6f7ee"; AIe = "#1a9a5a"
def box(x, y, w, h, t, fc, ec, fs=9):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=fc, ec=ec, lw=1.5))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, color="#222")
def arr(x1, y1, x2, y2, c="#888"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=c, lw=1.5))
spine = [("Multi-camera videos", 0, CL, CLe), ("Calibration  (~1 s)", 1, CL, CLe),
         ("2D pose · RTMPose  (~103 s)", 2, CL, CLe), ("Triangulation  (~38 s)", 3, CL, CLe),
         ("Filtering  (~2 s)", 4, CL, CLe), ("Stage B angles  (<0.1 s)", 5, CL, CLe),
         ("Stage C scaled Arm26  (<0.1 s)", 6, CL, CLe),
         ("OpenSim ID+SO+3CC  (~89 s)", 7, OSc, OSe), ("Biomechanics LABELS", 8, AIc, AIe)]
ys = [10.2, 9.05, 7.9, 6.75, 5.6, 4.65, 3.7, 2.5, 1.2]; xc = 4.0
for (t, i, fc, ec), y in zip(spine, ys):
    box(xc, y, 3.8, 0.85, t, fc, ec, 9.2)
    if i < 8: arr(xc, y - 0.43, xc, ys[i+1] + 0.43)
ax.text(xc, 10.85, "CLASSICAL  (total ~233 s / subject)", ha="center", fontsize=12, fontweight="bold", color=CLe)
# AI box remplace OpenSim
box(10.3, 2.5, 3.6, 1.4, "AI surrogate\nreplaces OpenSim\n~0.10 s  (~890x faster)\nmodel = 17 MB joblib", AIc, AIe, 9.5)
ax.add_patch(FancyArrowPatch((xc + 1.95, ys[6]), (10.3 - 1.85, 2.7), arrowstyle="-|>", mutation_scale=13, color=AIe, lw=2, linestyle=(0,(3,2))))
ax.text((xc+10.3)/2, 3.1, "tap 3D / .mot", fontsize=8.5, color=AIe, ha="center")
box(10.3, 1.0, 3.6, 0.8, "Predicted biomechanics\n(torque/forces/act/fatigue)", AIc, AIe, 9)
arr(10.3, 2.5 - 0.7, 10.3, 1.0 + 0.4, "#bca")
ax.text(10.3, 10.85, "AI INFERENCE", ha="center", fontsize=12, fontweight="bold", color=AIe)
fig.suptitle("Pipeline with per-stage durations — OpenSim (89 s) replaced by AI (0.10 s)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(os.path.join(FIG, "pipeline_diagram_timed.png"), dpi=140, bbox_inches="tight"); plt.close()
print("wrote fig_timing_stages.png, fig_timing_speedup.png, pipeline_diagram_timed.png")
