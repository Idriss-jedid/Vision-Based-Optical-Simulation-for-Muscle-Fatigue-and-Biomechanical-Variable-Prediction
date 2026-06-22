# pyright: reportMissingImports=false
"""Copie enrichie de pipeline_diagram.png : pipeline classique (durees) + AI (R2 + temps cumule + gain).
-> batch/report_figs/pipeline_diagram_full.png"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation\batch\report_figs\pipeline_diagram_full.png"
fig, ax = plt.subplots(figsize=(15.5, 11)); ax.set_xlim(0, 15.5); ax.set_ylim(0, 11.5); ax.axis("off")
CL = "#e8f0fb"; CLe = "#3a76c2"; AIc = "#fff1e0"; AIe = "#e8821e"; OUTc = "#e6f7ee"; OUTe = "#1a9a5a"; TS = "#f4f6f8"; OSc = "#fdecea"; OSe = "#d6492f"


def box(x, y, w, h, txt, fc, ec, fs=9.2, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color="#222")


def arrow(x1, y1, x2, y2, color="#555", dotted=False, lw=1.6, lab=None):
    ls = (0, (2, 2)) if dotted else "-"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=color, lw=lw, linestyle=ls))
    if lab: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.13, lab, fontsize=7.6, color=color, ha="center")


# ===== classical (durees) =====
xc = 3.5; ys = [10.6, 9.45, 8.3, 7.15, 6.0, 4.85, 3.7, 2.55, 1.3]
labels = ["Multi-camera videos\n4 cams - 50 fps", "Calibration  (t = -R*T)  ~1 s",
          "2D pose - RTMPose  ~103 s", "Triangulation -> 3D .trc  ~38 s",
          "Butterworth filtering  ~2 s", "Stage B - angles (.mot)  <0.1 s",
          "Stage C - scaled Arm26 (.osim)  <0.1 s", "OpenSim  ID + SO + 3CC   ~89 s",
          "Biomechanics LABELS\ntorque - forces - activations - fatigue"]
fcs = [CL, CL, CL, CL, CL, CL, CL, OSc, OUTc]; ecs = [CLe, CLe, CLe, CLe, CLe, CLe, CLe, OSe, OUTe]
for i, (y, t) in enumerate(zip(ys, labels)):
    box(xc, y, 3.7, 0.9, t, fcs[i], ecs[i], 9.0, i == 8)
    if i < 8: arrow(xc, y - 0.45, xc, ys[i + 1] + 0.45, color="#888")
ax.text(xc, 11.25, "CLASSICAL PIPELINE  (total ~233 s / subject)", ha="center", fontsize=12, fontweight="bold", color=CLe)

# ===== tests (gauche) =====
tests = {2: "Test 1 - 2D pose\nreproj 8.3 px", 3: "Test 2 - 3D vs Vicon\n27.8 mm - 5.5 deg",
         5: "Test 3 - angles\nelbow r=0.993 MAE 4.2 deg", 6: "Test 4 - scaling\nx0.85-1.05"}
for i, t in tests.items():
    box(0.95, ys[i], 1.85, 0.78, t, TS, "#aab", 7.4)
    arrow(1.9, ys[i], xc - 1.85, ys[i], color="#9aa", lw=1.0)

# ===== AI (montant) : R2 + temps cumule + gain =====
xa = 11.4
ai = [("L1 - AI from .mot + .osim\nR2 = 0.95  |  total 144.5 s\nsaves OpenSim ~89 s", 3.7, 6),
      ("L2 - AI from 3D joints\nR2 = 0.90  |  total 144.1 s\nsaves OpenSim + angles/scaling", 7.15, 3),
      ("L3 - AI from 2D keypoints (4 cams)\nR2 = 0.86  |  total 104.2 s\nsaves OpenSim + triangulation (~129 s)", 8.3, 2),
      ("NEXT - AI directly from video ?\nend-to-end (future)", 10.6, 0)]
for t, y, tap in ai:
    box(xa, y, 3.9, 1.05, t, AIc, AIe, 8.4)
    arrow(xc + 1.9, ys[tap], xa - 1.95, y, color=AIe, dotted=True, lab="tap")
ax.text(xa, 11.25, "AI SURROGATES  (climb up -> replace OpenSim, ~890x faster)", ha="center", fontsize=11.5, fontweight="bold", color=AIe)

box(xa, 1.25, 3.9, 0.95, "PREDICTED biomechanics\n(no OpenSim - AI inference < 0.4 s)", OUTc, OUTe, 9, True)
for _, y, _ in ai[:3]: arrow(xa, y - 0.53, xa, 1.25 + 0.5, color="#caa", lw=1.0)
ax.annotate("", xy=(xa + 2.05, 11.0), xytext=(xa + 2.05, 0.7), arrowprops=dict(arrowstyle="<-", color=AIe, lw=2, linestyle=(0, (1, 2))))
ax.text(xa + 2.25, 6, "climb\nup", fontsize=10, color=AIe, fontweight="bold", rotation=90, va="center")

fig.suptitle("Pipeline + AI surrogates with timing:  classical 233 s  vs  L1 144 s / L2 144 s / L3 104 s  (OpenSim 89 s -> AI 0.1-0.4 s)",
             fontsize=12.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.965]); fig.savefig(OUT, dpi=145, bbox_inches="tight"); plt.close()
print("wrote", OUT)
