# pyright: reportMissingImports=false
"""Dessine le pipeline (classique descendant + AI montant) en PNG -> batch/report_figs/pipeline_diagram.png"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation\batch\report_figs\pipeline_diagram.png"
fig, ax = plt.subplots(figsize=(15, 11)); ax.set_xlim(0, 15); ax.set_ylim(0, 11.5); ax.axis("off")
CL = "#e8f0fb"; CLe = "#3a76c2"; AIc = "#fff1e0"; AIe = "#e8821e"; OUTc = "#e6f7ee"; OUTe = "#1a9a5a"; TS = "#f4f6f8"


def box(x, y, w, h, txt, fc, ec, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color="#222")


def arrow(x1, y1, x2, y2, style="-|>", color="#555", dotted=False, lw=1.6, lab=None):
    ls = (0, (2, 2)) if dotted else "-"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, color=color, lw=lw, linestyle=ls))
    if lab: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, lab, fontsize=8, color=color, ha="center")


# ===== Classical pipeline (descendant), colonne x=3.4 =====
xc = 3.4; ys = [10.6, 9.45, 8.3, 7.15, 6.0, 4.85, 3.7, 2.55, 1.3]
labels = ["Multi-camera videos\n4 cams · 50 fps", "Calibration  (t = -R*T)\n-> Calib.toml",
          "2D pose · RTMPose\n-> keypoints JSON", "Triangulation\n-> 3D .trc",
          "Butterworth filtering\n-> filtered .trc", "Stage B · angles\n-> curl.mot",
          "Stage C · scaled Arm26\n-> .osim", "OpenSim\nID + SO + 3CC",
          "Biomechanics LABELS\ntorque · forces · activations · fatigue"]
for i, (y, t) in enumerate(zip(ys, labels)):
    isout = (i == 8)
    box(xc, y, 3.5, 0.9, t, OUTc if isout else CL, OUTe if isout else CLe, 9.5, isout)
    if i < 8: arrow(xc, y - 0.45, xc, ys[i + 1] + 0.45, color="#888")
ax.text(xc, 11.25, "CLASSICAL PIPELINE  (top -> down)", ha="center", fontsize=12, fontweight="bold", color=CLe)

# ===== Tests (a gauche) =====
tests = {2: "Test 1 · 2D pose\nreproj 8.3 px (good)", 3: "Test 2 · 3D vs Vicon\n27.8 mm · 5.5 deg",
         5: "Test 3 · angles vs Vicon\nelbow r=0.993 MAE 4.2 deg", 6: "Test 4 · scaling\nx0.85-1.05 (good)"}
for i, t in tests.items():
    box(0.95, ys[i], 1.85, 0.78, t, TS, "#aab", 7.6)
    arrow(1.88, ys[i], xc - 1.75, ys[i], color="#9aa", lw=1.1)

# ===== AI surrogates (montant), colonne x=10.8 =====
xa = 10.8
ai = [("L1 · ML from .mot + .osim\nreplaces OpenSim\nR2 = 0.95", 3.7, 6),    # tap MOD (index6)
      ("L2 · ML from 3D joints\nreplaces OpenSim+angles+scaling\nR2 = 0.90", 7.15, 3),  # tap TRI (3)
      ("L3 · ML from 2D keypoints (4 cams)\nreplaces triangulation+...\nR2 = 0.86", 8.3, 2),  # tap P2D (2)
      ("NEXT · ML directly from video ?\nend-to-end (future)", 10.6, 0)]     # tap V (0)
for t, y, tapidx in ai:
    fut = "future" in t
    box(xa, y, 3.6, 1.0, t, AIc, AIe, 8.8)
    arrow(xc + 1.75, ys[tapidx], xa - 1.8, y, color=AIe, dotted=True, lab="tap")
ax.text(xa, 11.25, "AI SURROGATES  (climb up = replace more)", ha="center", fontsize=12, fontweight="bold", color=AIe)

# ===== Predicted box =====
box(xa, 1.3, 3.6, 0.95, "PREDICTED biomechanics\n(no OpenSim at inference)", OUTc, OUTe, 9, True)
for _, y, _ in ai[:3]: arrow(xa, y - 0.5, xa, 1.3 + 0.5, color="#caa", lw=1.0)

ax.annotate("", xy=(xa, 11.0), xytext=(xa, 0.6), arrowprops=dict(arrowstyle="<-", color=AIe, lw=2, linestyle=(0, (1, 2))))
ax.text(xa + 1.95, 6, "climb\nup", fontsize=10, color=AIe, fontweight="bold", rotation=90, va="center")

fig.suptitle("Vision-Based Biomechanics: classical pipeline + AI surrogates (0.95 -> 0.90 -> 0.86)",
             fontsize=13, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT, dpi=140, bbox_inches="tight"); plt.close()
print("wrote", OUT)
