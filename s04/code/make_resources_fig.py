# pyright: reportMissingImports=false
"""Comparaison ressources OpenSim vs AI (segments empiles + callouts lisibles). -> batch/report_figs/fig_resources.png"""
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIG = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation\batch\report_figs"
fig, ax = plt.subplots(figsize=(13.5, 6.6))

# ---- OpenSim bar (y=2.0) : segments ----
osim = [("SimTK / OpenSim native DLLs", 900, "#d6492f"), ("opensim package", 74, "#e8821e"),
        ("Geometry (601 STL)", 73, "#f0a868"), (".osim", 0.11, "#7a1f12")]
x = 0
for lab, v, c in osim:
    ax.barh(2.0, v, left=x, height=0.5, color=c, edgecolor="white", lw=0.8); x += v
osim_tot = x
ax.text(450, 2.0, "SimTK / OpenSim native DLLs  =  900 MB", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
# callouts pour les petits segments
ax.annotate("opensim package\n74 MB", xy=(937, 2.27), xytext=(870, 2.85), fontsize=10, color="#b5651d", ha="center",
            arrowprops=dict(arrowstyle="->", color="#b5651d", lw=1.3))
ax.annotate("Geometry (601 STL)\n73 MB", xy=(1010, 2.27), xytext=(1075, 2.85), fontsize=10, color="#c07a30", ha="center",
            arrowprops=dict(arrowstyle="->", color="#c07a30", lw=1.3))
ax.text(osim_tot + 20, 2.0, "1047 MB", ha="left", va="center", fontsize=16, fontweight="bold", color="#d6492f")

# ---- AI bar (y=0.95) : segments ----
ai = [("model .joblib", 16.6, "#1a9a5a"), ("lightgbm lib", 4.8, "#5bc88a")]
x = 0
for lab, v, c in ai:
    ax.barh(0.95, v, left=x, height=0.5, color=c, edgecolor="white", lw=0.8); x += v
ai_tot = x
ax.text(ai_tot + 20, 0.95, "21 MB", ha="left", va="center", fontsize=16, fontweight="bold", color="#1a9a5a")
ax.text(150, 0.95, "model .joblib 16.6 MB  +  lightgbm lib 4.8 MB", ha="left", va="center", fontsize=11, color="#137a47")

# noms a gauche
ax.text(-25, 2.0, "OpenSim\nstack", ha="right", va="center", fontsize=13, fontweight="bold")
ax.text(-25, 0.95, "AI\nsurrogate", ha="right", va="center", fontsize=13, fontweight="bold")
# fleche
ax.annotate("~49x lighter", xy=(ai_tot, 1.25), xytext=(360, 1.62), fontsize=16, color="#1a9a5a", fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color="#1a9a5a", lw=2.2))

ax.set_xlim(-30, 1180); ax.set_ylim(0.3, 3.2); ax.set_yticks([]); ax.set_xlabel("deployment / storage footprint (MB)", fontsize=12)
ax.set_title("Deployment footprint: AI surrogate ~49x lighter than the OpenSim stack", fontsize=14, fontweight="bold")
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
fig.text(0.5, 0.075, "Runtime RAM:  OpenSim ~110 MB   vs   AI ~165 MB   (comparable)", ha="center", fontsize=11, color="#333")
fig.text(0.5, 0.02, "Dependencies:  native C++ SimTK / OpenSim   vs   one  pip install (lightgbm)", ha="center", fontsize=11, color="#333")
fig.subplots_adjust(bottom=0.21, left=0.12, right=0.96, top=0.9)
fig.savefig(os.path.join(FIG, "fig_resources.png"), dpi=145); plt.close()
print("wrote fig_resources.png")
