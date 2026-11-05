"""Single-camera Level 3 (BiGRU) on held-out subject s10, redrawn in the journal style of Fig. 9.

    python fig_heldout_single.py <L3_pred_example.npz> <figs dir>

Input: the file saved by the single-camera notebook (ml_data/L3_pred_example.npz, the one
plotted in its cell 119), with the arrays pred, true (frames x 13 targets), targets, r2, subj.
Output: <figs dir>/fig_heldout_single.pdf and .png (1000 dpi, at least 7480 px wide), with
the elbow moment on top and the activation, force and fatigue of the four flexors below.
"""
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

npz, out = sys.argv[1], sys.argv[2]
d = np.load(npz, allow_pickle=True)
P, Y = np.asarray(d["pred"], float), np.asarray(d["true"], float)
targets = [str(t) for t in d["targets"]]
subj = str(d["subj"]) if "subj" in d else "held-out subject"
fps = float(d["fps"]) if "fps" in d else None
x = np.arange(len(Y)) / fps if fps else np.arange(len(Y))
xlabel = "time (s)" if fps else "frame"


def r2(j):
    yt, yp = Y[:, j], P[:, j]
    return 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)


r2s = {t: (float(d["r2"][j]) if "r2" in d else r2(j)) for j, t in enumerate(targets)}
print("subject:", subj)
for t in targets:
    print("  %-16s R2 %.3f" % (t, r2s[t]))
print("  mean over the targets: %.3f" % np.mean(list(r2s.values())))

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
SIM, PRED = "#000000", "#D55E00"  # same colour code as the single-camera route in the other figures


def find(prefix, muscle):
    """Index of the target whose name starts with prefix + muscle (e.g. act_BRA, act_BRD_hand)."""
    for j, t in enumerate(targets):
        if t.lower().startswith((prefix + muscle).lower()):
            return j
    raise KeyError(prefix + muscle)


def trace(ax, j):
    ax.plot(x, Y[:, j], color=SIM, lw=1.0, label="simulated (OpenSim)")
    ax.plot(x, P[:, j], color=PRED, lw=0.9, ls=(0, (3, 1.5)), label="predicted (BiGRU, Level 3)")
    ax.set_title("$R^2$ = %.2f" % r2s[targets[j]], loc="right", fontsize=7, pad=3)
    ax.set_xlim(x[0], x[-1])
    ax.grid(axis="y", lw=0.3, color="#d9d9d9")


fig = plt.figure(figsize=(190 * MM, 180 * MM))
outer = fig.add_gridspec(2, 1, height_ratios=[0.85, 4], hspace=0.3)
gs = outer[1].subgridspec(4, 3, hspace=0.55, wspace=0.28)
ax = fig.add_subplot(outer[0])
trace(ax, find("elbow", ""))
ax.set_ylabel("moment (N$\\cdot$m)")
ax.set_xlabel(xlabel)
ax.set_title("(a) elbow moment", loc="left")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.6, borderaxespad=0.2)
FLEX = [("BIClong", "BIClong"), ("BICshort", "BICshort"), ("BRA", "BRA"), ("BRD", "BRD")]
COLS = [("act_", "activation", "(b)"), ("frc_", "force (N)", "(c)"), ("mf_", "fatigue (%)", "(d)")]
for r, (key, name) in enumerate(FLEX):
    for c, (pre, lab, letter) in enumerate(COLS):
        a = fig.add_subplot(gs[r, c])
        trace(a, find(pre, key))
        if r == 0:
            a.set_title("%s %s" % (letter, lab), loc="left")
        if c == 0:
            a.set_ylabel(name, fontweight="bold")
        if r == len(FLEX) - 1:
            a.set_xlabel(xlabel)
        else:
            a.tick_params(labelbottom=False)
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_heldout_single.pdf"), bbox_inches="tight", pad_inches=0.02)
width_in = fig.get_size_inches()[0]
fig.savefig(os.path.join(out, "fig_heldout_single.png"), dpi=max(1000, math.ceil(7480 / width_in)),
            bbox_inches="tight", pad_inches=0.02)
print("wrote fig_heldout_single.pdf and .png in", out)
