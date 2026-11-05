"""Approach A (four cameras, LightGBM) on held-out subject s11, redrawn in the journal style.

    python fig_heldout_multi.py <project root> <out dir>

The fold is refitted exactly as in multicam_pipeline_report.ipynb (cells 25 and 42/45):
same dataset, 22 features, paramsA and random_state=0. The per-target R2 are printed so
that they can be checked against the notebook before the figure is used.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.multioutput import MultiOutputRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT, OUT = sys.argv[1], sys.argv[2]
MM = 1 / 25.4
HELD = "s11"

FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]
FEAT_A = ["q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el", "time", "humerus_mass", "forearm_mass", "humerus_len",
          "forearm_len", "sin_qel", "cos_qel", "sin_qsh", "cos_qsh", "abs_qd_el", "abs_qdd_el", "qd_el2", "grav_load",
          "qel_x_fmass", "cum_path_el", "cum_grav_imp"]
paramsA = dict(n_estimators=975, num_leaves=15, learning_rate=0.196, subsample=0.605,
               colsample_bytree=0.911, min_child_samples=24, reg_lambda=8.19)


def build_A(df):
    g = np.pi / 180.0
    X = df.copy()
    for c, f in [("sin_qel", np.sin), ("cos_qel", np.cos)]:
        X[c] = f(X.q_el * g)
    for c, f in [("sin_qsh", np.sin), ("cos_qsh", np.cos)]:
        X[c] = f(X.q_sh * g)
    X["abs_qd_el"] = X.qd_el.abs(); X["abs_qdd_el"] = X.qdd_el.abs(); X["qd_el2"] = X.qd_el ** 2
    X["grav_load"] = (X.forearm_mass + 2) * X.forearm_len * np.sin((X.q_sh + X.q_el) * g)
    X["qel_x_fmass"] = X.q_el * X.forearm_mass
    out = []
    for _, sub in X.groupby("subj"):
        dt = float(np.median(np.diff(sub.time.values))) or 0.02
        sub = sub.copy()
        sub["cum_path_el"] = np.cumsum(sub.abs_qd_el.values) * dt
        sub["cum_grav_imp"] = np.cumsum(np.abs(sub.grav_load.values)) * dt
        out.append(sub)
    return pd.concat(out).sort_index()


dA = build_A(pd.read_csv(os.path.join(ROOT, "dataset", "ml_dataset_A.csv")).sort_values(["subj", "time"]).reset_index(drop=True))
tr, te = dA.subj != HELD, dA.subj == HELD
xs = StandardScaler().fit(dA.loc[tr, FEAT_A]); ys = StandardScaler().fit(dA.loc[tr, TARGETS])
mdl = MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **paramsA))
mdl.fit(xs.transform(dA.loc[tr, FEAT_A]), ys.transform(dA.loc[tr, TARGETS]))
P = ys.inverse_transform(mdl.predict(xs.transform(dA.loc[te, FEAT_A])))
Y = dA.loc[te, TARGETS].values
t = dA.loc[te, "time"].values
r2 = {k: r2_score(Y[:, j], P[:, j]) for j, k in enumerate(TARGETS)}
for k, v in r2.items():
    print("%-14s %.3f" % (k, v))
print("mean R2 over 13 targets: %.3f" % np.mean(list(r2.values())))

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})
SIM, PRED = "#000000", "#0072B2"


def trace(ax, j):
    ax.plot(t, Y[:, j], color=SIM, lw=1.0, label="simulated (OpenSim)")
    ax.plot(t, P[:, j], color=PRED, lw=0.9, ls=(0, (3, 1.5)), label="predicted (LightGBM)")
    ax.set_title("$R^2$ = %.2f" % r2[TARGETS[j]], loc="right", fontsize=7, pad=3)
    ax.set_xlim(t[0], t[-1])
    ax.grid(axis="y", lw=0.3, color="#d9d9d9")


fig = plt.figure(figsize=(190 * MM, 180 * MM))
outer = fig.add_gridspec(2, 1, height_ratios=[0.85, 4], hspace=0.3)
gs = outer[1].subgridspec(4, 3, hspace=0.55, wspace=0.28)
ax = fig.add_subplot(outer[0])
trace(ax, 0)
ax.set_ylabel("moment (N$\\cdot$m)")
ax.set_xlabel("time (s)")
ax.set_xticks(np.arange(0, t[-1] + 0.1, 2))
ax.set_title("(a) elbow moment", loc="left")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.6, borderaxespad=0.2)
NAMES = {"BIClong": "BIClong", "BICshort": "BICshort", "BRA": "BRA", "BRD_hand": "BRD"}
COLS = [("act_", "activation", "(b)"), ("frc_", "force (N)", "(c)"), ("MF_", "fatigue (%)", "(d)")]
for r, mus in enumerate(FLEX):
    for c, (pre, lab, letter) in enumerate(COLS):
        ax = fig.add_subplot(gs[r, c])
        trace(ax, TARGETS.index(pre + mus))
        if r == 0:
            ax.set_title("%s %s" % (letter, lab), loc="left")
        if c == 0:
            ax.set_ylabel(NAMES[mus], fontweight="bold")
        if r == len(FLEX) - 1:
            ax.set_xlabel("time (s)")
        else:
            ax.tick_params(labelbottom=False)
os.makedirs(os.path.join(OUT, "preview"), exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_heldout_multi.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(OUT, "preview", "fig_heldout_multi.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote fig_heldout_multi")
