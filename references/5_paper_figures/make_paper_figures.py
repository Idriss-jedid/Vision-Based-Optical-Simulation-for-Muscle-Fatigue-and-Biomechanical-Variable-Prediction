"""Journal-style figures for the BSPC manuscript (Elsevier artwork guidelines).

    python make_paper_figures.py <project root> <osim json> <out dir>

Style: Arial, 7-8 pt at final size, full-width figures of 190 mm, vector PDF,
Okabe-Ito colour-blind-safe palette with hatching so that every figure also
reads in greyscale, panel labels (a), (b), ... All numbers are the ones of the
manuscript tables (project reports, multi-camera notebook, demo data).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT, OSIM_JSON, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
FULL = 190 * MM

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "hatch.linewidth": 0.5,
})
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00", sky="#56B4E9",
          purple="#CC79A7", grey="#8C8C8C", black="#000000")
MUSCLES = [("BIClong", OI["blue"], "biceps long head (BIClong)"), ("BICshort", OI["sky"], "biceps short head (BICshort)"),
           ("BRA", OI["verm"], "brachialis (BRA)"), ("BRD_hand", OI["orange"], "brachioradialis (BRD)")]
ROUTE = {"multi": (OI["blue"], "Four cameras (Pose2Sim route)", ""),
         "single": (OI["verm"], "One camera (OpenCap Monocular route)", "///")}
LEVEL = [("joint angles", OI["blue"], ""), ("3D joints", OI["orange"], "...."), ("2D keypoints", OI["green"], "////")]
SUBJ = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]


def label(ax, s, x=-0.13, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    os.makedirs(os.path.join(OUT, "preview"), exist_ok=True)
    fig.savefig(os.path.join(OUT, "preview", name + ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def image(ax, path, crop=None):
    """Show an image; crop is (y0, y1) for a band of rows or (x0, y0, x1, y1) for a box."""
    img = plt.imread(path)
    if crop and len(crop) == 2:
        img = img[crop[0]:crop[1]]
    elif crop:
        img = img[crop[1]:crop[3], crop[0]:crop[2]]
    ax.imshow(img)
    ax.set_axis_off()


# ---------------------------------------------------------------- Arm26 construction
def fig_arm26():
    # top row: the three model renderings, cropped around the skeleton to the same 0.9 aspect ratio so that
    # they are shown at the same size; bottom row: the screening chart
    fig = plt.figure(figsize=(FULL, 140 * MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.25)
    top = outer[0].subgridspec(1, 3, wspace=0.12)
    A = os.path.join(ROOT, "demo", "assets")
    for k, (f, box, t) in enumerate([("arm26_base.png", (0, 60, 332, 429), "original Arm26"),
                                     ("arm26_full.png", (100, 0, 551, 501), "2 kg load, 11 candidate paths"),
                                     ("arm26_filtered.png", (160, 0, 625, 517), "retained four flexors")]):
        ax = fig.add_subplot(top[0, k]); image(ax, os.path.join(A, f), box)
        ax.set_title("(%s) %s" % ("abc"[k], t), fontsize=7, pad=3, loc="left")
    ax = fig.add_subplot(outer[1].subgridspec(1, 3, width_ratios=[0.18, 1, 0.18])[0, 1])
    paths = ["BRA", "BIClong", "BICshort", "BRD", "TRIlong", "TRIlat", "TRImed", "DELT ant.", "DELT post.", "PECM", "LAT"]
    share = [41.7, 39.9, 11.5, 6.9, 1, 1, 1, 0, 0, 0, 0]
    kept = [True] * 4 + [False] * 7
    y = np.arange(len(paths))[::-1]
    ax.barh(y, share, color=[OI["blue"] if k else "white" for k in kept], edgecolor=[OI["blue"] if k else OI["grey"] for k in kept],
            linewidth=0.6, height=0.7)
    for yy, s, k in zip(y, share, kept):
        ax.text(s + 0.8, yy, ("%.1f" % s) if k else ("~1" if s else "0"), va="center", fontsize=6.5,
                color="black" if k else OI["grey"])
    ax.set_yticks(y); ax.set_yticklabels(paths, fontsize=6.5)
    ax.set_xlabel("Share of the static-optimisation load (%)"); ax.set_xlim(0, 50)
    ax.axhline(y[3] - 0.5, color=OI["grey"], lw=0.5, ls=":")
    ax.text(48, y[2], "retained", ha="right", fontsize=6.5, color=OI["blue"])
    ax.text(48, y[8], "removed", ha="right", fontsize=6.5, color=OI["grey"])
    ax.set_title("(d) static-optimisation screening of the eleven paths", fontsize=7, pad=3, loc="left", x=-0.16)
    save(fig, "fig_arm26")


# ---------------------------------------------------------------- 3CC schematic + sensitivity
def fig_3cc():
    fig = plt.figure(figsize=(FULL, 56 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_axis_off()
    boxes = {"R": (0.1, 3.5, "Resting, $M_R$\n(available)"), "A": (6.7, 3.5, "Active, $M_A$\n(producing force)"),
             "F": (3.4, 0.45, "Fatigued, $M_F$\n(lost capacity)")}
    for k, (x, y, t) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), 3.2, 1.45, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="white", ec="black", lw=0.7))
        ax.text(x + 1.6, y + 0.72, t, ha="center", va="center", fontsize=7)
    arrows = [((3.35, 4.22), (6.65, 4.22), "recruitment\n(activation from SO)", (5.0, 4.85)),
              ((8.3, 3.45), (6.65, 1.95), "fatigue, $F$", (8.75, 2.55)),
              ((3.35, 1.2), (1.7, 3.45), "recovery, $R$", (1.25, 2.3))]
    for a, b, t, p in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.7, color="black"))
        ax.text(*p, t, ha="center", va="center", fontsize=6.5)
    ax.text(5.0, 5.75, r"$\frac{dM_F}{dt}=F\,M_A-R\,M_F,\quad \mathrm{cap}=1-M_F$", ha="center", fontsize=7.5)
    ax.text(5.0, -0.05, r"$F=0.00912\ \mathrm{s^{-1}},\ R=0.00094\ \mathrm{s^{-1}},\ \Delta t=0.01$ s; coupled to static optimisation",
            ha="center", fontsize=6.5)
    label(ax, "(a)", x=0.0, y=0.98)
    ax = fig.add_subplot(gs[0, 1])
    cond = ["0 kg\n10 rep.", "2 kg\n10 rep.", "2 kg\n30 rep.", "4 kg\n30 rep."]
    val = [2.7, 5.8, 17.3, 28.5]
    ax.bar(range(4), val, color=OI["verm"], width=0.62, edgecolor="black", lw=0.5)
    for i, v in enumerate(val):
        ax.text(i, v + 0.6, "%.1f" % v, ha="center", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(cond, fontsize=6.8)
    ax.set_ylabel("End-of-bout fatigued fraction,\nBIClong (%)"); ax.set_ylim(0, 32)
    label(ax, "(b)", x=-0.2)
    save(fig, "fig_3cc")


# ---------------------------------------------------------------- the 13 simulated signals (s04)
def fig_signals():
    O = json.load(open(OSIM_JSON, encoding="utf-8"))["OSIM"]
    t = np.array(O["time"])
    fig, axs = plt.subplots(2, 2, figsize=(FULL, 98 * MM), sharex=True)
    plt.subplots_adjust(hspace=0.32, wspace=0.22)
    ax = axs[0, 0]; ax.plot(t, O["torque"], color="black", lw=0.9)
    ax.set_ylabel("Elbow moment (N·m)"); ax.axhline(0, color=OI["grey"], lw=0.4)
    spec = [(axs[0, 1], "act", "Activation (0–1)"), (axs[1, 0], "frc", "Muscle force (N)"), (axs[1, 1], "MF", "Fatigued fraction (%)")]
    for ax_, key, yl in spec:
        for m, c, _ in MUSCLES:
            ax_.plot(t, O[key][m], color=c, lw=0.9)
        ax_.set_ylabel(yl); ax_.set_ylim(bottom=0)
    for a in axs[1]:
        a.set_xlabel("Time (s)")
    for k, a in enumerate(axs.flat):
        label(a, "(%s)" % "abcd"[k], x=-0.11)
    handles = [plt.Line2D([], [], color=c, lw=1.2) for _, c, _ in MUSCLES]
    fig.legend(handles, [n for _, _, n in MUSCLES], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_signals")


# ---------------------------------------------------------------- per-subject validation, both routes
def fig_validation(height_mm=120, name="fig_validation"):
    # 120 mm fills the validation page of main and mainmedium; the short version (92 mm) suits mainsmall
    single = dict(px=[7.3, 3.9, 9.2, 9.8, 12.7, 8.8, 5.0, 4.8], dir=[6.1, 6.2, 5.3, 7.0, 7.9, 7.8, 6.2, 6.0],
                  mae=[4.92, 6.67, 4.38, 8.64, 10.09, 8.48, 4.26, 8.04], sh=[3.16, 3.92, 5.06, 2.56, 6.50, 3.67, 5.04, 1.46],
                  hum=[0.95, 0.94, 0.93, 0.93, 0.94, 0.94, 0.97, 0.95], fa=[1.02, 1.03, 1.01, 1.00, 1.00, 1.01, 1.05, 1.02])
    multi = dict(px=[8.0, 5.0, 8.6, 10.5, 13.5, 9.5, 6.2, 5.2], dir=[5.46, 4.29, 6.46, 5.90, 7.13, 5.47, 5.70, 3.45],
                 mae=[4.52, 4.56, 4.48, 3.70, 4.04, 5.31, 3.79, 3.42], sh=[1.05, 1.00, 1.14, 0.88, 1.39, 2.01, 1.03, 0.78],
                 hum=[0.986, 0.985, 0.927, 0.891, 0.805, 0.892, 1.061, 0.928], fa=[1.034, 1.026, 0.943, 0.893, 0.836, 0.912, 1.077, 0.954])
    panels = [("px", "2D reprojection error (px)", 15, (0, 17)), ("dir", "Segment-direction error (°)", 8, (0, 9.5)),
              ("mae", "Elbow MAE, offset-harmonised (°)", 10, (0, 11.5)), ("sh", "Shoulder MAE (°)", 8, (0, 9.5)),
              ("hum", "Humerus scale factor", (0.85, 1.15), (0.75, 1.2)), ("fa", "Forearm scale factor", (0.85, 1.15), (0.75, 1.2))]
    fig, axs = plt.subplots(2, 3, figsize=(FULL, height_mm * MM))
    plt.subplots_adjust(hspace=0.42, wspace=0.3)
    x = np.arange(8); w = 0.38
    for k, (key, yl, lim, ylim) in enumerate(panels):
        ax = axs.flat[k]
        for j, (route, d) in enumerate([("multi", multi), ("single", single)]):
            c, _, h = ROUTE[route]
            base = ylim[0] if key in ("hum", "fa") else 0
            ax.bar(x + (j - 0.5) * w, np.array(d[key]) - base, w, bottom=base, color=c if not h else "white",
                   edgecolor=c, hatch=h, lw=0.6)
        if isinstance(lim, tuple):
            ax.axhspan(*lim, color=OI["green"], alpha=0.08, lw=0)
            for l in lim:
                ax.axhline(l, color="black", ls="--", lw=0.6)
        else:
            ax.axhline(lim, color="black", ls="--", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(SUBJ, fontsize=6.5); ax.set_ylim(*ylim); ax.set_ylabel(yl)
        label(ax, "(%s)" % "abcdef"[k], x=-0.2)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=ROUTE["multi"][0], ec=ROUTE["multi"][0]),
               plt.Rectangle((0, 0), 1, 1, fc="white", ec=ROUTE["single"][0], hatch="///"),
               plt.Line2D([], [], color="black", ls="--", lw=0.6)]
    fig.legend(handles, [ROUTE["multi"][1], ROUTE["single"][1], "acceptance limit"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.01))
    save(fig, name)


# ---------------------------------------------------------------- Blender visual check
def fig_blender():
    N = os.path.join(ROOT, "report (1)", "Figures", "nb")
    fig = plt.figure(figsize=(FULL, 68 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1035 / 537, 552 / 630], wspace=0.08)
    for k, f in enumerate(["blender_scene.png", "blender_cam.png"]):
        ax = fig.add_subplot(gs[0, k]); image(ax, os.path.join(N, f)); label(ax, "(%s)" % "ab"[k], x=0.0, y=1.01)
    save(fig, "fig_blender")


# ---------------------------------------------------------------- surrogate R2 by group, both routes
def fig_r2():
    groups = ["Moment", "Activation", "Force", "Fatigue"]
    multi = {"joint angles": [0.932, 0.961, 0.964, 0.936], "3D joints": [0.835, 0.903, 0.895, 0.897],
             "2D keypoints": [0.775, 0.845, 0.856, 0.857]}
    single = {"joint angles": [0.958, 0.986, 0.989, 0.984], "3D joints": [0.919, 0.948, 0.952, 0.975],
              "2D keypoints": [0.896, 0.931, 0.936, 0.956]}
    means = {"multi": [0.952, 0.893, 0.847], "single": [0.984, 0.956, 0.938]}
    names = {"multi": ["A", "B", "C"], "single": ["L1", "L2", "L3"]}
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 60 * MM), sharey=True)
    plt.subplots_adjust(wspace=0.08)
    x = np.arange(4); w = 0.26
    for ax, (key, data, title) in zip(axs, [("multi", multi, "Four cameras, LightGBM (LOSO, 8 subjects)"),
                                             ("single", single, "One camera, BiGRU (grouped 5-fold, 29 groups)")]):
        for j, (lv, c, h) in enumerate(LEVEL):
            ax.bar(x + (j - 1) * w, data[lv], w, color=c if not h else "white", edgecolor=c, hatch=h, lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0.6, 1.0)
        ax.set_title("(%s) %s\nmean $R^2$: %s" % ("a" if key == "multi" else "b", title,
                     ", ".join("%s %.3f" % (n, m) for n, m in zip(names[key], means[key]))), pad=3, loc="left", fontsize=7)
    axs[0].set_ylabel("Held-out $R^2$ against the simulated signals")
    hl = [plt.Rectangle((0, 0), 1, 1, fc=c if not h else "white", ec=c, hatch=h, lw=0.6) for _, c, h in LEVEL]
    fig.legend(hl, ["joint angles (A, L1)", "3D joints (B, L2)", "2D keypoints (C, L3)"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.07))
    save(fig, "fig_r2")


# ---------------------------------------------------------------- cost and accuracy-time trade-off
def fig_cost():
    fig = plt.figure(figsize=(FULL, 128 * MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.62)
    gs = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.32)
    # (a) end-to-end time of the classical chain and of the three surrogates, both routes, stacked by stage
    ax = fig.add_subplot(outer[0])
    stages = [("2D pose (RTMPose, four views / ViTPose)", OI["grey"]), ("3D reconstruction (triangulation and filter / WHAM)", OI["purple"]),
              ("calibration file, alignment, angles, scaling", OI["sky"]), ("OpenSim ID, SO, 3CC", OI["verm"]),
              ("learned surrogate", OI["orange"])]
    routes = [("Four cameras, classical", [103, 40, 1.09, 88.66, 0], "232.7 s"),
              ("Approach A (LightGBM)", [103, 40, 1.09, 0, 0.36], "144.5 s"),
              ("Approach B (LightGBM)", [103, 40, 1.0, 0, 0.10], "144.1 s"),
              ("Approach C (LightGBM)", [103, 0, 1.0, 0, 0.17], "104.2 s"),
              ("One camera, classical", [195, 260, 3.2, 26.7, 0], "485 s"),
              ("Level 1 (BiGRU)", [195, 260, 3.2, 0, 0.016], "458 s"),
              ("Level 2 (BiGRU)", [195, 260, 0, 0, 0.016], "455 s"),
              ("Level 3 (BiGRU, deployed)", [195, 0, 0, 0, 0.016], "195 s")]
    ys = [8.4, 7.4, 6.4, 5.4, 3.6, 2.6, 1.6, 0.6]
    for y, (n, seg, tot) in zip(ys, routes):
        left = 0
        for k, v in enumerate(seg):
            if v:
                ax.barh(y, max(v, 2 if k == 4 else v), left=left, height=0.72, color=stages[k][1], edgecolor="black", lw=0.4)
                left += v
        ax.text(left + 6, y, tot, va="center", fontsize=6.8)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in routes]); ax.set_xlim(0, 540); ax.set_ylim(0, 9.0)
    ax.axhline(4.5, color="#bdbdbd", lw=0.5, ls="--")
    ax.set_xlabel("End-to-end time per trial (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.4) for _, c in stages]
    ax.legend(handles, [s[0] for s in stages], loc="lower center", ncol=3, fontsize=6.3, handlelength=1.1,
              bbox_to_anchor=(0.42, 1.01), columnspacing=1.0)
    label(ax, "(a)", x=-0.2, y=1.1)
    # (b) biomechanics stage only, log scale
    ax = fig.add_subplot(gs[0, 0])
    bars = [("OpenSim", 26.7, OI["verm"]), ("BiGRU", 0.0159, OI["orange"]), ("OpenSim", 88.66, OI["verm"]),
            ("LGBM A", 0.36, OI["orange"]), ("LGBM B", 0.10, OI["orange"]), ("LGBM C", 0.17, OI["orange"])]
    xs = [0, 1, 2.6, 3.6, 4.6, 5.6]
    for xx, (n, v, c) in zip(xs, bars):
        ax.bar(xx, v, 0.7, color=c, edgecolor="black", lw=0.4)
        ax.text(xx, v * 1.35, ("%.1f ms" % (v * 1000)) if v < 0.1 else ("%.2f s" % v if v < 1 else "%.1f s" % v),
                ha="center", fontsize=6.2)
    ax.set_yscale("log"); ax.set_ylim(0.005, 400)
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("Biomechanics-stage time (s)")
    ax.text(0.5, 260, "one camera", ha="center", fontsize=6.5, style="italic")
    ax.text(4.1, 260, "four cameras", ha="center", fontsize=6.5, style="italic")
    ax.set_ylim(0.005, 900); ax.set_xlim(-0.8, 6.4)
    label(ax, "(b)", x=-0.22)
    # (c) accuracy against end-to-end time, all six surrogates
    ax = fig.add_subplot(gs[0, 1])
    pts = [("A", 144.5, 0.952, "multi"), ("B", 144.1, 0.893, "multi"), ("C", 104.2, 0.847, "multi"),
           ("L1", 458, 0.984, "single"), ("L2", 455, 0.956, "single"), ("L3", 195, 0.938, "single")]
    off = {"A": (8, 0.002), "B": (8, -0.004), "C": (8, -0.004), "L1": (-12, 0.006), "L2": (-12, -0.012), "L3": (8, 0.004)}
    for n, tt, r2, rt in pts:
        c = ROUTE[rt][0]
        ax.scatter(tt, r2, s=26, marker="o" if rt == "multi" else "s", color=c if rt == "multi" else "white", edgecolor=c,
                   lw=0.9, zorder=3)
        ax.text(tt + off[n][0], r2 + off[n][1], n + (" (deployed)" if n == "L3" else ""), fontsize=6.5,
                ha="right" if off[n][0] < 0 else "left")
    ax.axvline(485, color=ROUTE["single"][0], ls=":", lw=0.7); ax.axvline(233, color=ROUTE["multi"][0], ls=":", lw=0.7)
    ax.text(476, 0.83, "classical, one camera", rotation=90, ha="right", va="bottom", fontsize=6, color=ROUTE["single"][0])
    ax.text(224, 0.83, "classical, four cameras", rotation=90, ha="right", va="bottom", fontsize=6, color=ROUTE["multi"][0])
    ax.set_xlim(60, 520); ax.set_ylim(0.825, 1.0)
    ax.set_xlabel("End-to-end time per trial (s)"); ax.set_ylabel("Mean held-out $R^2$")
    h = [plt.Line2D([], [], marker="o", ls="", color=ROUTE["multi"][0], ms=4),
         plt.Line2D([], [], marker="s", ls="", mfc="white", mec=ROUTE["single"][0], ms=4)]
    ax.legend(h, ["four cameras", "one camera"], loc="upper left", fontsize=6.3, handletextpad=0.3)
    label(ax, "(c)", x=-0.2)
    save(fig, "fig_cost")


# ---------------------------------------------------------------- application screenshots
def fig_app():
    A = os.path.join(ROOT, "figures", "app")
    shots = [("patient_home.png", None, "Patient: prescription"), ("patient_record.png", (0, 640), "Patient: recording"),
             ("patient_results.png", None, "Patient: results"), ("clinician_review.png", None, "Clinician: session review")]
    asp = []
    for f, crop, _ in shots:
        im = plt.imread(os.path.join(A, f))
        h = (crop[1] - crop[0]) if crop else im.shape[0]
        asp.append(im.shape[1] / h)
    fig = plt.figure(figsize=(FULL, 78 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=asp, wspace=0.14)
    for k, (f, crop, t) in enumerate(shots):
        ax = fig.add_subplot(gs[0, k]); image(ax, os.path.join(A, f), crop)
        for s in ax.spines.values():
            s.set_visible(True)
        ax.set_title("(%s) %s" % ("abcd"[k], t), fontsize=6.8, pad=3, loc="left")
    save(fig, "fig_app")


fig_validation(92, "fig_validation_short")
for f in (fig_arm26, fig_3cc, fig_signals, fig_validation, fig_blender, fig_r2, fig_cost, fig_app):
    f()
