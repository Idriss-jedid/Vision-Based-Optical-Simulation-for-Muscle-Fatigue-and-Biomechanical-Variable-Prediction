# pyright: reportMissingImports=false
"""
Courbes explicatives de la motion FIRST (paper_minjerk_fatigue_10cycle_first.mot),
à partir des sorties OpenSim (ID moment + SO activation/force).
Montre la dégradation par la fatigue : ROM qui rétrécit, pic par rep qui baisse,
activations musculaires, couple au coude. -> Data/first_fatigue_curves.png
biomech env.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MOTION = os.path.join(HERE, "..", "Data", "paper_minjerk_fatigue_10cycle_first.mot")
RES = os.path.join(HERE, "..", "Results", "Stage2", "first_10rep")
ACT = os.path.join(RES, "first_SO_activation.sto")
FRC = os.path.join(RES, "first_SO_force.sto")
IDF = os.path.join(RES, "ID_genforces.sto")
OUT = os.path.join(HERE, "..", "Data", "first_fatigue_curves.png")
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
COLORS = {"BIClong": "#d62728", "BICshort": "#ff7f0e", "BRA": "#1f77b4", "BRD_hand": "#2ca02c"}


def read_sto(path):
    L = open(path).read().splitlines()
    i = next(k for k, l in enumerate(L) if l.strip().lower() == "endheader")
    cols = L[i + 1].split("\t")
    data = np.array([[float(x) for x in r.split()] for r in L[i + 2:] if r.strip()])
    return cols, data


def read_mot(path):
    L = open(path).read().splitlines()
    i = next(k for k, l in enumerate(L) if l.strip().lower() == "endheader")
    cols = L[i + 1].split("\t")
    data = np.array([[float(x) for x in r.split()] for r in L[i + 2:] if r.strip()])
    return cols, data


def main():
    mc, md = read_mot(MOTION)
    t = md[:, 0]; elbow = md[:, mc.index("r_elbow_flex")]
    ac, ad = read_sto(ACT)
    ic, idd = read_sto(IDF)
    # elbow ID moment column
    elcol = [c for c in ic if "elbow" in c.lower()][0]
    M = idd[:, ic.index(elcol)]

    # per-rep peak angle (10 segments)
    nrep = 10; seg = len(t) // nrep
    peaks = [elbow[k * seg:(k + 1) * seg].max() for k in range(nrep)]
    reps = np.arange(1, nrep + 1)

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    # (a) elbow angle full
    ax[0, 0].plot(t, elbow, color="#1f77b4", lw=1.0)
    ax[0, 0].axhline(120, ls="--", c="gray", lw=1); ax[0, 0].axhline(100, ls="--", c="red", lw=1)
    ax[0, 0].text(t[-1], 120, " frais 120°", va="center", c="gray")
    ax[0, 0].text(t[-1], 100, " fatigué 100°", va="center", c="red")
    ax[0, 0].set_title("(a) Angle du coude — la ROM rétrécit avec la fatigue")
    ax[0, 0].set_xlabel("temps (s)"); ax[0, 0].set_ylabel("flexion coude (°)"); ax[0, 0].grid(alpha=.3)

    # (b) per-rep peak
    ax[0, 1].plot(reps, peaks, "o-", color="#d62728", lw=2, ms=7)
    for r, p in zip(reps, peaks):
        ax[0, 1].text(r, p + 1, "%.0f°" % p, ha="center", fontsize=8)
    ax[0, 1].set_title("(b) Pic d'angle par répétition — décline (fatigue)")
    ax[0, 1].set_xlabel("répétition"); ax[0, 1].set_ylabel("pic flexion (°)")
    ax[0, 1].set_xticks(reps); ax[0, 1].grid(alpha=.3)

    # (c) muscle activations
    for m in FLEX:
        if m in ac:
            ax[1, 0].plot(ad[:, 0], 100 * ad[:, ac.index(m)], label=m, color=COLORS[m], lw=1.1)
    ax[1, 0].set_title("(c) Activation musculaire (SO) — 4 fléchisseurs")
    ax[1, 0].set_xlabel("temps (s)"); ax[1, 0].set_ylabel("activation (%)")
    ax[1, 0].legend(ncol=2, fontsize=8); ax[1, 0].grid(alpha=.3)

    # (d) elbow moment (ID)
    ax[1, 1].plot(t[:len(M)], np.abs(M), color="#9467bd", lw=1.0)
    ax[1, 1].set_title("(d) Couple au coude (Inverse Dynamics) — haltère 2 kg")
    ax[1, 1].set_xlabel("temps (s)"); ax[1, 1].set_ylabel("|couple| (N·m)"); ax[1, 1].grid(alpha=.3)

    fig.suptitle("Motion FIRST (10 reps, 40 s) — dégradation par fatigue (sorties OpenSim)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=140); plt.close()
    print("ROM rep1=%.0f° -> rep10=%.0f° (perte %.0f°)" % (peaks[0], peaks[-1], peaks[0] - peaks[-1]))
    print("couple coude: moy %.2f, pic %.2f N·m" % (np.mean(np.abs(M)), np.max(np.abs(M))))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
