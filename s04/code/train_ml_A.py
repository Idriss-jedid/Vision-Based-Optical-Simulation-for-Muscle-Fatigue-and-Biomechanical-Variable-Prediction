# pyright: reportMissingImports=false
"""
APPROCHE A — ML : prédire la biomécanique depuis la CINÉMATIQUE (vision-derived).
INPUT  : q_sh, q_el, qd_sh, qd_el, qdd_sh, qdd_el, time          (cinématique)
         + humerus_mass, forearm_mass, humerus_len, forearm_len  (anthropométrie/sujet)
         (rep retiré : inutile pour torque/forces/act qui sont per-frame ; pour la fatigue
          MF il faut une feature cumulative -> 'time' est plus propre que 'rep'.
          L'anthropométrie rend le ML subject-aware : même pose + corps différent = torque différent.)
OUTPUT : elbow_moment (torque) | act_<4flex> | frc_<4flex> | MF_<4flex> (fatigue)
Évaluation : LEAVE-ONE-SUBJECT-OUT (train 7 sujets, test sur le sujet exclu) — pas de
fuite temporelle. NN (MLP) vs baseline (Ridge). -> batch/ml_results_A.csv + plot. biomech env.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
DATA = os.path.join(ROOT, "batch", "ml_dataset_A.csv"); OUT = os.path.join(ROOT, "batch")
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
X_COLS = ["q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el", "time",
          # anthropométrie (constante/sujet, du modèle scalé) -> rend le ML subject-aware
          "humerus_mass", "forearm_mass", "humerus_len", "forearm_len"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]


def main():
    df = pd.read_csv(DATA)
    subs = sorted(df["subj"].unique())
    print("dataset A: %d frames, %d sujets %s" % (len(df), len(subs), subs))
    print("INPUT (%d): %s\nOUTPUT (%d): %s\n" % (len(X_COLS), X_COLS, len(TARGETS), TARGETS))

    # LEAVE-ONE-SUBJECT-OUT
    per_target = {t: [] for t in TARGETS}; base_r2 = []
    held_pred = None
    for held in subs:
        tr = df[df.subj != held]; te = df[df.subj == held]
        Xtr, Ytr = tr[X_COLS].values, tr[TARGETS].values
        Xte, Yte = te[X_COLS].values, te[TARGETS].values
        xs = StandardScaler().fit(Xtr); ys = StandardScaler().fit(Ytr)
        nn = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=600, random_state=0, early_stopping=True)
        nn.fit(xs.transform(Xtr), ys.transform(Ytr))
        pred = ys.inverse_transform(nn.predict(xs.transform(Xte)))
        rg = Ridge(alpha=1.0).fit(xs.transform(Xtr), ys.transform(Ytr))
        predr = ys.inverse_transform(rg.predict(xs.transform(Xte)))
        for j, t in enumerate(TARGETS):
            per_target[t].append((np.sqrt(mean_squared_error(Yte[:, j], pred[:, j])),
                                  mean_absolute_error(Yte[:, j], pred[:, j]),
                                  r2_score(Yte[:, j], pred[:, j])))
        base_r2.append(np.mean([r2_score(Yte[:, j], predr[:, j]) for j in range(len(TARGETS))]))
        if held == subs[-1]: held_pred = (te, pred)

    print("%-16s %9s %9s %9s" % ("target (LOSO)", "RMSE", "MAE", "R2"))
    rows = []
    for t in TARGETS:
        arr = np.array(per_target[t]); rmse, mae, r2 = arr.mean(0)
        unit = "N·m" if t == "elbow_moment" else ("N" if t.startswith("frc") else ("%MF" if t.startswith("MF") else ""))
        print("%-16s %9.3f %9.3f %9.3f  %s" % (t, rmse, mae, r2, unit))
        rows.append((t, rmse, mae, r2, unit))
    nn_meanr2 = np.mean([np.array(per_target[t]).mean(0)[2] for t in TARGETS])
    print("\nR² moyen NN (LOSO) : %.3f  |  Ridge baseline : %.3f" % (nn_meanr2, np.mean(base_r2)))
    # group means
    grp = {"torque": ["elbow_moment"], "activations": ["act_" + m for m in FLEX],
           "forces": ["frc_" + m for m in FLEX], "fatigue (MF)": ["MF_" + m for m in FLEX]}
    print("\n--- R² moyen par groupe ---")
    for g, ts in grp.items():
        print("  %-14s R²=%.3f" % (g, np.mean([np.array(per_target[t]).mean(0)[2] for t in ts])))

    with open(os.path.join(OUT, "ml_results_A.csv"), "w", newline="\n") as f:
        f.write("target,RMSE,MAE,R2,unit\n")
        for t, rmse, mae, r2, u in rows: f.write("%s,%.4f,%.4f,%.4f,%s\n" % (t, rmse, mae, r2, u))

    # plot held-out subject : torque, biceps act, fatigue
    te, pred = held_pred; t = te["time"].values
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a, col, lab in [(ax[0], "elbow_moment", "torque coude (N·m)"),
                        (ax[1], "act_BIClong", "activation BIClong"),
                        (ax[2], "MF_BIClong", "fatigue MF BIClong (%)")]:
        j = TARGETS.index(col)
        a.plot(t, te[col].values, "g-", lw=2, label="OpenSim (vrai)")
        a.plot(t, pred[:, j], "b--", lw=1.3, label="ML prédit")
        a.set_xlabel("temps (s)"); a.set_title(lab + " — sujet exclu %s" % te["subj"].iloc[0]); a.legend(); a.grid(alpha=.3)
    fig.suptitle("Approche A : cinématique -> biomécanique (sujet jamais vu)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(OUT, "ml_A_pred.png"), dpi=125); plt.close()
    print("wrote ml_results_A.csv + ml_A_pred.png")


if __name__ == "__main__":
    main()
