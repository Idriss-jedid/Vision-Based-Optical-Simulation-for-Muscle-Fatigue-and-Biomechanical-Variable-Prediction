# pyright: reportMissingImports=false
"""Génère toutes les figures du rapport -> batch/report_figs/*.png. biomech env."""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch"); FIG = os.path.join(B, "report_figs"); os.makedirs(FIG, exist_ok=True)
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]


def bar(csv, col, title, fname, color="#1f77b4"):
    d = pd.read_csv(csv, index_col=0).sort_values(col)
    plt.figure(figsize=(8, 4.2)); plt.barh(d.index, d[col], color=color)
    for i, v in enumerate(d[col]): plt.text(v + 0.003, i, "%.3f" % v, va="center", fontsize=9)
    plt.xlabel("R² moyen (LOSO)"); plt.title(title); plt.xlim(0, 1.02); plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname), dpi=130); plt.close()


def fe_gain():
    d = pd.read_csv(os.path.join(B, "ml_advanced.csv"), index_col=0)
    rows = ["LGBM_base", "LGBM_eng", "LGBM_eng_optuna"]; labels = ["BASE (11)", "ENG (22)", "ENG+Optuna"]
    grp = ["torque", "activations", "forces", "fatigue"]
    x = np.arange(len(grp)); w = 0.25
    plt.figure(figsize=(9, 4.6))
    for i, (r, lab) in enumerate(zip(rows, labels)):
        plt.bar(x + (i - 1) * w, [d.loc[r, g] for g in grp], w, label=lab)
    plt.xticks(x, grp); plt.ylabel("R² (LOSO)"); plt.ylim(0.7, 1.0)
    plt.title("Effet du feature engineering + Optuna (LightGBM, 8 sujets)"); plt.legend()
    plt.grid(axis="y", alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_fe_gain.png"), dpi=130); plt.close()


def ts_compare():
    u = pd.read_csv(os.path.join(B, "ts_fatigue.csv"), index_col=0)["mean"]
    t = pd.read_csv(os.path.join(B, "ts_fatigue_tuned.csv"), index_col=0)["mean"]
    # référence LightGBM = MODÈLE FINAL trial #11 (moyenne des R² des 4 MF, depuis metrics_final)
    mfin = pd.read_csv(os.path.join(B, "metrics_final.csv"), index_col=0)
    lgbm_final = mfin.loc[[i for i in mfin.index if i.startswith("MF_")], "R2"].mean()
    data = {"LightGBM\n(trial #11)": lgbm_final, "TST": u.get("TST", np.nan),
            "LSTM": u.get("LSTM", np.nan), "PatchTST": u.get("PatchTST", np.nan),
            "LSTM\ntuned": t.get("LSTM_tuned", np.nan), "PatchTST\ntuned": t.get("PatchTST_tuned", np.nan)}
    ks = list(data); vs = [data[k] for k in ks]
    cols = ["#2ca02c"] + ["#1f77b4"] * 3 + ["#ff7f0e"] * 2
    plt.figure(figsize=(9, 4.4)); plt.bar(ks, vs, color=cols)
    for i, v in enumerate(vs): plt.text(i, v + 0.005, "%.3f" % v, ha="center", fontsize=9)
    plt.ylabel("R² fatigue (LOSO)"); plt.ylim(0.7, 0.97)
    plt.title("FATIGUE : feature-based (LightGBM) vs séquentiel (deep)"); plt.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_ts_fatigue.png"), dpi=130); plt.close()


def xai_heat():
    d = pd.read_csv(os.path.join(B, "xai_importance.csv"), index_col=0)
    d = d.sort_values("overall", ascending=False).head(12)[["torque", "activations", "forces", "fatigue"]]
    plt.figure(figsize=(7.5, 6)); im = plt.imshow(d.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="SHAP mean|valeur|"); plt.yticks(range(len(d)), d.index)
    plt.xticks(range(4), ["torque", "activations", "forces", "fatigue"])
    for i in range(len(d)):
        for j in range(4): plt.text(j, i, "%.2f" % d.values[i, j], ha="center", va="center",
                                     color="white" if d.values[i, j] < d.values.max() * .6 else "black", fontsize=8)
    plt.title("XAI — importance SHAP par feature et par cible"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_xai_heatmap.png"), dpi=130); plt.close()


def pred_timeseries(held="s11"):
    df = pd.read_csv(os.path.join(B, "ml_dataset_A.csv")); g = np.pi / 180
    df["sin_qel"] = np.sin(df.q_el * g); df["cos_qel"] = np.cos(df.q_el * g)
    df["sin_qsh"] = np.sin(df.q_sh * g); df["cos_qsh"] = np.cos(df.q_sh * g)
    df["abs_qd_el"] = df.qd_el.abs(); df["abs_qdd_el"] = df.qdd_el.abs(); df["qd_el2"] = df.qd_el ** 2
    df["grav_load"] = (df.forearm_mass + 2.0) * df.forearm_len * np.sin((df.q_sh + df.q_el) * g)
    df["qel_x_fmass"] = df.q_el * df.forearm_mass
    cp, cg = [], []
    for s, sub in df.groupby("subj", sort=False):
        dt = np.median(np.diff(sub.time.values)); cp.append(np.cumsum(sub.abs_qd_el.values) * dt)
        cg.append(np.cumsum(np.abs(sub.grav_load.values)) * dt)
    df["cum_path_el"] = np.concatenate(cp); df["cum_grav_imp"] = np.concatenate(cg)
    ENG = ["q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el", "time", "humerus_mass", "forearm_mass",
           "humerus_len", "forearm_len", "sin_qel", "cos_qel", "sin_qsh", "cos_qsh", "abs_qd_el",
           "abs_qdd_el", "qd_el2", "grav_load", "qel_x_fmass", "cum_path_el", "cum_grav_imp"]
    tr, te = df[df.subj != held], df[df.subj == held]
    xs = StandardScaler().fit(tr[ENG]); ys = StandardScaler().fit(tr[TARGETS])
    m = MultiOutputRegressor(LGBMRegressor(n_estimators=975, num_leaves=15, learning_rate=0.196,
                                           subsample=0.605, colsample_bytree=0.911, min_child_samples=24,
                                           reg_lambda=8.19, n_jobs=-1, random_state=0, verbose=-1))
    m.fit(xs.transform(tr[ENG]), ys.transform(tr[TARGETS]))
    pred = ys.inverse_transform(m.predict(xs.transform(te[ENG])))
    t = te.time.values
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col, lab in [(ax[0], "elbow_moment", "Torque coude (N·m)"),
                        (ax[1], "act_BIClong", "Activation BIClong"),
                        (ax[2], "MF_BIClong", "Fatigue MF BIClong (%)")]:
        j = TARGETS.index(col)
        a.plot(t, te[col].values, "g-", lw=2, label="OpenSim (vérité)")
        a.plot(t, pred[:, j], "b--", lw=1.3, label="LightGBM (prédit)")
        a.set_xlabel("temps (s)"); a.set_title(lab + " — sujet %s (jamais vu)" % held); a.legend(); a.grid(alpha=.3)
    fig.suptitle("Prédiction vs OpenSim sur un sujet exclu (LOSO) — LightGBM + features", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(os.path.join(FIG, "fig_pred_timeseries.png"), dpi=130); plt.close()


bar(os.path.join(B, "bench_tabular.csv"), "mean", "Benchmark — tabular models (8 subjects, LOSO)", "fig_tabular.png", "#1f77b4")
bar(os.path.join(B, "bench_deep.csv"), "mean", "Benchmark — deep learning (8 subjects, LOSO)", "fig_deep.png", "#9467bd")
fe_gain(); ts_compare(); xai_heat(); pred_timeseries()
print("figures écrites dans", FIG)
print(os.listdir(FIG))
