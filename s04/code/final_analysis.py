# pyright: reportMissingImports=false
"""
Analyse finale du modèle retenu (LightGBM + features + Optuna), 8 sujets, LOSO :
  - métriques MULTIPLES par cible (R2, RMSE, MAE, Pearson r, NRMSE)
  - métriques par sujet
  - figures : séries temporelles prédit vs réel (plusieurs sujets/cibles), parité, barres
  - Optuna : ré-exécution avec sauvegarde des essais (top-K) + history + importance des params
Sorties -> batch/report_figs/*.png, batch/metrics_final.csv, batch/metrics_per_subject.csv,
           batch/optuna_trials.csv. biomech env.
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch"); FIG = os.path.join(B, "report_figs"); os.makedirs(FIG, exist_ok=True)
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]
UNITS = {"elbow_moment": "N·m", **{"act_" + m: "" for m in FLEX}, **{"frc_" + m: "N" for m in FLEX},
         **{"MF_" + m: "%" for m in FLEX}}
GROUPS = {"torque": ["elbow_moment"], "activations": ["act_" + m for m in FLEX],
          "forces": ["frc_" + m for m in FLEX], "fatigue": ["MF_" + m for m in FLEX]}
BEST = dict(n_estimators=975, num_leaves=15, learning_rate=0.196, subsample=0.605,
            colsample_bytree=0.911, min_child_samples=24, reg_lambda=8.19)


def engineer(df):
    g = np.pi / 180
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
    return df, ["q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el", "time", "humerus_mass",
                "forearm_mass", "humerus_len", "forearm_len", "sin_qel", "cos_qel", "sin_qsh",
                "cos_qsh", "abs_qd_el", "abs_qdd_el", "qd_el2", "grav_load", "qel_x_fmass",
                "cum_path_el", "cum_grav_imp"]


df, ENG = engineer(pd.read_csv(os.path.join(B, "ml_dataset_A.csv")))
SUBS = sorted(df.subj.unique())


def model(): return MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **BEST))


def loso_predict():
    """retourne dict subj -> (df_sujet, prédictions array)."""
    out = {}
    for held in SUBS:
        tr, te = df[df.subj != held], df[df.subj == held]
        xs = StandardScaler().fit(tr[ENG]); ys = StandardScaler().fit(tr[TARGETS])
        m = model(); m.fit(xs.transform(tr[ENG]), ys.transform(tr[TARGETS]))
        out[held] = (te.reset_index(drop=True), ys.inverse_transform(m.predict(xs.transform(te[ENG]))))
    return out


def pearson(a, b): return float(np.corrcoef(a, b)[0, 1])


def metrics():
    pr = loso_predict()
    # concat global par cible
    rows = []
    persubj = []
    for j, t in enumerate(TARGETS):
        yt = np.concatenate([pr[s][0][t].values for s in SUBS])
        yp = np.concatenate([pr[s][1][:, j] for s in SUBS])
        rng = yt.max() - yt.min()
        rows.append(dict(target=t, unit=UNITS[t], R2=r2_score(yt, yp),
                         RMSE=np.sqrt(mean_squared_error(yt, yp)), MAE=mean_absolute_error(yt, yp),
                         Pearson_r=pearson(yt, yp), NRMSE_pct=100 * np.sqrt(mean_squared_error(yt, yp)) / rng))
    md = pd.DataFrame(rows).set_index("target")
    md.round(4).to_csv(os.path.join(B, "metrics_final.csv"))
    print("=== Métriques finales par cible (LOSO) ===\n", md.round(3).to_string())
    # par sujet (R2 moyen sur les 13 cibles)
    def grp_r2(te, p, pref):
        return np.mean([r2_score(te[pref + m].values, p[:, TARGETS.index(pref + m)]) for m in FLEX])
    for s in SUBS:
        te, p = pr[s]
        r2s = [r2_score(te[t].values, p[:, j]) for j, t in enumerate(TARGETS)]
        persubj.append(dict(subj=s, R2_mean=np.mean(r2s),
                            R2_torque=r2_score(te["elbow_moment"].values, p[:, 0]),
                            R2_activations=grp_r2(te, p, "act_"),
                            R2_forces=grp_r2(te, p, "frc_"),
                            R2_fatigue=grp_r2(te, p, "MF_")))
    ps = pd.DataFrame(persubj).set_index("subj"); ps.round(4).to_csv(os.path.join(B, "metrics_per_subject.csv"))
    print("\n=== R² par sujet (held-out) ===\n", ps.round(3).to_string())
    return pr, md, ps


def fig_muscles(pr, subj):
    """4 muscles x 3 quantities (activation / force / fatigue), predicted vs ground truth over time."""
    te, p = pr[subj]; t = te.time.values
    cats = [("act_", "activation"), ("frc_", "force (N)"), ("MF_", "fatigue MF (%)")]
    fig, ax = plt.subplots(len(FLEX), 3, figsize=(15, 12))
    for i, mus in enumerate(FLEX):
        for jc, (pref, lab) in enumerate(cats):
            a = ax[i, jc]; col = pref + mus; j = TARGETS.index(col)
            a.plot(t, te[col].values, "g-", lw=1.8, label="OpenSim (truth)")
            a.plot(t, p[:, j], "b--", lw=1.2, label="predicted")
            a.set_title("%s - %s  (R2=%.3f)" % (mus, lab, r2_score(te[col].values, p[:, j])), fontsize=10)
            if i == len(FLEX) - 1: a.set_xlabel("time (s)")
            if i == 0 and jc == 0: a.legend(fontsize=8)
            a.grid(alpha=.3)
    fig.suptitle("Per-muscle predicted vs ground truth over time - subject %s (held-out, LOSO)" % subj,
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .98]); fig.savefig(os.path.join(FIG, "fig_muscles_%s.png" % subj), dpi=115); plt.close()


def fig_torque(pr, subjs):
    fig, ax = plt.subplots(1, len(subjs), figsize=(5 * len(subjs), 4))
    for a, s in zip(np.ravel(ax), subjs):
        te, p = pr[s]; t = te.time.values; j = TARGETS.index("elbow_moment")
        a.plot(t, te["elbow_moment"].values, "g-", lw=1.8, label="OpenSim (truth)")
        a.plot(t, p[:, j], "b--", lw=1.2, label="predicted")
        a.set_title("Joint torque - %s (R2=%.3f)" % (s, r2_score(te["elbow_moment"].values, p[:, j])))
        a.set_xlabel("time (s)"); a.set_ylabel("N.m"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle("Joint torque: predicted vs ground truth (held-out subjects)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FIG, "fig_torque.png"), dpi=130); plt.close()


def fig_parity(pr):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    reps = [("elbow_moment", "Torque (N·m)"), ("act_BIClong", "Activation BIClong"),
            ("frc_BRA", "Force BRA (N)"), ("MF_BIClong", "Fatigue MF BIClong (%)")]
    for a, (col, lab) in zip(ax, reps):
        j = TARGETS.index(col)
        yt = np.concatenate([pr[s][0][col].values for s in SUBS]); yp = np.concatenate([pr[s][1][:, j] for s in SUBS])
        a.scatter(yt, yp, s=3, alpha=.25, c="#1f77b4")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_xlabel("OpenSim (true)"); a.set_ylabel("predicted"); a.set_title("%s\nR²=%.3f" % (lab, r2_score(yt, yp)))
        a.grid(alpha=.3)
    fig.suptitle("Parity plots — predicted vs ground truth (all held-out frames)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(os.path.join(FIG, "fig_parity.png"), dpi=130); plt.close()


def fig_per_subject(ps):
    plt.figure(figsize=(11, 4.6))
    x = np.arange(len(ps)); w = 0.16
    cols = ["R2_mean", "R2_torque", "R2_activations", "R2_forces", "R2_fatigue"]
    labs = ["mean", "torque", "activations", "forces", "fatigue"]
    for i, (c, l) in enumerate(zip(cols, labs)):
        plt.bar(x + (i - 2) * w, ps[c], w, label=l)
    plt.xticks(x, ps.index); plt.ylabel("R2 (held-out)"); plt.ylim(0, 1.0)
    plt.title("Per-subject generalization (LOSO) - all target groups")
    plt.legend(ncol=5, fontsize=8); plt.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_per_subject.png"), dpi=130); plt.close()


def fig_metric_bars(md):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    md2 = md.copy()
    ax[0].barh(md2.index, md2.R2, color="#2ca02c"); ax[0].set_xlabel("R²"); ax[0].set_xlim(0, 1)
    ax[0].set_title("R² per target"); ax[0].grid(axis="x", alpha=.3)
    ax[1].barh(md2.index, md2.NRMSE_pct, color="#d62728"); ax[1].set_xlabel("NRMSE (%)")
    ax[1].set_title("Normalized RMSE per target"); ax[1].grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_metrics_bars.png"), dpi=130); plt.close()


def optuna_topk():
    print("\n=== Optuna (ré-exécution, sauvegarde des essais) ===")
    def loso_mean(kw):
        accs = []
        for held in SUBS:
            tr, te = df[df.subj != held], df[df.subj == held]
            xs = StandardScaler().fit(tr[ENG]); ys = StandardScaler().fit(tr[TARGETS])
            m = MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **kw))
            m.fit(xs.transform(tr[ENG]), ys.transform(tr[TARGETS]))
            p = ys.inverse_transform(m.predict(xs.transform(te[ENG])))
            r2 = [r2_score(te[t].values, p[:, j]) for j, t in enumerate(TARGETS)]
            fat = np.mean([r2[TARGETS.index("MF_" + m)] for m in FLEX])
            accs.append(0.5 * fat + 0.5 * np.mean(r2))
        return float(np.mean(accs))

    def obj(tr):
        kw = dict(n_estimators=tr.suggest_int("n_estimators", 300, 1000),
                  num_leaves=tr.suggest_int("num_leaves", 15, 90),
                  learning_rate=tr.suggest_float("learning_rate", 0.01, 0.2, log=True),
                  subsample=tr.suggest_float("subsample", 0.6, 1.0),
                  colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0),
                  min_child_samples=tr.suggest_int("min_child_samples", 5, 60),
                  reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10, log=True))
        return loso_mean(kw)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                             pruner=optuna.pruners.HyperbandPruner())
    st.optimize(obj, n_trials=40, show_progress_bar=False)
    tdf = st.trials_dataframe().sort_values("value", ascending=False)
    tdf.to_csv(os.path.join(B, "optuna_trials.csv"), index=False)
    cols = ["number", "value"] + [c for c in tdf.columns if c.startswith("params_")]
    print("Top-10 essais Optuna :\n", tdf[cols].head(10).round(4).to_string(index=False))
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        ax = plot_optimization_history(st); ax.figure.set_size_inches(8, 4.2)
        ax.figure.tight_layout(); ax.figure.savefig(os.path.join(FIG, "fig_optuna_history.png"), dpi=130); plt.close()
        ax = plot_param_importances(st); ax.figure.set_size_inches(8, 4.2)
        ax.figure.tight_layout(); ax.figure.savefig(os.path.join(FIG, "fig_optuna_importance.png"), dpi=130); plt.close()
    except Exception as e:
        print("optuna viz skip:", e)
    return tdf


def main():
    pr, md, ps = metrics()
    for s in ["s11", "s04", "s08"]: fig_muscles(pr, s)
    fig_torque(pr, ["s11", "s04", "s08"])
    fig_parity(pr); fig_per_subject(ps); fig_metric_bars(md)
    if not os.path.exists(os.path.join(B, "optuna_trials.csv")):
        optuna_topk()  # déjà calculé -> on saute la ré-exécution lente
    print("\nFIGURES per-muscle + torque + parity + per-subject regénérées")
    print("CSV: metrics_final.csv, metrics_per_subject.csv")


if __name__ == "__main__":
    main()
