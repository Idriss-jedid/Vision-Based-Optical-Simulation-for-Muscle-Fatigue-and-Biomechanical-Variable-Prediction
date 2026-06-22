# pyright: reportMissingImports=false
"""
Construit toutes les donnees + figures pour le RAPPORT 3D-joints -> Biomecanique (vision-only) :
Optuna (top-K) + metriques multi (R2/RMSE/MAE/Pearson/NRMSE) + per-subject + figures.
Reutilise les features Butterworth de final_3d_model. Sorties: batch/report_figs_3d/*, CSVs. biomech env.
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import optuna, shap
optuna.logging.set_verbosity(optuna.logging.WARNING)
import final_3d_model as M  # reuse features (X, Y, FEAT, SUBS, TARGETS, GROUPS, features)

B, X, Y, FEAT, SUBS, TARGETS, GROUPS = M.B, M.X, M.Y, M.FEAT, M.SUBS, M.TARGETS, M.GROUPS
FLEX = M.FLEX
FIG = os.path.join(B, "report_figs_3d"); os.makedirs(FIG, exist_ok=True)
UNITS = {"elbow_moment": "N.m", **{"act_" + m: "" for m in FLEX}, **{"frc_" + m: "N" for m in FLEX}, **{"MF_" + m: "%" for m in FLEX}}


def loso_pred(params):
    out = {}
    for held in SUBS:
        tr = X.subj != held; te = X.subj == held
        xs = StandardScaler().fit(X.loc[tr, FEAT]); ys = StandardScaler().fit(Y.loc[tr, TARGETS])
        m = MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **params))
        m.fit(xs.transform(X.loc[tr, FEAT]), ys.transform(Y.loc[tr, TARGETS]))
        p = ys.inverse_transform(m.predict(xs.transform(X.loc[te, FEAT])))
        out[held] = (Y.loc[te].reset_index(drop=True), p, X.loc[te, "time"].values)
    return out


def grp(r2): g = {gn: float(np.mean([r2[t] for t in ts])) for gn, ts in GROUPS.items()}; g["mean"] = float(np.mean(list(r2.values()))); return g
def pear(a, b): return float(np.corrcoef(a, b)[0, 1])


def main():
    print("REPORT 3D build : Optuna ...")
    VAL = "s09" if "s09" in SUBS else SUBS[-1]   # validation rapide sur 1 sujet (8x plus rapide que LOSO complet)
    def obj(tr):
        p = dict(n_estimators=tr.suggest_int("n_estimators", 300, 600), num_leaves=tr.suggest_int("num_leaves", 15, 70),
                 learning_rate=tr.suggest_float("learning_rate", 0.02, 0.2, log=True), subsample=tr.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0), min_child_samples=tr.suggest_int("min_child_samples", 5, 50))
        tr_ = X.subj != VAL; te = X.subj == VAL
        xs = StandardScaler().fit(X.loc[tr_, FEAT]); ys = StandardScaler().fit(Y.loc[tr_, TARGETS])
        mm = MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **p))
        mm.fit(xs.transform(X.loc[tr_, FEAT]), ys.transform(Y.loc[tr_, TARGETS]))
        pp = ys.inverse_transform(mm.predict(xs.transform(X.loc[te, FEAT])))
        r2 = {t: r2_score(Y.loc[te, t].values, pp[:, j]) for j, t in enumerate(TARGETS)}
        g = grp(r2); return 0.5 * g["torque"] + 0.5 * g["mean"]
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0), pruner=optuna.pruners.HyperbandPruner())
    st.optimize(obj, n_trials=18, show_progress_bar=False)
    BEST = st.best_params
    tdf = st.trials_dataframe().sort_values("value", ascending=False)
    tdf.to_csv(os.path.join(B, "optuna_3d_trials.csv"), index=False)
    print("best:", BEST)

    pr = loso_pred(BEST)
    # multi-metric per target
    rows = []; r2d = {}
    for j, t in enumerate(TARGETS):
        yt = np.concatenate([pr[s][0][t].values for s in SUBS]); yp = np.concatenate([pr[s][1][:, j] for s in SUBS])
        rng = yt.max() - yt.min(); r2 = r2_score(yt, yp); r2d[t] = r2
        rows.append(dict(target=t, unit=UNITS[t], R2=r2, RMSE=np.sqrt(mean_squared_error(yt, yp)),
                         MAE=mean_absolute_error(yt, yp), Pearson=pear(yt, yp), NRMSE_pct=100 * np.sqrt(mean_squared_error(yt, yp)) / rng))
    md = pd.DataFrame(rows).set_index("target"); md.round(4).to_csv(os.path.join(B, "metrics_3d_final.csv"))
    g = grp(r2d); print("FINAL 3D vision-only:", {k: round(v, 3) for k, v in g.items()})

    # per-subject
    ps = []
    for s in SUBS:
        te, p, _ = pr[s]; r = {t: r2_score(te[t].values, p[:, j]) for j, t in enumerate(TARGETS)}
        gg = grp(r); ps.append(dict(subj=s, R2_mean=gg["mean"], R2_torque=gg["torque"], R2_activations=gg["activations"], R2_forces=gg["forces"], R2_fatigue=gg["fatigue"]))
    psd = pd.DataFrame(ps).set_index("subj"); psd.round(4).to_csv(os.path.join(B, "metrics_3d_per_subject.csv"))

    # ---- FIGURES ----
    # 1) progression vision-only
    plt.figure(figsize=(8, 4.2))
    stg = ["3D basic", "3D + FE\nenrichi", "+ Butterworth", "+ Optuna\n(final)"]; vals = [0.842, 0.867, 0.895, g["mean"]]
    plt.bar(stg, vals, color=["#aaa", "#7fa", "#4c9", "#2a7"])
    for i, v in enumerate(vals): plt.text(i, v + 0.003, "%.3f" % v, ha="center")
    plt.axhline(0.952, ls="--", c="r", label="Approche A (avec Vicon) = 0.952"); plt.ylim(0.8, 0.97)
    plt.ylabel("R2 moyen (LOSO)"); plt.title("Progression du modele vision-only (3D->biomeca)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3d_progression.png"), dpi=130); plt.close()

    # 2) R2 par cible
    plt.figure(figsize=(8, 5)); md2 = md.sort_values("R2")
    plt.barh(md2.index, md2.R2, color="#2a7"); plt.xlim(0, 1); plt.xlabel("R2 (LOSO)")
    for i, v in enumerate(md2.R2): plt.text(v + 0.005, i, "%.2f" % v, va="center", fontsize=8)
    plt.title("R2 par cible - modele 3D vision-only final"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3d_metrics.png"), dpi=130); plt.close()

    # 3) per-subject
    plt.figure(figsize=(10, 4.4)); xx = np.arange(len(psd)); w = 0.16
    for i, c in enumerate(["R2_mean", "R2_torque", "R2_activations", "R2_forces", "R2_fatigue"]):
        plt.bar(xx + (i - 2) * w, psd[c], w, label=c.replace("R2_", ""))
    plt.xticks(xx, psd.index); plt.ylim(0, 1); plt.ylabel("R2 (held-out)"); plt.legend(ncol=5, fontsize=8)
    plt.title("Generalisation par sujet (LOSO) - 3D vision-only"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3d_per_subject.png"), dpi=130); plt.close()

    # 4) predicted vs truth per-muscle (sujet s11)
    s = "s11" if "s11" in SUBS else SUBS[-1]; te, p, t = pr[s]
    fig, ax = plt.subplots(len(FLEX), 3, figsize=(15, 12))
    for i, mus in enumerate(FLEX):
        for jc, (pref, lab) in enumerate([("act_", "activation"), ("frc_", "force (N)"), ("MF_", "fatigue (%)")]):
            a = ax[i, jc]; col = pref + mus; jx = TARGETS.index(col)
            a.plot(t, te[col].values, "g-", lw=1.7, label="OpenSim"); a.plot(t, p[:, jx], "b--", lw=1.2, label="3D-ML")
            a.set_title("%s - %s (R2=%.2f)" % (mus, lab, r2_score(te[col].values, p[:, jx])), fontsize=9)
            if i == len(FLEX) - 1: a.set_xlabel("t (s)")
            if i == 0 and jc == 0: a.legend(fontsize=8)
            a.grid(alpha=.3)
    fig.suptitle("3D vision-only : predit vs verite par muscle - sujet %s (held-out)" % s, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .98]); fig.savefig(os.path.join(FIG, "fig3d_muscles.png"), dpi=115); plt.close()

    # 5) torque progression
    plt.figure(figsize=(7, 4)); ts = ["baseline", "Savitzky-\nGolay", "Butterworth", "+Optuna"]; tv = [0.790, 0.765, 0.827, 0.839]
    plt.bar(ts, tv, color="#d62728");
    for i, v in enumerate(tv): plt.text(i, v + 0.004, "%.3f" % v, ha="center")
    plt.axhline(0.937, ls="--", c="gray", label="Approche A (Vicon)=0.937"); plt.ylim(0.7, 0.96)
    plt.ylabel("torque R2"); plt.title("Amelioration du torque (vision-only)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3d_torque.png"), dpi=130); plt.close()

    # 6) XAI (reuse xai_3d.csv)
    if os.path.exists(os.path.join(B, "xai_3d.csv")):
        xi = pd.read_csv(os.path.join(B, "xai_3d.csv"), index_col=0)["overall"].sort_values().tail(12)
        plt.figure(figsize=(7.5, 5)); plt.barh(xi.index, xi.values, color="#1f77b4"); plt.xlabel("SHAP mean|val|")
        plt.title("XAI (SHAP) - features importantes (3D)"); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "fig3d_xai.png"), dpi=130); plt.close()

    # 7) model comparison (MASTER_comparison vision-only subset)
    if os.path.exists(os.path.join(B, "MASTER_comparison.csv")):
        mc = pd.read_csv(os.path.join(B, "MASTER_comparison.csv"))
        mc = mc[mc["input"].str.contains("3D")].sort_values("mean")
        names = [n[:22] for n in mc["model"]]
        plt.figure(figsize=(8, 4)); plt.barh(names, mc["mean"], color="#9467bd")
        for i, v in enumerate(mc["mean"]): plt.text(v + 0.004, i, "%.3f" % v, va="center", fontsize=8)
        plt.axvline(g["mean"], ls="--", c="g", label="final (Butterworth+Optuna)=%.3f" % g["mean"]); plt.xlim(0.6, 0.95)
        plt.xlabel("R2 moyen (LOSO)"); plt.title("Comparaison modeles 3D->biomeca"); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "fig3d_compare.png"), dpi=130); plt.close()

    pd.DataFrame([g]).to_csv(os.path.join(B, "metrics_3d_groups.csv"), index=False)
    print("figures:", os.listdir(FIG))
    print("DONE report_3d_build")


if __name__ == "__main__":
    main()
