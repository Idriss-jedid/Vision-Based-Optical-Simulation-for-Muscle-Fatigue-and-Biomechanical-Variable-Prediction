# pyright: reportMissingImports=false
"""Donnees + figures + modele pour le RAPPORT 2D keypoints -> Biomecanique.
Optuna (top-K) + metriques multi + per-subject + figures + save model. Reutilise research_2d. biomech env."""
import os, json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import optuna, shap
optuna.logging.set_verbosity(optuna.logging.WARNING)
import research_2d as R   # module-level: charge X, Y, FEAT, SUBS ...

B, X, Y, FEAT, SUBS, TARGETS, GROUPS, FLEX = R.B, R.X, R.Y, R.FEAT, R.SUBS, R.TARGETS, R.GROUPS, R.FLEX
FIG = os.path.join(B, "report_figs_2d"); os.makedirs(FIG, exist_ok=True)
UNITS = {"elbow_moment": "N.m", **{"act_" + m: "" for m in FLEX}, **{"frc_" + m: "N" for m in FLEX}, **{"MF_" + m: "%" for m in FLEX}}


def lgbm(**k): return MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **{**dict(n_estimators=500, num_leaves=31, learning_rate=0.05), **k}))
def grp(r2): g = {gn: float(np.mean([r2[t] for t in ts])) for gn, ts in GROUPS.items()}; g["mean"] = float(np.mean(list(r2.values()))); return g
def pear(a, b): return float(np.corrcoef(a, b)[0, 1])


def loso_pred(params):
    out = {}
    for held in SUBS:
        tr = X.subj != held; te = X.subj == held
        xs = StandardScaler().fit(X.loc[tr, FEAT]); ys = StandardScaler().fit(Y.loc[tr, TARGETS])
        m = lgbm(**params); m.fit(xs.transform(X.loc[tr, FEAT]), ys.transform(Y.loc[tr, TARGETS]))
        p = ys.inverse_transform(m.predict(xs.transform(X.loc[te, FEAT])))
        out[held] = (Y.loc[te].reset_index(drop=True), p, X.loc[te, "time"].values)
    return out


def main():
    print("REPORT 2D : Optuna (val rapide 1 sujet) ...", flush=True)
    VAL = "s09" if "s09" in SUBS else SUBS[-1]
    def obj(tr):
        p = dict(n_estimators=tr.suggest_int("n_estimators", 300, 700), num_leaves=tr.suggest_int("num_leaves", 15, 70),
                 learning_rate=tr.suggest_float("learning_rate", 0.02, 0.2, log=True), subsample=tr.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0), min_child_samples=tr.suggest_int("min_child_samples", 5, 50))
        trm = X.subj != VAL; te = X.subj == VAL
        xs = StandardScaler().fit(X.loc[trm, FEAT]); ys = StandardScaler().fit(Y.loc[trm, TARGETS])
        mm = lgbm(**p); mm.fit(xs.transform(X.loc[trm, FEAT]), ys.transform(Y.loc[trm, TARGETS]))
        pp = ys.inverse_transform(mm.predict(xs.transform(X.loc[te, FEAT])))
        return grp({t: r2_score(Y.loc[te, t].values, pp[:, j]) for j, t in enumerate(TARGETS)})["mean"]
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0), pruner=optuna.pruners.HyperbandPruner())
    st.optimize(obj, n_trials=18, show_progress_bar=False); BEST = st.best_params
    st.trials_dataframe().sort_values("value", ascending=False).to_csv(os.path.join(B, "optuna_2d_trials.csv"), index=False)
    print("best:", BEST, flush=True)

    pr = loso_pred(BEST)
    rows = []; r2d = {}
    for j, t in enumerate(TARGETS):
        yt = np.concatenate([pr[s][0][t].values for s in SUBS]); yp = np.concatenate([pr[s][1][:, j] for s in SUBS])
        rng = yt.max() - yt.min(); r2 = r2_score(yt, yp); r2d[t] = r2
        rows.append(dict(target=t, unit=UNITS[t], R2=r2, RMSE=np.sqrt(mean_squared_error(yt, yp)), MAE=mean_absolute_error(yt, yp),
                         Pearson=pear(yt, yp), NRMSE_pct=100 * np.sqrt(mean_squared_error(yt, yp)) / rng))
    md = pd.DataFrame(rows).set_index("target"); md.round(4).to_csv(os.path.join(B, "metrics_2d_final.csv"))
    g = grp(r2d); print("FINAL 2D:", {k: round(v, 3) for k, v in g.items()}, flush=True)
    ps = []
    for s in SUBS:
        te, p, _ = pr[s]; r = {t: r2_score(te[t].values, p[:, j]) for j, t in enumerate(TARGETS)}; gg = grp(r)
        ps.append(dict(subj=s, R2_mean=gg["mean"], R2_torque=gg["torque"], R2_activations=gg["activations"], R2_forces=gg["forces"], R2_fatigue=gg["fatigue"]))
    pd.DataFrame(ps).set_index("subj").round(4).to_csv(os.path.join(B, "metrics_2d_per_subject.csv"))

    # FIGURES
    plt.figure(figsize=(7.5, 4.2)); stg = ["2D brut\n(sans FE)", "2D + FE\n(fusion vues)", "+ Optuna"]; vals = [0.631, 0.845, g["mean"]]
    plt.bar(stg, vals, color=["#bbb", "#48c", "#26a"])
    for i, v in enumerate(vals): plt.text(i, v + 0.005, "%.3f" % v, ha="center")
    plt.axhline(0.904, ls="--", c="g", label="3D=0.904"); plt.axhline(0.952, ls="--", c="r", label=".mot+.osim=0.952")
    plt.ylim(0.55, 0.98); plt.ylabel("R2 moyen (LOSO)"); plt.title("Progression du modele 2D keypoints"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2d_progression.png"), dpi=130); plt.close()

    plt.figure(figsize=(8, 5)); md2 = md.sort_values("R2"); plt.barh(md2.index, md2.R2, color="#26a"); plt.xlim(0, 1); plt.xlabel("R2")
    for i, v in enumerate(md2.R2): plt.text(v + 0.005, i, "%.2f" % v, va="center", fontsize=8)
    plt.title("R2 par cible - 2D final"); plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2d_metrics.png"), dpi=130); plt.close()

    psd = pd.read_csv(os.path.join(B, "metrics_2d_per_subject.csv"), index_col=0)
    plt.figure(figsize=(10, 4.4)); xx = np.arange(len(psd)); w = 0.16
    for i, c in enumerate(["R2_mean", "R2_torque", "R2_activations", "R2_forces", "R2_fatigue"]): plt.bar(xx + (i - 2) * w, psd[c], w, label=c.replace("R2_", ""))
    plt.xticks(xx, psd.index); plt.ylim(0, 1); plt.ylabel("R2"); plt.legend(ncol=5, fontsize=8); plt.title("Per-subject (LOSO) - 2D"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2d_per_subject.png"), dpi=130); plt.close()

    s = "s11" if "s11" in SUBS else SUBS[-1]; te, p, t = pr[s]
    fig, ax = plt.subplots(len(FLEX), 3, figsize=(15, 12))
    for i, mus in enumerate(FLEX):
        for jc, (pref, lab) in enumerate([("act_", "activation"), ("frc_", "force (N)"), ("MF_", "fatigue (%)")]):
            a = ax[i, jc]; col = pref + mus; jx = TARGETS.index(col)
            a.plot(t, te[col].values, "g-", lw=1.6, label="OpenSim"); a.plot(t, p[:, jx], "b--", lw=1.1, label="2D-ML")
            a.set_title("%s - %s (R2=%.2f)" % (mus, lab, r2_score(te[col].values, p[:, jx])), fontsize=9)
            if i == 0 and jc == 0: a.legend(fontsize=8)
            a.grid(alpha=.3)
    fig.suptitle("2D keypoints : predit vs verite par muscle - sujet %s (held-out)" % s, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .98]); fig.savefig(os.path.join(FIG, "fig2d_muscles.png"), dpi=115); plt.close()

    if os.path.exists(os.path.join(B, "xai_2d.csv")):
        xi = pd.read_csv(os.path.join(B, "xai_2d.csv"), index_col=0).iloc[:, 0].sort_values().tail(12)
        plt.figure(figsize=(7.5, 5)); plt.barh(xi.index, xi.values, color="#48c"); plt.xlabel("SHAP mean|val|")
        plt.title("XAI (SHAP) - features 2D"); plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2d_xai.png"), dpi=130); plt.close()

    plt.figure(figsize=(7.5, 3.6)); appr = ["2D brut", "2D + FE", "3D joints", ".mot + .osim"]; av = [0.631, g["mean"], 0.904, 0.952]
    plt.barh(appr, av, color=["#bbb", "#26a", "#2a7", "#a22"])
    for i, v in enumerate(av): plt.text(v + 0.005, i, "%.3f" % v, va="center", fontsize=9)
    plt.xlim(0.5, 1.0); plt.xlabel("R2 moyen (LOSO)"); plt.title("Comparaison des 3 approches"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2d_compare.png"), dpi=130); plt.close()

    # save model (entraine sur tous les sujets)
    OUTD = os.path.join(B, "model_2d_final"); os.makedirs(OUTD, exist_ok=True)
    xs = StandardScaler().fit(X[FEAT]); ys = StandardScaler().fit(Y[TARGETS])
    mdl = lgbm(**BEST); mdl.fit(xs.transform(X[FEAT]), ys.transform(Y[TARGETS]))
    joblib.dump(dict(model=mdl, x_scaler=xs, y_scaler=ys, features=FEAT, targets=TARGETS, hyperparams=BEST), os.path.join(OUTD, "lgbm_2d.joblib"))
    json.dump(dict(model="LightGBM 2D keypoints (4 cams, FE+Optuna)", loso_mean_R2=g["mean"], hyperparams=BEST, n_features=len(FEAT)),
              open(os.path.join(OUTD, "model_card.json"), "w"), indent=2)
    pd.DataFrame([g]).to_csv(os.path.join(B, "metrics_2d_groups.csv"), index=False)
    print("figures:", os.listdir(FIG), flush=True); print("DONE report_2d_build", flush=True)


if __name__ == "__main__":
    main()
