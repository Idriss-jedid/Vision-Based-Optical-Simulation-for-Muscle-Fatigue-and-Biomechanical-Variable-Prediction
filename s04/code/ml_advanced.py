# pyright: reportMissingImports=false
"""
APPROCHE A — pipeline avancé (8 sujets, LOSO cross-validation) :
  1) FEATURE ENGINEERING (dérivé de la cinématique+anthropométrie, sans OpenSim) :
     sin/cos angles, |vitesses|, énergie, proxy de charge gravitaire, et features CUMULATIFS
     (path, impulsion gravitaire, rep) -> surtout pour la FATIGUE.
  2) LightGBM : BASE(11) vs ENGINEERED -> gain.
  3) OPTUNA (Bayesian + Hyperband/BOHB) : tuning LightGBM (objectif = R² fatigue + global).
  4) XAI : SHAP (TreeExplainer) + permutation importance -> features importantes/groupe,
     puis sélection de features pilotée par XAI -> ré-évaluation.
Sortie : batch/ml_advanced.csv, batch/xai_importance.csv, batch/xai_shap_fatigue.png. biomech env.
"""
import os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
import optuna, shap
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
DATA = os.path.join(ROOT, "batch", "ml_dataset_A.csv"); OUT = os.path.join(ROOT, "batch")
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]
GROUPS = {"torque": ["elbow_moment"], "activations": ["act_" + m for m in FLEX],
          "forces": ["frc_" + m for m in FLEX], "fatigue": ["MF_" + m for m in FLEX]}
BASE = ["q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el", "time",
        "humerus_mass", "forearm_mass", "humerus_len", "forearm_len"]


def engineer(df):
    """ajoute des features dérivées (par sujet pour les cumulatifs)."""
    g = np.pi / 180.0
    df["sin_qel"] = np.sin(df.q_el * g); df["cos_qel"] = np.cos(df.q_el * g)
    df["sin_qsh"] = np.sin(df.q_sh * g); df["cos_qsh"] = np.cos(df.q_sh * g)
    df["abs_qd_el"] = df.qd_el.abs(); df["abs_qdd_el"] = df.qdd_el.abs()
    df["qd_el2"] = df.qd_el ** 2
    # proxy de charge gravitaire au coude : (avant-bras + 2kg) * bras-levier * sin(orientation/vertical)
    df["grav_load"] = (df.forearm_mass + 2.0) * df.forearm_len * np.sin((df.q_sh + df.q_el) * g)
    df["qel_x_fmass"] = df.q_el * df.forearm_mass
    # CUMULATIFS par sujet (reset à chaque sujet) -> drivers de la fatigue
    cum_path, cum_grav = [], []
    for s, sub in df.groupby("subj", sort=False):
        dt = np.median(np.diff(sub.time.values)) if len(sub) > 1 else 0.01
        cum_path.append(np.cumsum(sub.abs_qd_el.values) * dt)
        cum_grav.append(np.cumsum(np.abs(sub.grav_load.values)) * dt)
    df["cum_path_el"] = np.concatenate(cum_path)
    df["cum_grav_imp"] = np.concatenate(cum_grav)
    # NB: 'rep' retiré -> compteur erroné (6 au lieu de 5) ; cum_path_el / cum_grav_imp
    # sont de meilleurs proxys cumulatifs pour la fatigue de toute façon.
    eng = BASE + ["sin_qel", "cos_qel", "sin_qsh", "cos_qsh", "abs_qd_el", "abs_qdd_el",
                  "qd_el2", "grav_load", "qel_x_fmass", "cum_path_el", "cum_grav_imp"]
    return df, eng


df = pd.read_csv(DATA); df, ENG = engineer(df)
SUBS = sorted(df.subj.unique())
print("dataset: %d frames, %d sujets ; BASE=%d feat, ENG=%d feat" % (len(df), len(SUBS), len(BASE), len(ENG)))


def loso(feats, make, ret_pred=False):
    acc = {t: [] for t in TARGETS}; preds = {}
    for held in SUBS:
        tr, te = df[df.subj != held], df[df.subj == held]
        xs = StandardScaler().fit(tr[feats]); ys = StandardScaler().fit(tr[TARGETS])
        m = make(); m.fit(xs.transform(tr[feats]), ys.transform(tr[TARGETS]))
        p = ys.inverse_transform(m.predict(xs.transform(te[feats])))
        for j, t in enumerate(TARGETS):
            acc[t].append(r2_score(te[t].values, p[:, j]))
        if ret_pred: preds[held] = (te, p)
    r2 = {t: float(np.mean(v)) for t, v in acc.items()}
    return (r2, preds) if ret_pred else r2


def grp(r2): return {g: float(np.mean([r2[t] for t in ts])) for g, ts in GROUPS.items()}


def lgbm(**kw):
    from sklearn.multioutput import MultiOutputRegressor
    d = dict(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.9,
             colsample_bytree=0.9, n_jobs=-1, random_state=0, verbose=-1)
    d.update(kw)
    return MultiOutputRegressor(LGBMRegressor(**d))


def show(tag, r2):
    g = grp(r2); g["mean"] = float(np.mean(list(r2.values())))
    print("%-22s mean=%.3f | torque %.3f act %.3f forces %.3f fatigue %.3f" %
          (tag, g["mean"], g["torque"], g["activations"], g["forces"], g["fatigue"]))
    return g


def main():
    res = {}
    print("\n--- 1) LightGBM : BASE vs ENGINEERED (LOSO) ---")
    res["LGBM_base"] = show("LGBM BASE(11)", loso(BASE, lgbm))
    res["LGBM_eng"] = show("LGBM ENG(%d)" % len(ENG), loso(ENG, lgbm))

    print("\n--- 2) OPTUNA tuning LightGBM sur ENG (BOHB-like) ---")
    def obj(tr):
        kw = dict(n_estimators=tr.suggest_int("n_estimators", 300, 1000),
                  num_leaves=tr.suggest_int("num_leaves", 15, 90),
                  learning_rate=tr.suggest_float("learning_rate", 0.01, 0.2, log=True),
                  subsample=tr.suggest_float("subsample", 0.6, 1.0),
                  colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0),
                  min_child_samples=tr.suggest_int("min_child_samples", 5, 60),
                  reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10, log=True))
        r2 = loso(ENG, lambda: lgbm(**kw))
        g = grp(r2)
        return 0.5 * g["fatigue"] + 0.5 * float(np.mean(list(r2.values())))  # focus fatigue + global
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0),
                                pruner=optuna.pruners.HyperbandPruner())
    study.optimize(obj, n_trials=40, show_progress_bar=False)
    best = study.best_params
    res["LGBM_eng_optuna"] = show("LGBM ENG+Optuna", loso(ENG, lambda: lgbm(**best)))
    print("  best:", best)

    # --- 3) XAI : SHAP (TreeExplainer) + permutation importance ---
    print("\n--- 3) XAI : importance des features ---")
    xs = StandardScaler().fit(df[ENG]); Xall = xs.transform(df[ENG])
    shap_imp = {g: np.zeros(len(ENG)) for g in GROUPS}
    for t in TARGETS:
        ys = StandardScaler().fit(df[[t]]); yv = ys.transform(df[[t]]).ravel()
        m = LGBMRegressor(**{**dict(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                    n_jobs=-1, random_state=0, verbose=-1), **best}).fit(Xall, yv)
        sv = shap.TreeExplainer(m).shap_values(Xall)
        mean_abs = np.abs(sv).mean(0)
        for g, ts in GROUPS.items():
            if t in ts: shap_imp[g] += mean_abs / len(ts)
    imp_df = pd.DataFrame(shap_imp, index=ENG)
    imp_df["overall"] = imp_df.mean(1)
    imp_df = imp_df.sort_values("overall", ascending=False)
    imp_df.round(4).to_csv(os.path.join(OUT, "xai_importance.csv"))
    print("Top 8 features (SHAP mean|val|, par groupe) :")
    print(imp_df.head(8).round(3))

    # plot SHAP importance pour la FATIGUE
    fat = imp_df["fatigue"].sort_values(ascending=True).tail(12)
    plt.figure(figsize=(7, 5)); plt.barh(fat.index, fat.values, color="#d62728")
    plt.xlabel("SHAP mean|valeur| (importance)"); plt.title("XAI — features pour la FATIGUE (MF)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "xai_shap_fatigue.png"), dpi=125); plt.close()

    # --- 4) sélection de features pilotée par XAI : top-K ---
    print("\n--- 4) Feature selection via XAI (top-K features) ---")
    for K in [8, 12]:
        topk = imp_df.head(K).index.tolist()
        res["LGBM_top%d" % K] = show("LGBM top-%d (XAI)" % K, loso(topk, lambda: lgbm(**best)))

    rep = pd.DataFrame(res).T[["mean", "torque", "activations", "forces", "fatigue"]].round(3)
    rep.to_csv(os.path.join(OUT, "ml_advanced.csv"))
    print("\n=== RÉCAP (R² LOSO, 8 sujets) ===\n", rep.sort_values("mean", ascending=False))
    print("\nwrote ml_advanced.csv, xai_importance.csv, xai_shap_fatigue.png")


if __name__ == "__main__":
    main()
