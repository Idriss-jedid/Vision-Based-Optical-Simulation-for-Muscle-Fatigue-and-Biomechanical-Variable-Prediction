# pyright: reportMissingImports=false
"""
APPROCHE A — BENCHMARK de modèles tabulaires (per-frame) en LEAVE-ONE-SUBJECT-OUT.
Compare : Ridge, RandomForest, ExtraTrees, MLP(sklearn), XGBoost, LightGBM
+ XGBoost tuné par OPTUNA (Bayesian optimization sur le R² moyen LOSO).
INPUT 11 features (cinématique + anthropométrie) -> 13 labels (torque/act/forces/MF).
Sortie : batch/bench_tabular.csv (R² par groupe et par modèle). biomech env.
"""
import os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
DATA = os.path.join(ROOT, "batch", "ml_dataset_A.csv"); OUT = os.path.join(ROOT, "batch")
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
X_COLS = ["q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el", "time",
          "humerus_mass", "forearm_mass", "humerus_len", "forearm_len"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]
GROUPS = {"torque": ["elbow_moment"], "activations": ["act_" + m for m in FLEX],
          "forces": ["frc_" + m for m in FLEX], "fatigue": ["MF_" + m for m in FLEX]}

df = pd.read_csv(DATA); SUBS = sorted(df["subj"].unique())


def loso_eval(make_model):
    """retourne (r2 par target moyenné sur les folds) en leave-one-subject-out."""
    acc = {t: [] for t in TARGETS}
    for held in SUBS:
        tr, te = df[df.subj != held], df[df.subj == held]
        xs = StandardScaler().fit(tr[X_COLS]); ys = StandardScaler().fit(tr[TARGETS])
        Xtr, Xte = xs.transform(tr[X_COLS]), xs.transform(te[X_COLS])
        Ytr = ys.transform(tr[TARGETS]); Yte = te[TARGETS].values
        mdl = make_model(); mdl.fit(Xtr, Ytr)
        pred = ys.inverse_transform(mdl.predict(Xte))
        for j, t in enumerate(TARGETS):
            acc[t].append(r2_score(Yte[:, j], pred[:, j]))
    return {t: float(np.mean(v)) for t, v in acc.items()}


def group_r2(r2_per_target):
    return {g: float(np.mean([r2_per_target[t] for t in ts])) for g, ts in GROUPS.items()}


def main():
    models = {
        "Ridge": lambda: Ridge(alpha=1.0),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0),
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=0),
        "MLP(sklearn)": lambda: MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=600,
                                             early_stopping=True, random_state=0),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = lambda: MultiOutputRegressor(XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, n_jobs=-1, random_state=0))
    except Exception as e:
        print("XGBoost absent:", e)
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = lambda: MultiOutputRegressor(LGBMRegressor(
            n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, n_jobs=-1, random_state=0, verbose=-1))
    except Exception as e:
        print("LightGBM absent:", e)

    results = {}
    for name, mk in models.items():
        t0 = time.time(); r2 = loso_eval(mk); g = group_r2(r2)
        g["mean"] = float(np.mean(list(r2.values()))); results[name] = g
        print("%-14s mean R2=%.3f  | torque %.3f  act %.3f  forces %.3f  fatigue %.3f  (%.0fs)" %
              (name, g["mean"], g["torque"], g["activations"], g["forces"], g["fatigue"], time.time() - t0))

    # ---- OPTUNA : tuning XGBoost (Bayesian) sur le R2 moyen LOSO ----
    try:
        import optuna
        from xgboost import XGBRegressor
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(tr):
            p = dict(n_estimators=tr.suggest_int("n_estimators", 200, 800),
                     max_depth=tr.suggest_int("max_depth", 3, 8),
                     learning_rate=tr.suggest_float("learning_rate", 0.01, 0.2, log=True),
                     subsample=tr.suggest_float("subsample", 0.6, 1.0),
                     colsample_bytree=tr.suggest_float("colsample_bytree", 0.6, 1.0),
                     min_child_weight=tr.suggest_int("min_child_weight", 1, 8),
                     reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
            r2 = loso_eval(lambda: MultiOutputRegressor(XGBRegressor(n_jobs=-1, random_state=0, **p)))
            return float(np.mean(list(r2.values())))

        print("\nOptuna (XGBoost, 30 trials, Bayesian)...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        best = study.best_params
        r2 = loso_eval(lambda: MultiOutputRegressor(XGBRegressor(n_jobs=-1, random_state=0, **best)))
        g = group_r2(r2); g["mean"] = float(np.mean(list(r2.values()))); results["XGB+Optuna"] = g
        print("XGB+Optuna   mean R2=%.3f  | torque %.3f  act %.3f  forces %.3f  fatigue %.3f" %
              (g["mean"], g["torque"], g["activations"], g["forces"], g["fatigue"]))
        print("  best params:", best)
    except Exception as e:
        print("Optuna/XGBoost tuning sauté:", e)

    # ---- rapport ----
    rep = pd.DataFrame(results).T[["mean", "torque", "activations", "forces", "fatigue"]].round(3)
    rep = rep.sort_values("mean", ascending=False)
    rep.to_csv(os.path.join(OUT, "bench_tabular.csv"))
    print("\n=== CLASSEMENT (R² moyen LOSO) ===\n", rep)
    print("\nwrote batch/bench_tabular.csv")


if __name__ == "__main__":
    main()
