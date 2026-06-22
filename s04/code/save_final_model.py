# pyright: reportMissingImports=false
"""
Entraîne et SAUVEGARDE le modèle final retenu = LightGBM (Optuna trial #11), multi-sortie,
sur les 8 sujets COMPLETS (modèle de production). Sauve le modèle + les scalers + les
métadonnées (features, cibles, hyper-paramètres, métriques LOSO) -> batch/model_final/.
Fournit aussi un exemple d'inférence. biomech env.
"""
import os, json
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch"); OUTD = os.path.join(B, "model_final"); os.makedirs(OUTD, exist_ok=True)
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]
TRIAL11 = dict(n_estimators=975, num_leaves=15, learning_rate=0.196, subsample=0.605,
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
    return df


FEATURES = ["q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el", "time", "humerus_mass", "forearm_mass",
            "humerus_len", "forearm_len", "sin_qel", "cos_qel", "sin_qsh", "cos_qsh", "abs_qd_el",
            "abs_qdd_el", "qd_el2", "grav_load", "qel_x_fmass", "cum_path_el", "cum_grav_imp"]


def main():
    df = engineer(pd.read_csv(os.path.join(B, "ml_dataset_A.csv")))
    X, Y = df[FEATURES].values, df[TARGETS].values
    xs = StandardScaler().fit(X); ys = StandardScaler().fit(Y)
    model = MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **TRIAL11))
    model.fit(xs.transform(X), ys.transform(Y))
    print("modèle entraîné sur %d frames / %d sujets, %d features -> %d cibles" %
          (len(df), df.subj.nunique(), len(FEATURES), len(TARGETS)))

    bundle = dict(model=model, x_scaler=xs, y_scaler=ys, features=FEATURES, targets=TARGETS,
                  hyperparams=TRIAL11, n_train_frames=len(df), n_subjects=int(df.subj.nunique()))
    joblib.dump(bundle, os.path.join(OUTD, "lgbm_trial11.joblib"))

    # métadonnées lisibles
    meta = dict(model="LightGBM (MultiOutput) - Optuna trial #11", trained_on="8 subjects (all data)",
                features=FEATURES, targets=TARGETS, hyperparams=TRIAL11,
                validation="Leave-One-Subject-Out (8 folds)")
    try:
        m = pd.read_csv(os.path.join(B, "metrics_final.csv"), index_col=0)
        meta["loso_metrics_per_target"] = {k: {c: round(float(m.loc[k, c]), 4) for c in m.columns if c != "unit"}
                                           for k in m.index}
    except Exception:
        pass
    json.dump(meta, open(os.path.join(OUTD, "model_card.json"), "w"), indent=2)

    # exemple d'inférence
    usage = '''# Exemple d'inference avec le modele sauvegarde
import joblib, numpy as np
b = joblib.load("batch/model_final/lgbm_trial11.joblib")
# X_new : tableau (n_frames, 22) dans l'ordre b["features"]
Xs = b["x_scaler"].transform(X_new)
Y  = b["y_scaler"].inverse_transform(b["model"].predict(Xs))  # (n_frames, 13) dans l'ordre b["targets"]
'''
    open(os.path.join(OUTD, "USAGE.py"), "w").write(usage)

    # sanity : ré-inférence sur un sujet
    sub = df[df.subj == df.subj.unique()[0]]
    yhat = ys.inverse_transform(model.predict(xs.transform(sub[FEATURES].values)))
    print("sanity: prédiction OK, forme", yhat.shape)
    print("\nSAUVEGARDÉ dans", OUTD)
    print(" - lgbm_trial11.joblib (modèle + scalers + features/targets)")
    print(" - model_card.json (métadonnées + métriques LOSO)")
    print(" - USAGE.py (exemple d'inférence)")


if __name__ == "__main__":
    main()
