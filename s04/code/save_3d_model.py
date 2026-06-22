# pyright: reportMissingImports=false
"""Entraine et SAUVEGARDE le meilleur modele 3D vision-only (Butterworth FE + Optuna) sur les 8
sujets complets -> batch/model_3d_final/. biomech env."""
import os, json
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
import final_3d_model as M  # features Butterworth (X, Y, FEAT, TARGETS)

B = M.B; OUTD = os.path.join(B, "model_3d_final"); os.makedirs(OUTD, exist_ok=True)
BEST = dict(n_estimators=306, num_leaves=43, learning_rate=0.120, subsample=0.698,
            colsample_bytree=0.893, min_child_samples=16)


def main():
    X, Y, FEAT, TARGETS = M.X, M.Y, M.FEAT, M.TARGETS
    xs = StandardScaler().fit(X[FEAT]); ys = StandardScaler().fit(Y[TARGETS])
    model = MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **BEST))
    model.fit(xs.transform(X[FEAT]), ys.transform(Y[TARGETS]))
    joblib.dump(dict(model=model, x_scaler=xs, y_scaler=ys, features=FEAT, targets=TARGETS,
                     hyperparams=BEST, input="3 joints (RShoulder/RElbow/RWrist) -> Butterworth FE"),
                os.path.join(OUTD, "lgbm_3d_vision.joblib"))
    meta = dict(model="LightGBM (MultiOutput) - 3D vision-only (Butterworth FE + Optuna)",
                input="3D joints RShoulder/RElbow/RWrist (.trc) -> 30 engineered features",
                output=TARGETS, hyperparams=BEST, validation="LOSO 8 subjects",
                loso_mean_R2=0.904, note="no Vicon, no OpenSim at inference")
    try:
        m = pd.read_csv(os.path.join(B, "metrics_3d_final.csv"), index_col=0)
        meta["loso_metrics"] = {k: round(float(m.loc[k, "R2"]), 4) for k in m.index}
    except Exception: pass
    json.dump(meta, open(os.path.join(OUTD, "model_card.json"), "w"), indent=2)
    print("SAUVEGARDE:", OUTD)
    print(" - lgbm_3d_vision.joblib (%d features -> %d cibles)" % (len(FEAT), len(TARGETS)))
    print(" - model_card.json (mean LOSO R2 = 0.904)")


if __name__ == "__main__":
    main()
