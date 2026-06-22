# pyright: reportMissingImports=false
"""
STAGE 3 — ML : prédire les variables biomécaniques (torque, forces, activations)
à partir des observations 2D vision (pinhole u,v bruités). Données : vision_dataset_v1
(OpenSim -> caméra virtuelle -> 2D + bruit -> labels ID/SO). NN (MLP) vs baseline (Ridge).
Sortie : batch/ml_results.csv + batch/ml_pred_elbow.png. biomech env (sklearn).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "Data", "GUI_Run_001", "vision_dataset_v1.csv")
OUT = os.path.join(ROOT, "batch")

df = pd.read_csv(DATA)
# INPUT = 2D vision observations (u,v bruités) — pas de profondeur (cas monoculaire réaliste)
X_cols = [c for c in df.columns if c.endswith("_u_noisy") or c.endswith("_v_noisy")]
TARGETS = ["id_elbow_moment", "id_shoulder_moment",
           "force_biceps", "force_triceps", "force_deltoid",
           "act_biceps", "act_triceps", "act_deltoid"]
X = df[X_cols].values
Y = df[TARGETS].values
print("dataset: %d frames, %d inputs 2D (%s), %d targets" % (len(df), len(X_cols), ", ".join(X_cols), len(TARGETS)))

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
xs = StandardScaler().fit(Xtr); ys = StandardScaler().fit(Ytr)
Xtr_s, Xte_s = xs.transform(Xtr), xs.transform(Xte)
Ytr_s = ys.transform(Ytr)

models = {
    "NN (MLP 128-64)": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800, random_state=0, early_stopping=True),
    "Baseline (Ridge)": Ridge(alpha=1.0),
}
results = {}
for name, mdl in models.items():
    mdl.fit(Xtr_s, Ytr_s)
    pred = ys.inverse_transform(mdl.predict(Xte_s))
    results[name] = pred

# --- report per target (RMSE, MAE, R2) for the NN ---
nn_pred = results["NN (MLP 128-64)"]
rows = []
print("\n%-20s %10s %10s %8s" % ("target", "RMSE", "MAE", "R2"))
for j, t in enumerate(TARGETS):
    rmse = np.sqrt(mean_squared_error(Yte[:, j], nn_pred[:, j]))
    mae = mean_absolute_error(Yte[:, j], nn_pred[:, j])
    r2 = r2_score(Yte[:, j], nn_pred[:, j])
    unit = "N·m" if t.startswith("id_") else ("N" if t.startswith("force") else "")
    print("%-20s %8.3f %10.3f %8.3f" % (t, rmse, mae, r2))
    rows.append((t, rmse, mae, r2, unit))

# baseline overall (mean R2)
ridge_pred = results["Baseline (Ridge)"]
nn_r2 = np.mean([r2_score(Yte[:, j], nn_pred[:, j]) for j in range(len(TARGETS))])
rg_r2 = np.mean([r2_score(Yte[:, j], ridge_pred[:, j]) for j in range(len(TARGETS))])
print("\nR² moyen — NN: %.3f | Ridge baseline: %.3f" % (nn_r2, rg_r2))

# save
with open(os.path.join(OUT, "ml_results.csv"), "w", newline="\n") as f:
    f.write("target,RMSE,MAE,R2,unit\n")
    for t, rmse, mae, r2, u in rows:
        f.write("%s,%.4f,%.4f,%.4f,%s\n" % (t, rmse, mae, r2, u))
    f.write("NN_meanR2,%.4f,,,\nRidge_meanR2,%.4f,,,\n" % (nn_r2, rg_r2))

# plot elbow torque + biceps activation (prediction vs true) on test set
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, j, lab in [(ax[0], 0, "elbow torque (N·m)"), (ax[1], 5, "biceps activation")]:
    o = np.argsort(Yte[:, j])
    a.plot(Yte[o, j], Yte[o, j], "k--", lw=1, label="ideal")
    a.scatter(Yte[:, j], nn_pred[:, j], s=4, alpha=0.4, c="#1f77b4")
    a.set_xlabel("Vicon/OpenSim (vrai)"); a.set_ylabel("ML prédit"); a.set_title(lab); a.grid(alpha=.3); a.legend()
fig.suptitle("ML : 2D vision -> variables biomécaniques (test set)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(OUT, "ml_pred_elbow.png"), dpi=130); plt.close()
print("wrote", os.path.join(OUT, "ml_results.csv"), "+ ml_pred_elbow.png")
