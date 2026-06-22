# Exemple d'inference avec le modele sauvegarde
import joblib, numpy as np
b = joblib.load("batch/model_final/lgbm_trial11.joblib")
# X_new : tableau (n_frames, 22) dans l'ordre b["features"]
Xs = b["x_scaler"].transform(X_new)
Y  = b["y_scaler"].inverse_transform(b["model"].predict(Xs))  # (n_frames, 13) dans l'ordre b["targets"]
