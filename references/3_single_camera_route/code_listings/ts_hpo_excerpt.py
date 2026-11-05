"""Excerpt of the module `ts_hpo`, as printed by the executed single-camera notebook
(notebook/arm26_pipeline_report.ipynb, cell 67, via inspect.getsource).

Only the functions the notebook displays are here; the full module lives on the GPU
workstation that ran the notebook (/home/idriss/arm26_bridge). The code is unchanged.
"""

# ---- notebook cell 67
def objective(trial):
    mn = trial.suggest_categorical("model", CANDIDATES)
    d = trial.suggest_categorical("d", [64, 96, 128, 192, 256])
    layers = trial.suggest_int("layers", 2, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 3e-4, 4e-3, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    W = trial.suggest_categorical("W", [64, 96, 128, 192])
    bs = trial.suggest_categorical("bs", [32, 64, 128])
    mkw = {}
    if mn == "FEDformer": mkw["modes"] = trial.suggest_categorical("modes", [16, 32, 48, 64])
    if mn in ("FEDformer", "Transformer", "Informer", "TFTlite"):
        mkw["nhead"] = trial.suggest_categorical("nhead", [4, 8])
        if d % mkw["nhead"] != 0: mkw["nhead"] = 4
    try:
        r2, _ = H.run_cv(LEVEL, mn, W=W, S=W//2, folds=3, epochs=35, lr=lr, bs=bs, wd=wd,
                         d=d, layers=layers, dropout=dropout, aug_cum=True, aug_kin=True,
                         model_kw=mkw, sel_joints=SEL_JOINTS, aug_scale=AUG_SCALE)
    except Exception as e:
        raise optuna.TrialPruned()
    o = float(np.mean(r2))
    trial.set_user_attr("torque", float(r2[0])); trial.set_user_attr("MF", float(np.mean(r2[9:])))
    return o
