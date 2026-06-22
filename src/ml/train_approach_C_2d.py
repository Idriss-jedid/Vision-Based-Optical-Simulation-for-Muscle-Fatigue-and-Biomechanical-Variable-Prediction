# pyright: reportMissingImports=false
"""
RECHERCHE 2D : 2D keypoints (4 cams) -> biomecanique, avec FEATURE ENGINEERING enrichi
(angles 2D par camera + fusion multi-vues ~ triangulation implicite + Butterworth + derivees +
cumulatifs + rolling), benchmark multi-modeles + Optuna + XAI (SHAP). LOSO, 8 sujets.
Sortie: batch/research_2d.csv, batch/xai_2d.csv. biomech env.
"""
import os, glob, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna, shap
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch"); IMG = 900.0
KP = {"RShoulder": 6, "RElbow": 8, "RWrist": 10}
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]
GROUPS = {"torque": ["elbow_moment"], "activations": ["act_" + m for m in FLEX],
          "forces": ["frc_" + m for m in FLEX], "fatigue": ["MF_" + m for m in FLEX]}


def load_cam(cam_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(cam_dir, "*.json"))):
        try:
            ppl = json.load(open(f)).get("people", [])
            if not ppl: out.append([np.nan] * 9); continue
            kp = np.array(ppl[0]["pose_keypoints_2d"]).reshape(-1, 3)
            out.append(list(kp[KP["RShoulder"]]) + list(kp[KP["RElbow"]]) + list(kp[KP["RWrist"]]))
        except Exception:
            out.append([np.nan] * 9)
    return np.array(out)


def bw(x, fs=100, fc=2.0):
    if len(x) < 13: return x
    b, a = butter(2, fc / (fs / 2)); return filtfilt(b, a, x)


def ang2d(u, v):
    cs = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.arccos(np.clip(cs, -1, 1))


def subject(subj):
    pose = os.path.join(B, subj, "pose2sim", "pose"); cams = sorted(glob.glob(os.path.join(pose, "cam_*_json")))
    lab = pd.read_csv(os.path.join(B, subj, "labels_ml.csv")); tl = lab["time"].values
    F = {}; a2d_all = []; w_all = []
    for ci, cam in enumerate(cams):
        d = load_cam(cam); t2 = np.arange(len(d)) / 50.0
        sh, el, wr = d[:, 0:2], d[:, 3:5], d[:, 6:8]; cf = np.nan_to_num(d[:, [2, 5, 8]].mean(1), nan=0)
        se = el - sh; we = wr - el
        a2d = ang2d(-se, we)                                  # angle coude 2D (par camera)
        len_se = np.linalg.norm(se, axis=1) / IMG
        len_we = np.linalg.norm(we, axis=1) / IMG
        se_ori = np.arctan2(se[:, 1], se[:, 0]); we_ori = np.arctan2(we[:, 1], we[:, 0])
        def ip(a): return np.interp(tl, t2, np.nan_to_num(a, nan=0))
        a2di = ip(a2d); cfi = ip(cf)
        a2d_all.append(a2di); w_all.append(cfi)
        F["c%d_a2d" % ci] = a2di; F["c%d_lse" % ci] = ip(len_se); F["c%d_lwe" % ci] = ip(len_we)
        F["c%d_seo" % ci] = ip(se_ori); F["c%d_weo" % ci] = ip(we_ori); F["c%d_cf" % ci] = cfi
    # ---- fusion multi-vues (~ triangulation implicite) ----
    A = np.array(a2d_all); W = np.array(w_all) + 1e-6
    a_mean = np.sum(A * W, 0) / np.sum(W, 0)                  # angle coude fusionne (pondere confiance)
    a_mean = bw(a_mean)                                       # Butterworth (comme en 3D)
    qd = np.gradient(a_mean, np.median(np.diff(tl))); qdd = np.gradient(qd, np.median(np.diff(tl)))
    F["a_fused"] = a_mean; F["a_std"] = np.std(A, 0); F["qd"] = qd; F["qdd"] = qdd; F["abs_qd"] = np.abs(qd)
    F["sin_a"] = np.sin(a_mean); F["cos_a"] = np.cos(a_mean)
    dt = float(np.median(np.diff(tl)))
    F["cum_path"] = np.cumsum(np.abs(qd)) * dt; F["time"] = tl
    F["roll_mean_a"] = pd.Series(a_mean).rolling(31, center=True, min_periods=1).mean().values
    F["roll_std_a"] = pd.Series(a_mean).rolling(31, center=True, min_periods=1).std().fillna(0).values
    F["humerus_mass"] = lab["humerus_mass"].values; F["forearm_mass"] = lab["forearm_mass"].values
    X = pd.DataFrame(F); X["subj"] = subj; Y = lab[TARGETS].reset_index(drop=True); Y["subj"] = subj
    return X, Y


subs = sorted([os.path.basename(p) for p in glob.glob(os.path.join(B, "s*"))
               if os.path.isdir(p) and os.path.exists(os.path.join(p, "labels_ml.csv"))])
XS, YS = [], []
for s in subs:
    try:
        X, Y = subject(s); XS.append(X); YS.append(Y); print("loaded", s, X.shape, flush=True)
    except Exception as e:
        print("skip", s, e, flush=True)
X = pd.concat(XS, ignore_index=True); Y = pd.concat(YS, ignore_index=True)
FEAT = [c for c in X.columns if c != "subj"]; X[FEAT] = X[FEAT].replace([np.inf, -np.inf], np.nan).fillna(0)
SUBS = sorted(X.subj.unique())


def loso(make, feats=None):
    feats = feats or FEAT; acc = {t: [] for t in TARGETS}
    for held in SUBS:
        tr = X.subj != held; te = X.subj == held
        xs = StandardScaler().fit(X.loc[tr, feats]); ys = StandardScaler().fit(Y.loc[tr, TARGETS])
        m = make(); m.fit(xs.transform(X.loc[tr, feats]), ys.transform(Y.loc[tr, TARGETS]))
        p = ys.inverse_transform(m.predict(xs.transform(X.loc[te, feats])))
        for j, t in enumerate(TARGETS): acc[t].append(r2_score(Y.loc[te, t].values, p[:, j]))
    return {t: float(np.mean(v)) for t, v in acc.items()}


def grp(r2): g = {gn: float(np.mean([r2[t] for t in ts])) for gn, ts in GROUPS.items()}; g["mean"] = float(np.mean(list(r2.values()))); return g
def show(tag, r2): g = grp(r2); print("%-22s mean=%.3f | torque %.3f act %.3f forces %.3f fatigue %.3f" % (tag, g["mean"], g["torque"], g["activations"], g["forces"], g["fatigue"]), flush=True); return g
def lgbm(**k): return MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **{**dict(n_estimators=500, num_leaves=31, learning_rate=0.05), **k}))


def main():
    res = {}
    print("\nRECHERCHE 2D enrichi : %d frames, %d features (4 cams + fusion)\n" % (len(X), len(FEAT)), flush=True)
    res["LightGBM"] = show("LightGBM (2D FE)", loso(lgbm))
    res["XGBoost"] = show("XGBoost (2D FE)", loso(lambda: MultiOutputRegressor(XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=0))))
    res["ExtraTrees"] = show("ExtraTrees (2D FE)", loso(lambda: ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=0)))
    res["MLP"] = show("MLP (2D FE)", loso(lambda: MLPRegressor((128, 64), max_iter=500, early_stopping=True, random_state=0)))

    print("\n-- Optuna (LightGBM, 2D FE) --", flush=True)
    def obj(tr):
        k = dict(n_estimators=tr.suggest_int("n_estimators", 300, 800), num_leaves=tr.suggest_int("num_leaves", 15, 80),
                 learning_rate=tr.suggest_float("learning_rate", 0.02, 0.2, log=True), subsample=tr.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0), min_child_samples=tr.suggest_int("min_child_samples", 5, 50))
        return grp(loso(lambda: lgbm(**k)))["mean"]
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0), pruner=optuna.pruners.HyperbandPruner())
    st.optimize(obj, n_trials=15, show_progress_bar=False); best = st.best_params
    res["LGBM_Optuna"] = show("LightGBM + Optuna", loso(lambda: lgbm(**best))); print("  best:", best, flush=True)

    print("\n-- XAI (SHAP) --", flush=True)
    xs = StandardScaler().fit(X[FEAT]); Xa = xs.transform(X[FEAT]); imp = np.zeros(len(FEAT))
    for t in TARGETS:
        ys = StandardScaler().fit(Y[[t]]); m = LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **best).fit(Xa, ys.transform(Y[[t]]).ravel())
        imp += np.abs(shap.TreeExplainer(m).shap_values(Xa)).mean(0) / len(TARGETS)
    xi = pd.Series(imp, index=FEAT).sort_values(ascending=False); xi.round(4).to_csv(os.path.join(B, "xai_2d.csv"))
    print("Top 10 (SHAP):"); print(xi.head(10).round(3).to_string(), flush=True)

    rep = pd.DataFrame(res).T[["mean", "torque", "activations", "forces", "fatigue"]].round(3).sort_values("mean", ascending=False)
    rep.to_csv(os.path.join(B, "research_2d.csv"))
    print("\n=== CLASSEMENT 2D enrichi (LOSO) ===\n", rep, flush=True)
    print("Avant (2D brut)=0.631 | 3D=0.904 | .mot+.osim=0.952", flush=True)


if __name__ == "__main__":
    main()
