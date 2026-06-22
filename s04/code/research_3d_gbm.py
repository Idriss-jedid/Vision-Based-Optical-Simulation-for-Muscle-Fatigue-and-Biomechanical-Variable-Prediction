# pyright: reportMissingImports=false
"""
BRANCHE 2 (3D -> biomeca) : Gradient Boosting (XGBoost / LightGBM) + FEATURE ENGINEERING enrichi
des 3 keypoints 3D (RShoulder/RElbow/RWrist + verticale), + OPTUNA + XAI (SHAP) + selection de
features. Cycle FE -> modele -> Optuna -> XAI -> selection. LOSO, 8 sujets.
Sortie: batch/research_3d_gbm.csv + batch/xai_3d.csv. biomech env.
"""
import os, glob, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna, shap
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch")
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]
GROUPS = {"torque": ["elbow_moment"], "activations": ["act_" + m for m in FLEX],
          "forces": ["frc_" + m for m in FLEX], "fatigue": ["MF_" + m for m in FLEX]}
MKIDX = {"Hip": 1, "RShoulder": 17, "RElbow": 18, "RWrist": 19}


def read_trc(path):
    rows = []
    for ln in open(path).read().splitlines():
        p = ln.split("\t")
        if len(p) > 59:
            try:
                float(p[0]); float(p[1]); rows.append([float(x) if x.strip() else np.nan for x in p])
            except ValueError:
                continue
    d = np.array(rows); t = d[:, 1]
    def mk(n): c = 2 + (MKIDX[n] - 1) * 3; return d[:, c:c + 3]
    return t, mk("Hip"), mk("RShoulder"), mk("RElbow"), mk("RWrist")


def ang(u, v):
    cs = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.arccos(np.clip(cs, -1, 1))


def features_3d(subj):
    trc = sorted(glob.glob(os.path.join(B, subj, "pose2sim", "pose-3d", "*filt_butterworth.trc")))[0]
    t, hip, sh, el, wr = read_trc(trc)
    lab = pd.read_csv(os.path.join(B, subj, "labels_ml.csv")); tl = lab["time"].values
    up_vec = np.nanmean(sh, 0) - np.nanmean(hip, 0); axu = int(np.argmax(np.abs(up_vec)))
    up = np.zeros(3); up[axu] = np.sign(up_vec[axu])
    def itp(a): return np.column_stack([np.interp(tl, t, a[:, k]) for k in range(3)])
    sh, el, wr = itp(sh), itp(el), itp(wr)
    dt = float(np.median(np.diff(tl)))
    SE = el - sh; WE = wr - el
    ua_len = np.linalg.norm(SE, axis=1); fa_len = np.linalg.norm(WE, axis=1); sw = np.linalg.norm(wr - sh, axis=1)
    q_el = ang(-SE, WE); ua_elev = ang(SE, np.tile(up, (len(SE), 1))); fa_elev = ang(WE, np.tile(up, (len(WE), 1)))
    SE_up = SE @ up; WE_up = WE @ up; wrist_up = (wr - sh) @ up

    def d2(x): v = np.gradient(x, dt); a = np.gradient(v, dt); return v, a
    qd, qdd = d2(q_el); uad, _ = d2(ua_elev); fad, _ = d2(fa_elev); wud, _ = d2(wrist_up)
    hm = lab["humerus_mass"].values; fm = lab["forearm_mass"].values
    grav = (fm + 2.0) * fa_len * np.sin(fa_elev)
    cum_path = np.cumsum(np.abs(qd)) * dt; cum_grav = np.cumsum(np.abs(grav)) * dt
    F = dict(ua_len=ua_len, fa_len=fa_len, sw_dist=sw, q_el=q_el, ua_elev=ua_elev, fa_elev=fa_elev,
             SE_up=SE_up, WE_up=WE_up, wrist_up=wrist_up, qd=qd, qdd=qdd, uad=uad, fad=fad, wrist_up_d=wud,
             abs_qd=np.abs(qd), abs_qdd=np.abs(qdd), qd2=qd ** 2, sin_qel=np.sin(q_el), cos_qel=np.cos(q_el),
             grav=grav, cum_path=cum_path, cum_grav=cum_grav, time=tl, humerus_mass=hm, forearm_mass=fm)
    X = pd.DataFrame(F)
    # contexte temporel (rolling, +-15) -> donne au modele tabulaire un peu de sequence
    for col, w in [("q_el", 15), ("grav", 15), ("qd", 9)]:
        X["roll_mean_" + col] = X[col].rolling(2 * w + 1, center=True, min_periods=1).mean()
        X["roll_std_" + col] = X[col].rolling(2 * w + 1, center=True, min_periods=1).std().fillna(0)
    X["subj"] = subj
    Y = lab[TARGETS].reset_index(drop=True); Y["subj"] = subj
    return X, Y


# ----- charge tous les sujets -----
subs = sorted([os.path.basename(p) for p in glob.glob(os.path.join(B, "s*"))
               if os.path.isdir(p) and os.path.exists(os.path.join(p, "labels_ml.csv"))])
XS, YS = [], []
for s in subs:
    try:
        X, Y = features_3d(s); XS.append(X); YS.append(Y)
    except Exception as e:
        print("skip", s, e)
X = pd.concat(XS, ignore_index=True); Y = pd.concat(YS, ignore_index=True)
FEAT = [c for c in X.columns if c != "subj"]
SUBS = sorted(X.subj.unique())


def loso(make, feats):
    acc = {t: [] for t in TARGETS}
    for held in SUBS:
        tr = X.subj != held; te = X.subj == held
        xs = StandardScaler().fit(X.loc[tr, feats]); ys = StandardScaler().fit(Y.loc[tr, TARGETS])
        m = make(); m.fit(xs.transform(X.loc[tr, feats]), ys.transform(Y.loc[tr, TARGETS]))
        p = ys.inverse_transform(m.predict(xs.transform(X.loc[te, feats])))
        for j, t in enumerate(TARGETS): acc[t].append(r2_score(Y.loc[te, t].values, p[:, j]))
    return {t: float(np.mean(v)) for t, v in acc.items()}


def grp(r2): g = {gn: float(np.mean([r2[t] for t in ts])) for gn, ts in GROUPS.items()}; g["mean"] = float(np.mean(list(r2.values()))); return g
def show(tag, r2): g = grp(r2); print("%-22s mean=%.3f | torque %.3f act %.3f forces %.3f fatigue %.3f" % (tag, g["mean"], g["torque"], g["activations"], g["forces"], g["fatigue"])); return g


def lgbm(**k): return MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **{**dict(n_estimators=500, num_leaves=31, learning_rate=0.05), **k}))
def xgb(**k): return MultiOutputRegressor(XGBRegressor(n_jobs=-1, random_state=0, **{**dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9), **k}))


def main():
    res = {}
    print("BRANCHE 2 : GBM + FE 3D enrichi (%d features, LOSO)\n" % len(FEAT))
    res["XGBoost_3Dfe"] = show("XGBoost (3D FE)", loso(xgb, FEAT))
    res["LightGBM_3Dfe"] = show("LightGBM (3D FE)", loso(lgbm, FEAT))

    print("\n-- OPTUNA (LightGBM, 3D FE, BOHB-like) --")
    def obj(tr):
        k = dict(n_estimators=tr.suggest_int("n_estimators", 300, 1000), num_leaves=tr.suggest_int("num_leaves", 15, 90),
                 learning_rate=tr.suggest_float("learning_rate", 0.01, 0.2, log=True),
                 subsample=tr.suggest_float("subsample", 0.6, 1.0), colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0),
                 min_child_samples=tr.suggest_int("min_child_samples", 5, 60), reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10, log=True))
        r2 = loso(lambda: lgbm(**k), FEAT); g = grp(r2)
        return 0.5 * g["fatigue"] + 0.5 * g["mean"]
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0), pruner=optuna.pruners.HyperbandPruner())
    st.optimize(obj, n_trials=35, show_progress_bar=False); best = st.best_params
    res["LGBM_3Dfe_Optuna"] = show("LGBM 3D FE + Optuna", loso(lambda: lgbm(**best), FEAT)); print("  best:", best)

    print("\n-- XAI (SHAP) + selection --")
    xs = StandardScaler().fit(X[FEAT]); Xall = xs.transform(X[FEAT]); imp = np.zeros(len(FEAT))
    for t in TARGETS:
        ys = StandardScaler().fit(Y[[t]]); yv = ys.transform(Y[[t]]).ravel()
        m = LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **best).fit(Xall, yv)
        imp += np.abs(shap.TreeExplainer(m).shap_values(Xall)).mean(0) / len(TARGETS)
    xi = pd.Series(imp, index=FEAT).sort_values(ascending=False); xi.round(4).to_csv(os.path.join(B, "xai_3d.csv"))
    print("Top 10 features (SHAP):"); print(xi.head(10).round(3).to_string())
    for K in [10, 15]:
        top = xi.head(K).index.tolist()
        res["LGBM_3Dfe_top%d" % K] = show("LGBM 3D FE top-%d (XAI)" % K, loso(lambda: lgbm(**best), top))

    rep = pd.DataFrame(res).T[["mean", "torque", "activations", "forces", "fatigue"]].round(3).sort_values("mean", ascending=False)
    rep.to_csv(os.path.join(B, "research_3d_gbm.csv"))
    print("\n=== CLASSEMENT (3D -> biomeca, GBM+FE, LOSO) ===\n", rep)
    print("Reference: Approche A (angles cleaned+FE+Optuna) = 0.952")
    print("wrote research_3d_gbm.csv + xai_3d.csv")


if __name__ == "__main__":
    main()
