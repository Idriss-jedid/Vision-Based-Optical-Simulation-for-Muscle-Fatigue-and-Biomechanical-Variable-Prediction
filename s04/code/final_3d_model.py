# pyright: reportMissingImports=false
"""
MODELE VISION-ONLY FINAL (3D, 13 cibles) : on applique la recette gagnante du torque
(lissage Butterworth 2 Hz zero-phase -> q'' propre) a TOUTES les features, + features
physiques + rolling + cumulatifs, puis LightGBM multi-sortie + Optuna (LOSO, 8 sujets).
But : remonter le mean au-dela de 0.867 en corrigeant le torque (0.79->0.84). biomech env.
"""
import os, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch"); G = 9.81
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


def bw(x, fs, fc=2.0):
    b, a = butter(2, fc / (fs / 2)); return filtfilt(b, a, x)


def features(subj):
    trc = sorted(glob.glob(os.path.join(B, subj, "pose2sim", "pose-3d", "*filt_butterworth.trc")))[0]
    t, hip, sh, el, wr = read_trc(trc); lab = pd.read_csv(os.path.join(B, subj, "labels_ml.csv")); tl = lab["time"].values
    up_vec = np.nanmean(sh, 0) - np.nanmean(hip, 0); axu = int(np.argmax(np.abs(up_vec)))
    up = np.zeros(3); up[axu] = np.sign(up_vec[axu])
    def itp(a): return np.column_stack([np.interp(tl, t, a[:, k]) for k in range(3)])
    sh, el, wr = itp(sh), itp(el), itp(wr); dt = float(np.median(np.diff(tl))); fs = 1 / dt
    SE = el - sh; WE = wr - el
    ua_len = np.linalg.norm(SE, axis=1); fa_len = np.linalg.norm(WE, axis=1); sw = np.linalg.norm(wr - sh, axis=1)
    q = bw(ang(-SE, WE), fs); ua_e = bw(ang(SE, np.tile(up, (len(SE), 1))), fs); fa_e = bw(ang(WE, np.tile(up, (len(WE), 1))), fs)
    WE_up = WE @ up; wrist_up = (wr - sh) @ up
    qd = np.gradient(q, dt); qdd = np.gradient(qd, dt); uad = np.gradient(ua_e, dt); fad = np.gradient(fa_e, dt)
    hm = lab["humerus_mass"].values; fm = lab["forearm_mass"].values
    grav = G * np.sin(fa_e) * (fm * fa_len / 2 + 2.0 * fa_len)
    inertia = (fm * fa_len ** 2 / 3 + 2.0 * fa_len ** 2) * qdd
    cum_path = np.cumsum(np.abs(qd)) * dt; cum_grav = np.cumsum(np.abs(grav)) * dt
    F = dict(q=q, qd=qd, qdd=qdd, ua_elev=ua_e, fa_elev=fa_e, WE_up=WE_up, wrist_up=wrist_up,
             ua_len=ua_len, fa_len=fa_len, sw_dist=sw, sin_q=np.sin(q), cos_q=np.cos(q),
             abs_qd=np.abs(qd), abs_qdd=np.abs(qdd), uad=uad, fad=fad, humerus_mass=hm, forearm_mass=fm,
             grav=grav, inertia=inertia, phys=grav + inertia, cum_path=cum_path, cum_grav=cum_grav, time=tl)
    X = pd.DataFrame(F)
    for col in ["q", "grav", "qd"]:
        X["roll_mean_" + col] = X[col].rolling(31, center=True, min_periods=1).mean()
        X["roll_std_" + col] = X[col].rolling(31, center=True, min_periods=1).std().fillna(0)
    X["subj"] = subj; Y = lab[TARGETS].reset_index(drop=True); Y["subj"] = subj
    return X, Y


SUBS = sorted([os.path.basename(p) for p in glob.glob(os.path.join(B, "s*"))
               if os.path.isdir(p) and os.path.exists(os.path.join(p, "labels_ml.csv"))])
XS, YS = [], []
for s in SUBS:
    X, Y = features(s); XS.append(X); YS.append(Y)
X = pd.concat(XS, ignore_index=True); Y = pd.concat(YS, ignore_index=True)
FEAT = [c for c in X.columns if c != "subj"]


def loso(params):
    acc = {t: [] for t in TARGETS}
    for held in SUBS:
        tr = X.subj != held; te = X.subj == held
        xs = StandardScaler().fit(X.loc[tr, FEAT]); ys = StandardScaler().fit(Y.loc[tr, TARGETS])
        m = MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **params))
        m.fit(xs.transform(X.loc[tr, FEAT]), ys.transform(Y.loc[tr, TARGETS]))
        p = ys.inverse_transform(m.predict(xs.transform(X.loc[te, FEAT])))
        for j, t in enumerate(TARGETS): acc[t].append(r2_score(Y.loc[te, t].values, p[:, j]))
    r2 = {t: float(np.mean(v)) for t, v in acc.items()}
    g = {gn: float(np.mean([r2[t] for t in ts])) for gn, ts in GROUPS.items()}; g["mean"] = float(np.mean(list(r2.values())))
    return g


def main():
    print("MODELE VISION-ONLY FINAL (3D + Butterworth, %d features, LOSO %d sujets)\n" % (len(FEAT), len(SUBS)))
    g = loso(dict(n_estimators=600, num_leaves=31, learning_rate=0.05))
    print("Butterworth + FE (default LGBM)  mean=%.3f | torque %.3f act %.3f forces %.3f fatigue %.3f" %
          (g["mean"], g["torque"], g["activations"], g["forces"], g["fatigue"]))
    print("\n-- Optuna (20 trials, combined obj) --")
    def obj(tr):
        p = dict(n_estimators=tr.suggest_int("n_estimators", 300, 700), num_leaves=tr.suggest_int("num_leaves", 15, 80),
                 learning_rate=tr.suggest_float("learning_rate", 0.02, 0.2, log=True), subsample=tr.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0), min_child_samples=tr.suggest_int("min_child_samples", 5, 50))
        gg = loso(p); return 0.5 * gg["torque"] + 0.5 * gg["mean"]
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0), pruner=optuna.pruners.HyperbandPruner())
    st.optimize(obj, n_trials=20, show_progress_bar=False)
    gF = loso(st.best_params)
    print("Butterworth + FE + Optuna        mean=%.3f | torque %.3f act %.3f forces %.3f fatigue %.3f" %
          (gF["mean"], gF["torque"], gF["activations"], gF["forces"], gF["fatigue"]))
    print("  best:", st.best_params)
    print("\nAvant (FE 3D savgol): mean=0.867 torque=0.790 | Approche A (Vicon)=0.952")
    pd.DataFrame([gF]).to_csv(os.path.join(B, "final_3d_model.csv"), index=False)
    print("wrote final_3d_model.csv")


if __name__ == "__main__":
    main()
