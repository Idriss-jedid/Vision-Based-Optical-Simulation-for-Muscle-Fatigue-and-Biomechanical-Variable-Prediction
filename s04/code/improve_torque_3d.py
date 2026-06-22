# pyright: reportMissingImports=false
"""
AMELIORATION TORQUE (3D vision-only) : on pousse le torque au-dela de 0.79 sans Vicon, par
(1) lissage Butterworth 2 Hz zero-phase de l'angle avant derivation (q'' propre),
(2) features de RETARD (lags +/-5, +/-10) pour donner un contexte dynamique,
(3) Optuna SPECIFIQUE au torque (single-output -> ses propres hyperparams).
Compare la progression du torque (LOSO, 8 sujets). biomech env.
"""
import os, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch"); G = 9.81
MKIDX = {"Hip": 1, "RShoulder": 17, "RElbow": 18, "RWrist": 19}
TGT = "elbow_moment"


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


def feats(subj, mode):
    """mode: 'savgol' (reference), 'butter', 'butter_lag'."""
    trc = sorted(glob.glob(os.path.join(B, subj, "pose2sim", "pose-3d", "*filt_butterworth.trc")))[0]
    t, hip, sh, el, wr = read_trc(trc)
    lab = pd.read_csv(os.path.join(B, subj, "labels_ml.csv")); tl = lab["time"].values
    up_vec = np.nanmean(sh, 0) - np.nanmean(hip, 0); axu = int(np.argmax(np.abs(up_vec)))
    up = np.zeros(3); up[axu] = np.sign(up_vec[axu])
    def itp(a): return np.column_stack([np.interp(tl, t, a[:, k]) for k in range(3)])
    sh, el, wr = itp(sh), itp(el), itp(wr); dt = float(np.median(np.diff(tl))); fs = 1 / dt
    SE = el - sh; WE = wr - el
    ua_len = np.linalg.norm(SE, axis=1); fa_len = np.linalg.norm(WE, axis=1)
    q = ang(-SE, WE); ua_e = ang(SE, np.tile(up, (len(SE), 1))); fa_e = ang(WE, np.tile(up, (len(WE), 1)))
    WE_up = WE @ up; hm = lab["humerus_mass"].values; fm = lab["forearm_mass"].values
    if mode == "savgol":
        win = 21 if len(q) > 21 else (len(q) // 2 * 2 - 1)
        qs = savgol_filter(q, win, 3); qd = savgol_filter(q, win, 3, 1, delta=dt); qdd = savgol_filter(q, win, 3, 2, delta=dt)
        fae = savgol_filter(fa_e, win, 3)
    else:  # butter / butter_lag
        qs = bw(q, fs); fae = bw(fa_e, fs); qd = np.gradient(qs, dt); qdd = np.gradient(qd, dt)
    grav = G * np.sin(fae) * (fm * fa_len / 2 + 2.0 * fa_len)
    inertia = (fm * fa_len ** 2 / 3 + 2.0 * fa_len ** 2) * qdd
    F = dict(q=qs, qd=qd, qdd=qdd, fa_elev=fae, ua_elev=ua_e, WE_up=WE_up, ua_len=ua_len, fa_len=fa_len,
             humerus_mass=hm, forearm_mass=fm, grav=grav, inertia=inertia, phys=grav + inertia, time=tl)
    X = pd.DataFrame(F)
    X["roll_mean_q"] = X["q"].rolling(31, center=True, min_periods=1).mean()
    X["roll_mean_grav"] = X["grav"].rolling(31, center=True, min_periods=1).mean()
    if mode == "butter_lag":
        for L in [5, 10, 20]:
            X["q_lag%d" % L] = X["q"].shift(L).bfill(); X["q_lead%d" % L] = X["q"].shift(-L).ffill()
            X["grav_lag%d" % L] = X["grav"].shift(L).bfill()
    X["subj"] = subj; return X, lab[TGT].values


SUBS = sorted([os.path.basename(p) for p in glob.glob(os.path.join(B, "s*"))
               if os.path.isdir(p) and os.path.exists(os.path.join(p, "labels_ml.csv"))])


def build(mode):
    XS, ys = [], []
    for s in SUBS:
        X, y = feats(s, mode); XS.append(X); ys.append(pd.DataFrame({"y": y, "subj": s}))
    return pd.concat(XS, ignore_index=True), pd.concat(ys, ignore_index=True)


def loso_torque(X, Y, feats_, params=None):
    p = params or dict(n_estimators=600, num_leaves=31, learning_rate=0.05)
    r2s, maes = [], []
    for held in SUBS:
        tr = X.subj != held; te = X.subj == held
        xs = StandardScaler().fit(X.loc[tr, feats_]); ys = StandardScaler().fit(Y.loc[tr, ["y"]])
        m = LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **p)
        m.fit(xs.transform(X.loc[tr, feats_]), ys.transform(Y.loc[tr, ["y"]]).ravel())
        pr = ys.inverse_transform(m.predict(xs.transform(X.loc[te, feats_])).reshape(-1, 1)).ravel()
        r2s.append(r2_score(Y.loc[te, "y"].values, pr)); maes.append(mean_absolute_error(Y.loc[te, "y"].values, pr))
    return float(np.mean(r2s)), float(np.mean(maes))


def main():
    print("AMELIORATION TORQUE (3D vision-only, LOSO %d sujets)\n" % len(SUBS))
    for mode, tag in [("savgol", "ref: Savitzky-Golay"), ("butter", "Butterworth 2Hz"), ("butter_lag", "Butterworth + lags")]:
        X, Y = build(mode); fe = [c for c in X.columns if c != "subj"]
        r2, mae = loso_torque(X, Y, fe)
        print("%-24s torque R2=%.3f  MAE=%.3f N.m  (%d feat)" % (tag, r2, mae, len(fe)))
        if mode == "butter_lag": Xb, Yb, feb = X, Y, fe

    print("\n-- OPTUNA torque-specifique (single-output, Butterworth+lags) --")
    def obj(tr):
        p = dict(n_estimators=tr.suggest_int("n_estimators", 300, 1200), num_leaves=tr.suggest_int("num_leaves", 15, 120),
                 learning_rate=tr.suggest_float("learning_rate", 0.01, 0.2, log=True), subsample=tr.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0), min_child_samples=tr.suggest_int("min_child_samples", 5, 60),
                 reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10, log=True))
        return loso_torque(Xb, Yb, feb, p)[0]
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0), pruner=optuna.pruners.HyperbandPruner())
    st.optimize(obj, n_trials=40, show_progress_bar=False)
    r2, mae = loso_torque(Xb, Yb, feb, st.best_params)
    print("Butterworth+lags+Optuna   torque R2=%.3f  MAE=%.3f N.m" % (r2, mae))
    print("  best:", st.best_params)
    print("\nRappel: torque baseline (FE 3D)=0.790 | Approche A (Vicon)=0.937")
    pd.DataFrame([dict(stage="butter_lag_optuna", torque_R2=r2, MAE=mae)]).to_csv(os.path.join(B, "improve_torque_3d.csv"), index=False)
    print("wrote improve_torque_3d.csv")


if __name__ == "__main__":
    main()
