# pyright: reportMissingImports=false
"""
DEEP-DIVE TORQUE (cas 3D) : pourquoi le torque est faible et comment le corriger.
Hypothese : torque = M(q)qdd + C qd + G(q). Sur le 3D brut, qdd (derivee 2nde) est bruyant et
les termes physiques (gravite, inertie) ne sont pas explicites -> torque faible (0.79).
FIX : (1) lissage Savitzky-Golay des angles avant derivation -> qdd propre ;
      (2) features PHYSIQUES explicites : moment gravitaire + terme inertiel + leur somme.
On compare le torque LOSO AVANT/APRES, + SHAP pour confirmer. biomech env.
"""
import os, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
import shap

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


def sg(x, dt, win=21, deriv=0):
    win = min(win, len(x) - (1 - len(x) % 2));  win = win if win % 2 else win - 1
    if win < 5: return x if deriv == 0 else np.gradient(x, dt)
    return savgol_filter(x, win, 3, deriv=deriv, delta=dt)


def features(subj, physics=True, smooth=True):
    trc = sorted(glob.glob(os.path.join(B, subj, "pose2sim", "pose-3d", "*filt_butterworth.trc")))[0]
    t, hip, sh, el, wr = read_trc(trc)
    lab = pd.read_csv(os.path.join(B, subj, "labels_ml.csv")); tl = lab["time"].values
    up_vec = np.nanmean(sh, 0) - np.nanmean(hip, 0); axu = int(np.argmax(np.abs(up_vec)))
    up = np.zeros(3); up[axu] = np.sign(up_vec[axu])
    def itp(a): return np.column_stack([np.interp(tl, t, a[:, k]) for k in range(3)])
    sh, el, wr = itp(sh), itp(el), itp(wr); dt = float(np.median(np.diff(tl)))
    SE = el - sh; WE = wr - el
    ua_len = np.linalg.norm(SE, axis=1); fa_len = np.linalg.norm(WE, axis=1)
    q_el = ang(-SE, WE); ua_elev = ang(SE, np.tile(up, (len(SE), 1))); fa_elev = ang(WE, np.tile(up, (len(WE), 1)))
    WE_up = WE @ up
    hm = lab["humerus_mass"].values; fm = lab["forearm_mass"].values
    # derivees : Savitzky-Golay (propre) vs gradient (bruyant)
    if smooth:
        qd = sg(q_el, dt, 21, 1); qdd = sg(q_el, dt, 21, 2); q_s = sg(q_el, dt, 21, 0)
        fae_s = sg(fa_elev, dt, 21, 0)
    else:
        qd = np.gradient(q_el, dt); qdd = np.gradient(qd, dt); q_s = q_el; fae_s = fa_elev
    F = dict(ua_len=ua_len, fa_len=fa_len, q_el=q_s, ua_elev=ua_elev, fa_elev=fae_s, WE_up=WE_up,
             qd=qd, qdd=qdd, abs_qd=np.abs(qd), humerus_mass=hm, forearm_mass=fm, time=tl)
    if physics:
        # moment gravitaire au coude (terme dominant) : g*sin(angle/vertical)*(m_fa*L/2 + m_load*L)
        F["grav_torque"] = G * np.sin(fae_s) * (fm * fa_len / 2 + 2.0 * fa_len)
        # terme inertiel : I*qdd , I = m_fa*L^2/3 + m_load*L^2
        F["inertia_torque"] = (fm * fa_len ** 2 / 3 + 2.0 * fa_len ** 2) * qdd
        F["phys_torque"] = F["grav_torque"] + F["inertia_torque"]
    X = pd.DataFrame(F)
    for col, w in [("q_el", 15), ("grav_torque" if physics else "q_el", 9)]:
        X["roll_mean_" + col] = X[col].rolling(2 * w + 1, center=True, min_periods=1).mean()
    X["subj"] = subj; Y = lab[TARGETS].reset_index(drop=True); Y["subj"] = subj
    return X, Y


def build(physics, smooth):
    XS, YS = [], []
    for s in SUBS:
        X, Y = features(s, physics, smooth); XS.append(X); YS.append(Y)
    return pd.concat(XS, ignore_index=True), pd.concat(YS, ignore_index=True)


SUBS = sorted([os.path.basename(p) for p in glob.glob(os.path.join(B, "s*"))
               if os.path.isdir(p) and os.path.exists(os.path.join(p, "labels_ml.csv"))])


def loso(X, Y, feats):
    acc = {t: [] for t in TARGETS}; mae_t = []
    for held in SUBS:
        tr = X.subj != held; te = X.subj == held
        xs = StandardScaler().fit(X.loc[tr, feats]); ys = StandardScaler().fit(Y.loc[tr, TARGETS])
        m = MultiOutputRegressor(LGBMRegressor(n_estimators=600, num_leaves=31, learning_rate=0.05, n_jobs=-1, random_state=0, verbose=-1))
        m.fit(xs.transform(X.loc[tr, feats]), ys.transform(Y.loc[tr, TARGETS]))
        p = ys.inverse_transform(m.predict(xs.transform(X.loc[te, feats])))
        for j, t in enumerate(TARGETS): acc[t].append(r2_score(Y.loc[te, t].values, p[:, j]))
        mae_t.append(mean_absolute_error(Y.loc[te, "elbow_moment"].values, p[:, 0]))
    r2 = {t: float(np.mean(v)) for t, v in acc.items()}
    g = {gn: float(np.mean([r2[t] for t in ts])) for gn, ts in GROUPS.items()}; g["mean"] = float(np.mean(list(r2.values())))
    return g, float(np.mean(mae_t))


def show(tag, X, Y, feats):
    g, mae = loso(X, Y, feats)
    print("%-32s torque R2=%.3f (MAE %.2f N.m) | act %.3f forces %.3f fatigue %.3f | mean %.3f" %
          (tag, g["torque"], mae, g["activations"], g["forces"], g["fatigue"], g["mean"]))
    return g


def main():
    print("DEEP-DIVE TORQUE (3D, LOSO %d sujets)\n" % len(SUBS))
    # A) baseline : gradient brut, sans physique
    Xb, Yb = build(physics=False, smooth=False); fb = [c for c in Xb.columns if c != "subj"]
    show("A) brut (gradient, no physics)", Xb, Yb, fb)
    # B) + Savitzky-Golay (qdd propre)
    Xs, Ys = build(physics=False, smooth=True); fs = [c for c in Xs.columns if c != "subj"]
    show("B) + Savitzky-Golay smoothing", Xs, Ys, fs)
    # C) + features physiques (gravite + inertie)
    Xp, Yp = build(physics=True, smooth=True); fp = [c for c in Xp.columns if c != "subj"]
    gC = show("C) + physics features (grav+inertie)", Xp, Yp, fp)

    # XAI : SHAP sur le torque uniquement (modele C)
    print("\n-- SHAP (torque, modele C) : quelles features portent le torque ? --")
    xs = StandardScaler().fit(Xp[fp]); Xall = xs.transform(Xp[fp])
    ys = StandardScaler().fit(Yp[["elbow_moment"]]); yv = ys.transform(Yp[["elbow_moment"]]).ravel()
    m = LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.05, n_jobs=-1, random_state=0, verbose=-1).fit(Xall, yv)
    imp = pd.Series(np.abs(shap.TreeExplainer(m).shap_values(Xall)).mean(0), index=fp).sort_values(ascending=False)
    print(imp.head(8).round(3).to_string())
    pd.DataFrame({"torque_baseline": [0.790], "torque_C": [gC["torque"]]}).to_csv(os.path.join(B, "fix_torque_3d.csv"), index=False)
    print("\nwrote fix_torque_3d.csv")


if __name__ == "__main__":
    main()
