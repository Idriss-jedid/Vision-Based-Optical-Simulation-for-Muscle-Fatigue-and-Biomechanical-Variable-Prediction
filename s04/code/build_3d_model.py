# pyright: reportMissingImports=false
"""
Modele END-TO-END : input = 3D keypoints bruts du .trc (RShoulder/RElbow/RWrist), output =
biomecanique (torque/forces/activations/fatigue). On NE passe PAS par les angles precalcules
(.mot) ni par OpenSim : les features sont derivees directement des 3 points 3D + la verticale
(gravite), de facon invariante au sujet (longueurs, angle du coude, elevations vs vertical,
+ derivees + cumulatifs). Entraine LightGBM (trial #11) en LOSO et compare a l'Approche A.
Sortie: batch/metrics_3d_model.csv. biomech env.
"""
import os, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch")
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]
GROUPS = {"torque": ["elbow_moment"], "activations": ["act_" + m for m in FLEX],
          "forces": ["frc_" + m for m in FLEX], "fatigue": ["MF_" + m for m in FLEX]}
TRIAL11 = dict(n_estimators=975, num_leaves=15, learning_rate=0.196, subsample=0.605,
               colsample_bytree=0.911, min_child_samples=24, reg_lambda=8.19)
# ordre des marqueurs (1-indexe) -> Hip=1 ... RShoulder=17, RElbow=18, RWrist=19
MK = {"Hip": 1, "RShoulder": 17, "RElbow": 18, "RWrist": 19}


def read_trc(path):
    lines = open(path).read().splitlines()
    data = []
    for ln in lines:
        p = ln.split("\t")
        if len(p) > 6:
            try:
                float(p[0]); float(p[1]); data.append([float(x) if x.strip() else np.nan for x in p])
            except ValueError:
                continue
    d = np.array([r for r in data if len(r) >= 59])
    t = d[:, 1]
    def mk(name):
        c = 2 + (MK[name] - 1) * 3
        return d[:, c:c + 3]
    return t, mk("Hip"), mk("RShoulder"), mk("RElbow"), mk("RWrist")


def interp_to(t_src, arr, t_dst):
    return np.column_stack([np.interp(t_dst, t_src, arr[:, k]) for k in range(arr.shape[1])])


def angle(u, v):
    cs = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.arccos(np.clip(cs, -1, 1))


def build_features(subj):
    trc = glob.glob(os.path.join(B, subj, "pose2sim", "pose-3d", "*filt_butterworth.trc"))[0]
    t, hip, sh, el, wr = read_trc(trc)
    lab = pd.read_csv(os.path.join(B, subj, "labels_ml.csv")); tl = lab["time"].values
    # verticale (up) : axe ou l'epaule est au-dessus de la hanche
    up_vec = np.nanmean(sh, 0) - np.nanmean(hip, 0); ax = int(np.argmax(np.abs(up_vec)))
    up = np.zeros(3); up[ax] = np.sign(up_vec[ax])
    # re-echantillonne les 3 points 3D sur la grille temporelle des labels (100 Hz)
    sh, el, wr = interp_to(t, sh, tl), interp_to(t, el, tl), interp_to(t, wr, tl)
    dt = float(np.median(np.diff(tl)))
    SE = el - sh; WE = wr - el                       # vecteurs os (bras, avant-bras)
    ua_len = np.linalg.norm(SE, axis=1); fa_len = np.linalg.norm(WE, axis=1)
    sw = np.linalg.norm(wr - sh, axis=1)
    q_el = angle(-SE, WE)                            # angle interne du coude (rad)
    ua_elev = angle(SE, np.tile(up, (len(SE), 1)))   # elevation bras vs vertical
    fa_elev = angle(WE, np.tile(up, (len(WE), 1)))   # elevation avant-bras vs vertical
    SE_up = SE @ up; WE_up = WE @ up                 # composantes verticales (signe)

    def d2(x): v = np.gradient(x, dt); return v, np.gradient(v, dt)
    qd, qdd = d2(q_el); uad, _ = d2(ua_elev); fad, _ = d2(fa_elev)
    # anthropo (subject-aware) depuis les labels (deja calcule); longueurs aussi dispo en 3D
    hm = lab["humerus_mass"].values; fm = lab["forearm_mass"].values
    # proxy charge gravitaire 3D + cumulatifs (pour la fatigue)
    grav = (fm + 2.0) * fa_len * np.sin(fa_elev)
    cum_path = np.cumsum(np.abs(qd)) * dt
    cum_grav = np.cumsum(np.abs(grav)) * dt
    feats = dict(ua_len=ua_len, fa_len=fa_len, sw_dist=sw, q_el=q_el, ua_elev=ua_elev, fa_elev=fa_elev,
                 SE_up=SE_up, WE_up=WE_up, qd=qd, qdd=qdd, uad=uad, fad=fad, time=tl,
                 humerus_mass=hm, forearm_mass=fm, grav=grav, cum_path=cum_path, cum_grav=cum_grav)
    X = pd.DataFrame(feats); Y = lab[TARGETS].reset_index(drop=True)
    X["subj"] = subj; Y_ = pd.concat([X[["subj"]], Y], axis=1)
    return X, Y_


def main():
    subs = sorted([os.path.basename(p) for p in glob.glob(os.path.join(B, "s*"))
                   if os.path.isdir(p) and os.path.exists(os.path.join(p, "labels_ml.csv"))])
    Xs, Ys = [], []
    for s in subs:
        try:
            X, Y = build_features(s); Xs.append(X); Ys.append(Y)
        except Exception as e:
            print("skip", s, e)
    X = pd.concat(Xs, ignore_index=True); Y = pd.concat(Ys, ignore_index=True)
    FEAT = [c for c in X.columns if c != "subj"]
    SUBS = sorted(X.subj.unique())
    print("3D model: %d frames, %d sujets, %d features (depuis .trc)\nfeatures: %s\n" % (len(X), len(SUBS), len(FEAT), FEAT))

    acc = {t: [] for t in TARGETS}
    for held in SUBS:
        tr = X.subj != held; te = X.subj == held
        xs = StandardScaler().fit(X.loc[tr, FEAT]); ys = StandardScaler().fit(Y.loc[tr, TARGETS])
        m = MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=0, verbose=-1, **TRIAL11))
        m.fit(xs.transform(X.loc[tr, FEAT]), ys.transform(Y.loc[tr, TARGETS]))
        p = ys.inverse_transform(m.predict(xs.transform(X.loc[te, FEAT])))
        for j, t in enumerate(TARGETS):
            acc[t].append(r2_score(Y.loc[te, t].values, p[:, j]))
    r2 = {t: float(np.mean(v)) for t, v in acc.items()}
    g = {gn: float(np.mean([r2[t] for t in ts])) for gn, ts in GROUPS.items()}
    g["mean"] = float(np.mean(list(r2.values())))
    print("=== Modele 3D->biomeca (LOSO) ===")
    print("mean=%.3f | torque %.3f | activations %.3f | forces %.3f | fatigue %.3f" %
          (g["mean"], g["torque"], g["activations"], g["forces"], g["fatigue"]))
    print("\nComparaison Approche A (angles+features): mean=0.952 | torque .937 act .962 forces .964 fatigue .936")
    pd.DataFrame([g]).to_csv(os.path.join(B, "metrics_3d_model.csv"), index=False)
    print("\nwrote batch/metrics_3d_model.csv")


if __name__ == "__main__":
    main()
