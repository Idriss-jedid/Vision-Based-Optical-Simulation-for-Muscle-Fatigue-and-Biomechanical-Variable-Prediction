# pyright: reportMissingImports=false
"""
IDEE 2 : 2D keypoints (4 cameras) -> AI -> Biomecanique.
Input = pour chaque camera, les 2D (u,v,conf) de RShoulder/RElbow/RWrist, normalises (relatifs a
l'epaule, / taille image) -> 4 cams x [elbow,wrist relatifs + 3 conf] + anthropometrie + temps +
cumul. LightGBM LOSO, 8 sujets. Compare a 3D (0.904) et .mot+.osim (0.952). biomech env.
"""
import os, glob, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch"); IMG = 900.0
KP = {"RShoulder": 6, "RElbow": 8, "RWrist": 10}  # HALPE_26
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]
GROUPS = {"torque": ["elbow_moment"], "activations": ["act_" + m for m in FLEX],
          "forces": ["frc_" + m for m in FLEX], "fatigue": ["MF_" + m for m in FLEX]}


def load_cam_2d(cam_dir):
    """retourne un tableau (n_frames, 9) : sh(u,v,c), el(u,v,c), wr(u,v,c)."""
    files = sorted(glob.glob(os.path.join(cam_dir, "*.json")))
    out = []
    for f in files:
        try:
            ppl = json.load(open(f)).get("people", [])
            if not ppl: out.append([np.nan] * 9); continue
            kp = np.array(ppl[0]["pose_keypoints_2d"]).reshape(-1, 3)
            row = []
            for nm in ["RShoulder", "RElbow", "RWrist"]: row += list(kp[KP[nm]])
            out.append(row)
        except Exception:
            out.append([np.nan] * 9)
    return np.array(out)


def subject_2d(subj):
    pose = os.path.join(B, subj, "pose2sim", "pose")
    cams = sorted(glob.glob(os.path.join(pose, "cam_*_json")))
    lab = pd.read_csv(os.path.join(B, subj, "labels_ml.csv")); tl = lab["time"].values
    feats = {}
    for ci, cam in enumerate(cams):
        d = load_cam_2d(cam); n = len(d); t2d = np.arange(n) / 50.0  # 50 fps
        sh, el, wr = d[:, 0:3], d[:, 3:6], d[:, 6:9]
        # relatif a l'epaule, normalise / taille image (invariant a la translation)
        elu = (el[:, 0] - sh[:, 0]) / IMG; elv = (el[:, 1] - sh[:, 1]) / IMG
        wru = (wr[:, 0] - sh[:, 0]) / IMG; wrv = (wr[:, 1] - sh[:, 1]) / IMG
        for nm, arr in [("c%d_elu" % ci, elu), ("c%d_elv" % ci, elv), ("c%d_wru" % ci, wru), ("c%d_wrv" % ci, wrv),
                        ("c%d_cs" % ci, sh[:, 2]), ("c%d_ce" % ci, el[:, 2]), ("c%d_cw" % ci, wr[:, 2])]:
            a = np.nan_to_num(arr, nan=0.0)
            feats[nm] = np.interp(tl, t2d, a)
    X = pd.DataFrame(feats)
    # anthropometrie + temps + cumul (pour la fatigue)
    X["humerus_mass"] = lab["humerus_mass"].values; X["forearm_mass"] = lab["forearm_mass"].values
    X["time"] = tl
    # proxy de mouvement 2D cumule (moyenne sur cams de |d(wrist-shoulder)|)
    disp = np.zeros(len(tl))
    for ci in range(len(cams)):
        du = np.gradient(X["c%d_wru" % ci].values); dv = np.gradient(X["c%d_wrv" % ci].values)
        disp += np.sqrt(du ** 2 + dv ** 2)
    X["cum_2d"] = np.cumsum(disp)
    X["subj"] = subj; Y = lab[TARGETS].reset_index(drop=True); Y["subj"] = subj
    return X, Y


def main():
    subs = sorted([os.path.basename(p) for p in glob.glob(os.path.join(B, "s*"))
                   if os.path.isdir(p) and os.path.exists(os.path.join(p, "labels_ml.csv"))])
    XS, YS = [], []
    for s in subs:
        try:
            X, Y = subject_2d(s); XS.append(X); YS.append(Y); print("loaded", s, X.shape)
        except Exception as e:
            print("skip", s, e)
    X = pd.concat(XS, ignore_index=True); Y = pd.concat(YS, ignore_index=True)
    FEAT = [c for c in X.columns if c != "subj"]; SUBS = sorted(X.subj.unique())
    X[FEAT] = X[FEAT].replace([np.inf, -np.inf], np.nan).fillna(0.0)  # garde anti-NaN/inf
    print("\n2D model: %d frames, %d sujets, %d features\n" % (len(X), len(SUBS), len(FEAT)), flush=True)

    acc = {t: [] for t in TARGETS}
    for held in SUBS:
        tr = X.subj != held; te = X.subj == held
        xs = StandardScaler().fit(X.loc[tr, FEAT]); ys = StandardScaler().fit(Y.loc[tr, TARGETS])
        m = MultiOutputRegressor(LGBMRegressor(n_estimators=600, num_leaves=31, learning_rate=0.05, n_jobs=-1, random_state=0, verbose=-1))
        m.fit(xs.transform(X.loc[tr, FEAT]), ys.transform(Y.loc[tr, TARGETS]))
        p = ys.inverse_transform(m.predict(xs.transform(X.loc[te, FEAT])))
        for j, t in enumerate(TARGETS): acc[t].append(r2_score(Y.loc[te, t].values, p[:, j]))
        print("  fold %s done" % held, flush=True)
    r2 = {t: float(np.mean(v)) for t, v in acc.items()}
    g = {gn: float(np.mean([r2[t] for t in ts])) for gn, ts in GROUPS.items()}; g["mean"] = float(np.mean(list(r2.values())))
    print("=== 2D keypoints -> biomeca (LOSO) ===")
    print("mean=%.3f | torque %.3f | activations %.3f | forces %.3f | fatigue %.3f" %
          (g["mean"], g["torque"], g["activations"], g["forces"], g["fatigue"]))
    print("\nComparaison: 3D vision-only=0.904 | .mot+.osim (Approche A)=0.952")
    pd.DataFrame([g]).to_csv(os.path.join(B, "metrics_2d_model.csv"), index=False)
    print("wrote metrics_2d_model.csv")


if __name__ == "__main__":
    main()
