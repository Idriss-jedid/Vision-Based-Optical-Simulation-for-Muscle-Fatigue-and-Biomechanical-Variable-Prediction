import research_3d_gbm as R
import shap, numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
xs = StandardScaler().fit(R.X[R.FEAT]); Xall = xs.transform(R.X[R.FEAT])
grp = {"torque":["elbow_moment"],"activations":["act_"+m for m in R.FLEX],
       "forces":["frc_"+m for m in R.FLEX],"fatigue":["MF_"+m for m in R.FLEX]}
cols = {g: np.zeros(len(R.FEAT)) for g in grp}
for t in R.TARGETS:
    ys = StandardScaler().fit(R.Y[[t]]); yv = ys.transform(R.Y[[t]]).ravel()
    m = LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05, n_jobs=-1, random_state=0, verbose=-1).fit(Xall, yv)
    ma = np.abs(shap.TreeExplainer(m).shap_values(Xall)).mean(0)
    for g,ts in grp.items():
        if t in ts: cols[g]+=ma/len(ts)
df = pd.DataFrame(cols, index=R.FEAT); df["overall"]=df.mean(1)
df = df.sort_values("overall", ascending=False)
df.round(4).to_csv(os.path.join(R.B,"xai_3d.csv"))
print("=== XAI 3D (SHAP mean|val|, top 12) ===")
print(df.head(12).round(3).to_string())
