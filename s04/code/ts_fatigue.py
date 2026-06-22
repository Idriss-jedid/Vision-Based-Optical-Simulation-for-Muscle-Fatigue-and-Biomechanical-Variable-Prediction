# pyright: reportMissingImports=false
"""
FATIGUE (MF) en time-series — modèles séquentiels (PyTorch, CPU) en LEAVE-ONE-SUBJECT-OUT :
  LSTM, PatchTST (transformer patché spécifique séries temporelles), TST (transformer encoder).
La fatigue est CUMULATIVE -> on lui donne une fenêtre de W frames passées + les features
cumulatifs (cum_path_el, cum_grav_imp, time). Comparé à LightGBM (référence tabulaire).
Cibles : MF_BIClong/BICshort/BRA/BRD_hand. Sortie : batch/ts_fatigue.csv. biomech env.
"""
import os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

torch.manual_seed(0); np.random.seed(0)
ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
DATA = os.path.join(ROOT, "batch", "ml_dataset_A.csv"); OUT = os.path.join(ROOT, "batch")
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
MF = ["MF_" + m for m in FLEX]
FEATS = ["q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el", "time",
         "humerus_mass", "forearm_mass", "humerus_len", "forearm_len",
         "sin_qel", "cos_qel", "abs_qd_el", "grav_load", "cum_path_el", "cum_grav_imp"]
W = 48; F = len(FEATS); O = len(MF)


def engineer(df):
    g = np.pi / 180.0
    df["sin_qel"] = np.sin(df.q_el * g); df["cos_qel"] = np.cos(df.q_el * g)
    df["abs_qd_el"] = df.qd_el.abs()
    df["grav_load"] = (df.forearm_mass + 2.0) * df.forearm_len * np.sin((df.q_sh + df.q_el) * g)
    cp, cg = [], []
    for s, sub in df.groupby("subj", sort=False):
        dt = np.median(np.diff(sub.time.values)) if len(sub) > 1 else 0.01
        cp.append(np.cumsum(sub.abs_qd_el.values) * dt)
        cg.append(np.cumsum(np.abs(sub.grav_load.values)) * dt)
    df["cum_path_el"] = np.concatenate(cp); df["cum_grav_imp"] = np.concatenate(cg)
    return df


df = engineer(pd.read_csv(DATA)); SUBS = sorted(df.subj.unique())


def make_windows(sub, xs, ys):
    X = xs.transform(sub[FEATS].values).astype("float32"); Y = ys.transform(sub[MF].values).astype("float32")
    xs_, ys_ = [], []
    for i in range(len(X)):
        s = max(0, i - W + 1); w = X[s:i + 1]
        if len(w) < W: w = np.vstack([np.repeat(w[:1], W - len(w), 0), w])
        xs_.append(w); ys_.append(Y[i])
    return np.asarray(xs_, "float32"), np.asarray(ys_, "float32")


class LSTMNet(nn.Module):
    def __init__(s):
        super().__init__(); s.lstm = nn.LSTM(F, 96, 2, batch_first=True, dropout=0.1); s.fc = nn.Linear(96, O)
    def forward(s, x): o, _ = s.lstm(x); return s.fc(o[:, -1])


class PatchTST(nn.Module):
    """patch la fenêtre temporelle -> tokens -> Transformer encoder -> tête (séries temporelles)."""
    def __init__(s, patch=8, dm=96):
        super().__init__(); s.p = patch; s.np = W // patch
        s.emb = nn.Linear(patch * F, dm); s.pos = nn.Parameter(torch.randn(1, s.np, dm) * 0.02)
        enc = nn.TransformerEncoderLayer(dm, 4, 192, dropout=0.1, batch_first=True)
        s.tr = nn.TransformerEncoder(enc, 3); s.fc = nn.Linear(dm, O)
    def forward(s, x):
        b = x.shape[0]; z = x[:, :s.np * s.p].reshape(b, s.np, s.p * F)
        z = s.emb(z) + s.pos; return s.fc(s.tr(z).mean(1))


class TST(nn.Module):
    """transformer encoder par frame (pas de patch)."""
    def __init__(s, dm=80):
        super().__init__(); s.emb = nn.Linear(F, dm); s.pos = nn.Parameter(torch.randn(1, W, dm) * 0.02)
        enc = nn.TransformerEncoderLayer(dm, 4, 160, dropout=0.1, batch_first=True)
        s.tr = nn.TransformerEncoder(enc, 2); s.fc = nn.Linear(dm, O)
    def forward(s, x): return s.fc(s.tr(s.emb(x) + s.pos).mean(1))


NETS = {"LSTM": LSTMNet, "PatchTST": PatchTST, "TST": TST}


def train_net(Net, held):
    tr = [x for x in SUBS if x != held]; val = tr[-1]; fit = tr[:-1]
    xs = StandardScaler().fit(df[df.subj.isin(tr)][FEATS]); ys = StandardScaler().fit(df[df.subj.isin(tr)][MF])
    def pack(subs):
        A, B = [], []
        for su in subs:
            a, b = make_windows(df[df.subj == su], xs, ys); A.append(a); B.append(b)
        return torch.tensor(np.concatenate(A)), torch.tensor(np.concatenate(B))
    Xtr, Ytr = pack(fit); Xva, Yva = pack([val]); Xte, _ = pack([held]); real = df[df.subj == held][MF].values
    net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4); lf = nn.MSELoss()
    best, bad, bs = 1e9, 0, 256; idx = np.arange(len(Xtr)); state = None
    for ep in range(80):
        net.train(); np.random.shuffle(idx)
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]; opt.zero_grad(); lf(net(Xtr[b]), Ytr[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = lf(net(Xva), Yva).item()
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10: break
    if state: net.load_state_dict(state)
    net.eval()
    with torch.no_grad(): pred = ys.inverse_transform(net(Xte).numpy())
    return {t: r2_score(real[:, j], pred[:, j]) for j, t in enumerate(MF)}


def lgbm_fatigue():
    acc = {t: [] for t in MF}
    for held in SUBS:
        trd, ted = df[df.subj != held], df[df.subj == held]
        xs = StandardScaler().fit(trd[FEATS]); ys = StandardScaler().fit(trd[MF])
        m = MultiOutputRegressor(LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.05,
                                               n_jobs=-1, random_state=0, verbose=-1))
        m.fit(xs.transform(trd[FEATS]), ys.transform(trd[MF]))
        p = ys.inverse_transform(m.predict(xs.transform(ted[FEATS])))
        for j, t in enumerate(MF): acc[t].append(r2_score(ted[t].values, p[:, j]))
    return {t: float(np.mean(v)) for t, v in acc.items()}


def main():
    res = {}
    print("FATIGUE time-series (LOSO, 8 sujets) ; fenêtre W=%d, %d features\n" % (W, F))
    r = lgbm_fatigue(); res["LightGBM"] = r
    print("%-10s mean=%.3f | %s" % ("LightGBM", np.mean(list(r.values())),
          " ".join("%s=%.2f" % (t.replace("MF_", ""), r[t]) for t in MF)))
    for name, Net in NETS.items():
        t0 = time.time(); acc = {t: [] for t in MF}
        for held in SUBS:
            rr = train_net(Net, held)
            for t in MF: acc[t].append(rr[t])
        r = {t: float(np.mean(v)) for t, v in acc.items()}; res[name] = r
        print("%-10s mean=%.3f | %s (%.0fs)" % (name, np.mean(list(r.values())),
              " ".join("%s=%.2f" % (t.replace("MF_", ""), r[t]) for t in MF), time.time() - t0))
    rep = pd.DataFrame(res).T; rep["mean"] = rep.mean(1)
    rep.round(3).sort_values("mean", ascending=False).to_csv(os.path.join(OUT, "ts_fatigue.csv"))
    print("\n=== FATIGUE : classement (R² LOSO) ===\n", rep.round(3).sort_values("mean", ascending=False))
    print("\nwrote batch/ts_fatigue.csv")


if __name__ == "__main__":
    main()
