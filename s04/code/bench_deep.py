# pyright: reportMissingImports=false
"""
APPROCHE A — BENCHMARK deep learning (séquentiel) en LEAVE-ONE-SUBJECT-OUT.
Modèles (PyTorch, CPU) : ANN(MLP), 1D-CNN, LSTM, CNN+LSTM, Transformer.
Les modèles séquentiels prennent une fenêtre de W frames passées -> utile surtout pour la
FATIGUE (cumulative). Fenêtres construites par sujet (pas de fuite inter-sujet).
INPUT 11 features -> 13 labels. Sortie : batch/bench_deep.csv. biomech env (torch).
"""
import os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"
ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
DATA = os.path.join(ROOT, "batch", "ml_dataset_A.csv"); OUT = os.path.join(ROOT, "batch")
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
X_COLS = ["q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el", "time",
          "humerus_mass", "forearm_mass", "humerus_len", "forearm_len"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]
GROUPS = {"torque": ["elbow_moment"], "activations": ["act_" + m for m in FLEX],
          "forces": ["frc_" + m for m in FLEX], "fatigue": ["MF_" + m for m in FLEX]}
W = 32; F = len(X_COLS); O = len(TARGETS)
df = pd.read_csv(DATA); SUBS = sorted(df["subj"].unique())


def windows(sub_df, xs, ys, seq):
    """fenêtres glissantes par sujet. seq=True -> X [n,W,F] ; seq=False -> X [n,F] (dernier frame)."""
    X = xs.transform(sub_df[X_COLS].values); Y = ys.transform(sub_df[TARGETS].values)
    n = len(X)
    if not seq:
        return X.astype("float32"), Y.astype("float32")
    xs_, ys_ = [], []
    for i in range(n):
        s = max(0, i - W + 1); win = X[s:i + 1]
        if len(win) < W:  # pad au début
            win = np.vstack([np.repeat(win[:1], W - len(win), 0), win])
        xs_.append(win); ys_.append(Y[i])
    return np.asarray(xs_, "float32"), np.asarray(ys_, "float32")


# ---------- architectures ----------
class ANN(nn.Module):
    def __init__(s):
        super().__init__(); s.seq = False
        s.net = nn.Sequential(nn.Linear(F, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, O))
    def forward(s, x): return s.net(x)

class CNN1D(nn.Module):
    def __init__(s):
        super().__init__(); s.seq = True
        s.c = nn.Sequential(nn.Conv1d(F, 64, 5, padding=2), nn.ReLU(),
                            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        s.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, O))
    def forward(s, x): return s.fc(s.c(x.transpose(1, 2)).squeeze(-1))

class LSTMNet(nn.Module):
    def __init__(s):
        super().__init__(); s.seq = True
        s.lstm = nn.LSTM(F, 96, num_layers=2, batch_first=True, dropout=0.1)
        s.fc = nn.Linear(96, O)
    def forward(s, x): out, _ = s.lstm(x); return s.fc(out[:, -1])

class CNNLSTM(nn.Module):
    def __init__(s):
        super().__init__(); s.seq = True
        s.c = nn.Sequential(nn.Conv1d(F, 64, 3, padding=1), nn.ReLU())
        s.lstm = nn.LSTM(64, 96, batch_first=True); s.fc = nn.Linear(96, O)
    def forward(s, x):
        z = s.c(x.transpose(1, 2)).transpose(1, 2); out, _ = s.lstm(z); return s.fc(out[:, -1])

class TransNet(nn.Module):
    def __init__(s):
        super().__init__(); s.seq = True
        s.emb = nn.Linear(F, 64); s.pos = nn.Parameter(torch.randn(1, W, 64) * 0.02)
        enc = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1, batch_first=True)
        s.tr = nn.TransformerEncoder(enc, 2); s.fc = nn.Linear(64, O)
    def forward(s, x): z = s.emb(x) + s.pos; return s.fc(s.tr(z).mean(1))


MODELS = {"ANN": ANN, "1D-CNN": CNN1D, "LSTM": LSTMNet, "CNN+LSTM": CNNLSTM, "Transformer": TransNet}


def train_eval(Mclass, held):
    seq = Mclass().seq
    tr_subs = [x for x in SUBS if x != held]; val = tr_subs[-1]; fit_subs = tr_subs[:-1]
    xs = StandardScaler().fit(df[df.subj.isin(tr_subs)][X_COLS])
    ys = StandardScaler().fit(df[df.subj.isin(tr_subs)][TARGETS])
    def pack(subs):
        Xs, Ys = [], []
        for su in subs:
            x, y = windows(df[df.subj == su], xs, ys, seq); Xs.append(x); Ys.append(y)
        return torch.tensor(np.concatenate(Xs)), torch.tensor(np.concatenate(Ys))
    Xtr, Ytr = pack(fit_subs); Xva, Yva = pack([val])
    Xte, _ = pack([held]); Yte_real = df[df.subj == held][TARGETS].values
    net = Mclass().to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss(); best = 1e9; bad = 0; best_state = None
    idx = np.arange(len(Xtr)); bs = 256
    for ep in range(60):
        net.train(); np.random.shuffle(idx)
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]; opt.zero_grad()
            l = lossf(net(Xtr[b]), Ytr[b]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xva), Yva).item()
        if vl < best - 1e-4: best = vl; bad = 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
    if best_state: net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = ys.inverse_transform(net(Xte).numpy())
    return {t: r2_score(Yte_real[:, j], pred[:, j]) for j, t in enumerate(TARGETS)}


def main():
    results = {}
    for name, M in MODELS.items():
        t0 = time.time(); acc = {t: [] for t in TARGETS}
        for held in SUBS:
            r2 = train_eval(M, held)
            for t in TARGETS: acc[t].append(r2[t])
        r2m = {t: float(np.mean(v)) for t, v in acc.items()}
        g = {gn: float(np.mean([r2m[t] for t in ts])) for gn, ts in GROUPS.items()}
        g["mean"] = float(np.mean(list(r2m.values()))); results[name] = g
        print("%-12s mean R2=%.3f | torque %.3f act %.3f forces %.3f fatigue %.3f (%.0fs)" %
              (name, g["mean"], g["torque"], g["activations"], g["forces"], g["fatigue"], time.time() - t0))

    rep = pd.DataFrame(results).T[["mean", "torque", "activations", "forces", "fatigue"]].round(3)
    rep = rep.sort_values("mean", ascending=False); rep.to_csv(os.path.join(OUT, "bench_deep.csv"))
    print("\n=== CLASSEMENT deep (R² moyen LOSO) ===\n", rep)
    print("\nwrote batch/bench_deep.csv")


if __name__ == "__main__":
    main()
