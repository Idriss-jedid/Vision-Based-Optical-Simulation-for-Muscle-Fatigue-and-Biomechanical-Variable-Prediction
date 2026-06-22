# pyright: reportMissingImports=false
"""
BRANCHE 1 etendue : plusieurs architectures Transformer time-series sur le 3D BRUT (fenetre W),
+ anthropometrie. Implementations compactes en PyTorch (pas de lib lourde) :
  - PatchTST      (patches temporels -> transformer)
  - iTransformer  (tokens = variables/canaux ; attention sur les canaux)
  - TFT-lite      (LSTM + gating de covariables statiques + attention)
  - TST (vanilla, rappel)
LOSO, 8 sujets. Sortie: batch/ts_transformers_3d.csv. biomech env.
"""
import os, glob, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
torch.manual_seed(0); np.random.seed(0)

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch")
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]
TARGETS = ["elbow_moment"] + ["act_" + m for m in FLEX] + ["frc_" + m for m in FLEX] + ["MF_" + m for m in FLEX]
GROUPS = {"torque": ["elbow_moment"], "activations": ["act_" + m for m in FLEX],
          "forces": ["frc_" + m for m in FLEX], "fatigue": ["MF_" + m for m in FLEX]}
SCAL = ["humerus_len", "forearm_len", "humerus_mass", "forearm_mass", "time"]
MKIDX = {"RShoulder": 17, "RElbow": 18, "RWrist": 19}
W = 32; NJ = 9; NS = len(SCAL); O = len(TARGETS)


def read_trc_arm(path):
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
    return t, mk("RShoulder"), mk("RElbow"), mk("RWrist")


def subj_arrays(subj):
    trc = sorted(glob.glob(os.path.join(B, subj, "pose2sim", "pose-3d", "*filt_butterworth.trc")))[0]
    t, sh, el, wr = read_trc_arm(trc); lab = pd.read_csv(os.path.join(B, subj, "labels_ml.csv")); tl = lab["time"].values
    def itp(a): return np.column_stack([np.interp(tl, t, a[:, k]) for k in range(3)])
    sh, el, wr = itp(sh), itp(el), itp(wr); m = sh.mean(0)
    seq = np.concatenate([sh - m, el - m, wr - m], 1).astype("float32")
    n = len(seq); h = W // 2; pad = np.vstack([np.repeat(seq[:1], h, 0), seq, np.repeat(seq[-1:], h, 0)])
    win = np.stack([pad[i:i + W] for i in range(n)]).astype("float32")
    return win, lab[SCAL].values.astype("float32"), lab[TARGETS].values.astype("float32")


class PatchTST(nn.Module):
    def __init__(s, patch=8, dm=64, nl=3, nh=4):
        super().__init__(); s.p = patch; s.np = W // patch
        s.emb = nn.Linear(patch * NJ, dm); s.pos = nn.Parameter(torch.randn(1, s.np, dm) * 0.02)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh, dm * 2, dropout=0.1, batch_first=True), nl)
        s.head = nn.Sequential(nn.Linear(dm + NS, 128), nn.ReLU(), nn.Linear(128, O))
    def forward(s, x, a):
        b = x.shape[0]; z = x[:, :s.np * s.p].reshape(b, s.np, s.p * NJ)
        return s.head(torch.cat([s.tr(s.emb(z) + s.pos).mean(1), a], 1))


class iTransformer(nn.Module):
    """tokens = variables (9 canaux) ; chaque canal = serie de W -> token ; attention entre canaux."""
    def __init__(s, dm=64, nl=3, nh=4):
        super().__init__(); s.emb = nn.Linear(W, dm)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh, dm * 2, dropout=0.1, batch_first=True), nl)
        s.head = nn.Sequential(nn.Linear(dm * NJ + NS, 128), nn.ReLU(), nn.Linear(128, O))
    def forward(s, x, a):
        z = s.emb(x.transpose(1, 2))           # (B, NJ, dm) : un token par canal
        z = s.tr(z).reshape(x.shape[0], -1)    # flatten canaux
        return s.head(torch.cat([z, a], 1))


class TFTlite(nn.Module):
    """LSTM temporel + gating des covariables statiques (GRN-style) + attention pooling."""
    def __init__(s, dm=80):
        super().__init__(); s.lstm = nn.LSTM(NJ, dm, 2, batch_first=True, dropout=0.1)
        s.att = nn.Linear(dm, 1)
        s.stat = nn.Sequential(nn.Linear(NS, dm), nn.ReLU())
        s.gate = nn.Sequential(nn.Linear(dm * 2, dm), nn.Sigmoid())
        s.head = nn.Sequential(nn.Linear(dm, 64), nn.ReLU(), nn.Linear(64, O))
    def forward(s, x, a):
        o, _ = s.lstm(x)                       # (B,W,dm)
        w = torch.softmax(s.att(o), 1)         # attention pooling
        ctx = (w * o).sum(1)                   # (B,dm)
        st = s.stat(a); g = s.gate(torch.cat([ctx, st], 1))
        return s.head(g * ctx + (1 - g) * st)


class TST(nn.Module):
    def __init__(s, dm=64, nl=4, nh=4):
        super().__init__(); s.emb = nn.Linear(NJ, dm); s.pos = nn.Parameter(torch.randn(1, W, dm) * 0.02)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh, dm * 2, dropout=0.1, batch_first=True), nl)
        s.head = nn.Sequential(nn.Linear(dm + NS, 128), nn.ReLU(), nn.Linear(128, O))
    def forward(s, x, a): return s.head(torch.cat([s.tr(s.emb(x) + s.pos).mean(1), a], 1))


NETS = {"PatchTST": PatchTST, "iTransformer": iTransformer, "TFT-lite": TFTlite, "TST": TST}


def main():
    subs = sorted([os.path.basename(p) for p in glob.glob(os.path.join(B, "s*"))
                   if os.path.isdir(p) and os.path.exists(os.path.join(p, "labels_ml.csv"))])
    data = {s: subj_arrays(s) for s in subs}
    print("Multi-Transformer 3D : %d sujets, W=%d, (%d,%d)+scal(%d) -> %d\n" % (len(subs), W, W, NJ, NS, O))
    res = {}
    for name, Net in NETS.items():
        t0 = time.time(); acc = {t: [] for t in TARGETS}
        for held in subs:
            tr = [x for x in subs if x != held]; val = tr[-1]; fit = tr[:-1]
            xs = StandardScaler().fit(np.concatenate([data[s][0] for s in tr]).reshape(-1, NJ))
            sc = StandardScaler().fit(np.concatenate([data[s][1] for s in tr]))
            ys = StandardScaler().fit(np.concatenate([data[s][2] for s in tr]))
            def pk(ss):
                Xs = np.concatenate([data[s][0] for s in ss]); A = np.concatenate([data[s][1] for s in ss]); Y = np.concatenate([data[s][2] for s in ss])
                Xs = xs.transform(Xs.reshape(-1, NJ)).reshape(Xs.shape)
                return torch.tensor(Xs.astype("float32")), torch.tensor(sc.transform(A).astype("float32")), torch.tensor(ys.transform(Y).astype("float32"))
            Xtr, Atr, Ytr = pk(fit); Xva, Ava, Yva = pk([val]); Xte, Ate, _ = pk([held]); real = data[held][2]
            net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4); lf = nn.MSELoss()
            best, bad, state, bs = 1e9, 0, None, 256; idx = np.arange(len(Xtr))
            for ep in range(55):
                net.train(); np.random.shuffle(idx)
                for k in range(0, len(idx), bs):
                    b = idx[k:k + bs]; opt.zero_grad(); lf(net(Xtr[b], Atr[b]), Ytr[b]).backward(); opt.step()
                net.eval()
                with torch.no_grad(): vl = lf(net(Xva, Ava), Yva).item()
                if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 8: break
            if state: net.load_state_dict(state)
            net.eval()
            with torch.no_grad(): pred = ys.inverse_transform(net(Xte, Ate).numpy())
            for j, t in enumerate(TARGETS): acc[t].append(r2_score(real[:, j], pred[:, j]))
        r2 = {t: float(np.mean(v)) for t, v in acc.items()}
        g = {gn: float(np.mean([r2[t] for t in ts])) for gn, ts in GROUPS.items()}; g["mean"] = float(np.mean(list(r2.values()))); res[name] = g
        print("%-13s mean=%.3f | torque %.3f act %.3f forces %.3f fatigue %.3f (%.0fs)" %
              (name, g["mean"], g["torque"], g["activations"], g["forces"], g["fatigue"], time.time() - t0))
    rep = pd.DataFrame(res).T[["mean", "torque", "activations", "forces", "fatigue"]].round(3).sort_values("mean", ascending=False)
    rep.to_csv(os.path.join(B, "ts_transformers_3d.csv"))
    print("\n=== Multi-Transformer 3D (LOSO) ===\n", rep)
    print("Ref: GBM+FE 3D=0.867 | Approche A=0.952")
    print("wrote ts_transformers_3d.csv")


if __name__ == "__main__":
    main()
