# pyright: reportMissingImports=false
"""
END-TO-END Temporal Transformer (Version 1) :
  INPUT  = fenetre de W frames de 3D BRUT (RShoulder/RElbow/RWrist = 9 coords) + anthropometrie
  MODELE = embedding lineaire -> positional encoding -> Transformer encoder -> mean-pool
           -> fusion avec anthropometrie -> MLP head
  OUTPUT = 13 cibles (torque/forces/activations/fatigue)  [memes labels OpenSim]
Le modele apprend ses propres representations temporelles (PAS d'angles imposes). LOSO, 8 sujets.
Comparaison: Approche A (features+angles)=0.95 ; LightGBM 3D-features=0.84. Sortie: batch/metrics_tsT3d.csv. biomech env.
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
SCAL = ["humerus_len", "forearm_len", "humerus_mass", "forearm_mass", "time"]  # anthropo + horloge
MKIDX = {"RShoulder": 17, "RElbow": 18, "RWrist": 19}
W = 31              # fenetre centree (t-15..t+15)
NJ = 9              # 3 joints x XYZ


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


def subject_arrays(subj):
    trc = sorted(glob.glob(os.path.join(B, subj, "pose2sim", "pose-3d", "*filt_butterworth.trc")))[0]
    t, sh, el, wr = read_trc_arm(trc)
    lab = pd.read_csv(os.path.join(B, subj, "labels_ml.csv")); tl = lab["time"].values
    def itp(a): return np.column_stack([np.interp(tl, t, a[:, k]) for k in range(3)])
    sh, el, wr = itp(sh), itp(el), itp(wr)
    sh0 = sh - sh.mean(0)                              # centre sur l'epaule moyenne (invariant a la position debout)
    el0, wr0 = el - sh.mean(0), wr - sh.mean(0)
    seq = np.concatenate([sh0, el0, wr0], 1).astype("float32")   # (N,9) 3D brut centre
    scal = lab[SCAL].values.astype("float32")
    Y = lab[TARGETS].values.astype("float32")
    return seq, scal, Y


def windows(seq):
    n = len(seq); h = W // 2; out = np.zeros((n, W, NJ), "float32")
    pad = np.vstack([np.repeat(seq[:1], h, 0), seq, np.repeat(seq[-1:], h, 0)])
    for i in range(n): out[i] = pad[i:i + W]
    return out


class TSTransformer(nn.Module):
    def __init__(s, dm=64, nl=4, nh=4, nscal=len(SCAL), o=len(TARGETS)):
        super().__init__()
        s.emb = nn.Linear(NJ, dm); s.pos = nn.Parameter(torch.randn(1, W, dm) * 0.02)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh, dm * 2, dropout=0.1, batch_first=True), nl)
        s.head = nn.Sequential(nn.Linear(dm + nscal, 128), nn.ReLU(), nn.Dropout(0.1),
                               nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, o))
    def forward(s, x, a):
        z = s.tr(s.emb(x) + s.pos).mean(1)            # motion embedding (B,dm)
        return s.head(torch.cat([z, a], 1))


def main():
    subs = sorted([os.path.basename(p) for p in glob.glob(os.path.join(B, "s*"))
                   if os.path.isdir(p) and os.path.exists(os.path.join(p, "labels_ml.csv"))])
    data = {}
    for s in subs:
        seq, scal, Y = subject_arrays(s); data[s] = (windows(seq), scal, Y)
    print("TS-Transformer 3D : %d sujets, fenetre W=%d, input (%d,%d)+scal(%d) -> %d cibles\n" %
          (len(subs), W, W, NJ, len(SCAL), len(TARGETS)))

    acc = {t: [] for t in TARGETS}
    for held in subs:
        tr = [x for x in subs if x != held]; val = tr[-1]; fit = tr[:-1]
        # scalers (fit sur train)
        Xseq_tr = np.concatenate([data[s][0] for s in tr]); xs = StandardScaler().fit(Xseq_tr.reshape(-1, NJ))
        sc = StandardScaler().fit(np.concatenate([data[s][1] for s in tr]))
        ys = StandardScaler().fit(np.concatenate([data[s][2] for s in tr]))
        def pk(subset):
            Xs = np.concatenate([data[s][0] for s in subset]); A = np.concatenate([data[s][1] for s in subset])
            Y = np.concatenate([data[s][2] for s in subset])
            Xs = xs.transform(Xs.reshape(-1, NJ)).reshape(Xs.shape)
            return (torch.tensor(Xs.astype("float32")), torch.tensor(sc.transform(A).astype("float32")),
                    torch.tensor(ys.transform(Y).astype("float32")))
        Xtr, Atr, Ytr = pk(fit); Xva, Ava, Yva = pk([val]); Xte, Ate, _ = pk([held])
        real = data[held][2]
        net = TSTransformer(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4); lf = nn.MSELoss()
        best, bad, state, bs = 1e9, 0, None, 256; idx = np.arange(len(Xtr))
        for ep in range(60):
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
        g = {gn: np.mean([np.mean(acc[t]) for t in ts]) for gn, ts in GROUPS.items()}
        print("  held %s done | torque %.2f act %.2f forces %.2f fatigue %.2f" %
              (held, g["torque"], g["activations"], g["forces"], g["fatigue"]))

    r2 = {t: float(np.mean(v)) for t, v in acc.items()}
    g = {gn: float(np.mean([r2[t] for t in ts])) for gn, ts in GROUPS.items()}; g["mean"] = float(np.mean(list(r2.values())))
    print("\n=== TS-Transformer 3D->biomeca (LOSO) ===")
    print("mean=%.3f | torque %.3f | activations %.3f | forces %.3f | fatigue %.3f" %
          (g["mean"], g["torque"], g["activations"], g["forces"], g["fatigue"]))
    print("\nReferences: Approche A (angles+features)=0.952 | LightGBM 3D-features=0.842")
    pd.DataFrame([g]).to_csv(os.path.join(B, "metrics_tsT3d.csv"), index=False)
    print("wrote batch/metrics_tsT3d.csv")


if __name__ == "__main__":
    main()
