# pyright: reportMissingImports=false
"""
FATIGUE time-series — TUNING POUSSÉ (PyTorch, CPU) :
  - feature engineering étendu (cinématique + anthropo + cumulatifs + interactions)
  - OPTUNA (TPE + HyperbandPruner = BOHB-like) sur LSTM et PatchTST : tune hidden/layers/
    dropout/lr/weight_decay ET la taille de fenêtre W. Objectif rapide = R² fatigue sur 1
    sujet de validation, avec pruning par epoch ; puis ré-évaluation en FULL LOSO (8 folds).
  - XAI = permutation importance (occlusion par feature) sur le meilleur modèle.
Référence : LightGBM (features cumulatifs). Sortie : batch/ts_fatigue_tuned.csv + xai_ts.csv.
"""
import os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.manual_seed(0); np.random.seed(0)

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
DATA = os.path.join(ROOT, "batch", "ml_dataset_A.csv"); OUT = os.path.join(ROOT, "batch")
FLEX = ["BIClong", "BICshort", "BRA", "BRD_hand"]; MF = ["MF_" + m for m in FLEX]
FEATS = ["q_sh", "q_el", "qd_sh", "qd_el", "qdd_sh", "qdd_el", "time",
         "humerus_mass", "forearm_mass", "humerus_len", "forearm_len",
         "sin_qel", "cos_qel", "sin_qsh", "cos_qsh", "abs_qd_el", "abs_qdd_el",
         "qd_el2", "grav_load", "qel_x_fmass", "cum_path_el", "cum_grav_imp"]
F = len(FEATS); O = len(MF)


def engineer(df):
    g = np.pi / 180.0
    df["sin_qel"] = np.sin(df.q_el * g); df["cos_qel"] = np.cos(df.q_el * g)
    df["sin_qsh"] = np.sin(df.q_sh * g); df["cos_qsh"] = np.cos(df.q_sh * g)
    df["abs_qd_el"] = df.qd_el.abs(); df["abs_qdd_el"] = df.qdd_el.abs(); df["qd_el2"] = df.qd_el ** 2
    df["grav_load"] = (df.forearm_mass + 2.0) * df.forearm_len * np.sin((df.q_sh + df.q_el) * g)
    df["qel_x_fmass"] = df.q_el * df.forearm_mass
    cp, cg = [], []
    for s, sub in df.groupby("subj", sort=False):
        dt = np.median(np.diff(sub.time.values)) if len(sub) > 1 else 0.01
        cp.append(np.cumsum(sub.abs_qd_el.values) * dt); cg.append(np.cumsum(np.abs(sub.grav_load.values)) * dt)
    df["cum_path_el"] = np.concatenate(cp); df["cum_grav_imp"] = np.concatenate(cg)
    return df


df = engineer(pd.read_csv(DATA)); SUBS = sorted(df.subj.unique())


def windows(sub, xs, ys, W):
    X = xs.transform(sub[FEATS].values).astype("float32"); Y = ys.transform(sub[MF].values).astype("float32")
    out = []
    for i in range(len(X)):
        s = max(0, i - W + 1); w = X[s:i + 1]
        if len(w) < W: w = np.vstack([np.repeat(w[:1], W - len(w), 0), w])
        out.append(w)
    return np.asarray(out, "float32"), Y


def pack(subs, xs, ys, W):
    A, B = [], []
    for su in subs:
        a, b = windows(df[df.subj == su], xs, ys, W); A.append(a); B.append(b)
    return torch.tensor(np.concatenate(A)), torch.tensor(np.concatenate(B))


class LSTMNet(nn.Module):
    def __init__(s, h=96, nl=2, drop=0.1):
        super().__init__(); s.lstm = nn.LSTM(F, h, nl, batch_first=True, dropout=drop if nl > 1 else 0)
        s.fc = nn.Linear(h, O)
    def forward(s, x): o, _ = s.lstm(x); return s.fc(o[:, -1])


class PatchTST(nn.Module):
    def __init__(s, W, patch=8, dm=96, nl=3, nh=4, drop=0.1):
        super().__init__(); s.p = patch; s.np = max(1, W // patch); s.W = s.np * patch
        s.emb = nn.Linear(patch * F, dm); s.pos = nn.Parameter(torch.randn(1, s.np, dm) * 0.02)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh, dm * 2, dropout=drop, batch_first=True), nl)
        s.fc = nn.Linear(dm, O)
    def forward(s, x):
        b = x.shape[0]; z = x[:, :s.W].reshape(b, s.np, s.p * F); return s.fc(s.tr(s.emb(z) + s.pos).mean(1))


def train(build, W, fit_subs, val_sub, xs, ys, epochs=70, trial=None):
    Xtr, Ytr = pack(fit_subs, xs, ys, W); Xva, Yva = pack([val_sub], xs, ys, W)
    real_va = df[df.subj == val_sub][MF].values
    net = build(); opt = torch.optim.Adam(net.parameters(), getattr(net, "_lr", 1e-3), weight_decay=getattr(net, "_wd", 1e-4))
    lf = nn.MSELoss(); best, bad, state, bs = -1e9, 0, None, 256; idx = np.arange(len(Xtr))
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]; opt.zero_grad(); lf(net(Xtr[b]), Ytr[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = ys.inverse_transform(net(Xva).numpy()); r2 = np.mean([r2_score(real_va[:, j], p[:, j]) for j in range(O)])
        if r2 > best + 1e-4: best, bad, state = r2, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
        if trial is not None:
            trial.report(r2, ep)
            if trial.should_prune(): raise optuna.TrialPruned()
    if state: net.load_state_dict(state)
    return net, best


def scalers(tr_subs):
    return (StandardScaler().fit(df[df.subj.isin(tr_subs)][FEATS]),
            StandardScaler().fit(df[df.subj.isin(tr_subs)][MF]))


def tune(kind, n_trials=18):
    val = "s09"; fit = [x for x in SUBS if x not in (val,)]  # tuning : train sur 7, val sur s09
    xs, ys = scalers(fit + [val])
    def objective(tr):
        W = tr.suggest_categorical("W", [24, 48, 72, 96])
        lr = tr.suggest_float("lr", 3e-4, 5e-3, log=True); wd = tr.suggest_float("wd", 1e-6, 1e-3, log=True)
        drop = tr.suggest_float("drop", 0.0, 0.3)
        if kind == "LSTM":
            h = tr.suggest_categorical("h", [64, 96, 128]); nl = tr.suggest_int("nl", 1, 3)
            def build(): m = LSTMNet(h, nl, drop); m._lr, m._wd = lr, wd; return m
        else:
            patch = tr.suggest_categorical("patch", [6, 8, 12]); dm = tr.suggest_categorical("dm", [64, 96])
            nl = tr.suggest_int("nl", 2, 4); nh = tr.suggest_categorical("nh", [4, 8])
            def build(): m = PatchTST(W, patch, dm, nl, nh, drop); m._lr, m._wd = lr, wd; return m
        _, r2 = train(build, W, [x for x in fit if x != "s05"], "s05", xs, ys, epochs=50, trial=tr)
        return r2
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                             pruner=optuna.pruners.HyperbandPruner())
    st.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return st.best_params


def full_loso(kind, p):
    W = p["W"]; acc = {t: [] for t in MF}; nets = []
    for held in SUBS:
        tr = [x for x in SUBS if x != held]; val = tr[-1]; fit = tr[:-1]
        xs, ys = scalers(tr)
        if kind == "LSTM":
            def build(): m = LSTMNet(p["h"], p["nl"], p["drop"]); m._lr, m._wd = p["lr"], p["wd"]; return m
        else:
            def build(): m = PatchTST(W, p["patch"], p["dm"], p["nl"], p["nh"], p["drop"]); m._lr, m._wd = p["lr"], p["wd"]; return m
        net, _ = train(build, W, fit, val, xs, ys, epochs=70)
        Xte, _ = pack([held], xs, ys, W); real = df[df.subj == held][MF].values
        net.eval()
        with torch.no_grad(): pred = ys.inverse_transform(net(Xte).numpy())
        for j, t in enumerate(MF): acc[t].append(r2_score(real[:, j], pred[:, j]))
        nets.append((net, xs, ys, W, held))
    return {t: float(np.mean(v)) for t, v in acc.items()}, nets


def perm_importance(net, xs, ys, W, held):
    """occlusion : permute chaque feature dans la fenêtre, mesure la chute de R²."""
    sub = df[df.subj == held]; X = xs.transform(sub[FEATS].values).astype("float32"); real = sub[MF].values
    base_w, _ = windows(sub, xs, ys, W); base_w = torch.tensor(base_w)
    net.eval()
    with torch.no_grad(): base = np.mean([r2_score(real[:, j], ys.inverse_transform(net(base_w).numpy())[:, j]) for j in range(O)])
    imp = {}
    rng = np.random.default_rng(0)
    for fi, fn in enumerate(FEATS):
        Xp = X.copy(); Xp[:, fi] = rng.permutation(Xp[:, fi])
        ww = []
        for i in range(len(Xp)):
            s = max(0, i - W + 1); w = Xp[s:i + 1]
            if len(w) < W: w = np.vstack([np.repeat(w[:1], W - len(w), 0), w])
            ww.append(w)
        with torch.no_grad():
            p = ys.inverse_transform(net(torch.tensor(np.asarray(ww, "float32"))).numpy())
            r2 = np.mean([r2_score(real[:, j], p[:, j]) for j in range(O)])
        imp[fn] = base - r2
    return imp


def main():
    res = {}
    print("FATIGUE TUNED (LOSO 8 sujets, %d features)\n" % F)
    for kind in ["LSTM", "PatchTST"]:
        t0 = time.time(); print("=== Optuna %s ===" % kind)
        best = tune(kind); print("  best:", best)
        r2, nets = full_loso(kind, best); res[kind + "_tuned"] = r2
        print("%-14s mean=%.3f | %s (%.0fs)" % (kind + "_tuned", np.mean(list(r2.values())),
              " ".join("%s=%.2f" % (t.replace("MF_", ""), r2[t]) for t in MF), time.time() - t0))
        if kind == "LSTM":  # XAI sur le meilleur LSTM (1er fold)
            net, xs, ys, W, held = nets[0]
            imp = perm_importance(net, xs, ys, W, held)
            xi = pd.Series(imp).sort_values(ascending=False)
            xi.round(4).to_csv(os.path.join(OUT, "xai_ts.csv"))
            print("  XAI (perm. importance, top 8) :"); print(xi.head(8).round(3).to_string())

    rep = pd.DataFrame(res).T; rep["mean"] = rep.mean(1)
    rep.round(3).to_csv(os.path.join(OUT, "ts_fatigue_tuned.csv"))
    print("\n=== TS tuned (R² fatigue LOSO) ===\n", rep.round(3))
    print("wrote ts_fatigue_tuned.csv + xai_ts.csv")


if __name__ == "__main__":
    main()
