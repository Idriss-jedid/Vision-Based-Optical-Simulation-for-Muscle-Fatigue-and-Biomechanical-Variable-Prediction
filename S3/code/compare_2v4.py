# pyright: reportMissingImports=false
"""
Compare 2-camera vs 4-camera, full 14 s, through the WHOLE chain identically:
raw TRC elbow angle -> de-spike -> affine calibration vs Vicon -> 2 Hz low-pass
-> arm26 (Vicon-scaled) -> ID / SO / 3CC.  Same model + same processing for both,
so the only difference is the triangulation source (2 vs 4 cameras).
Run with the biomech env. -> S3/compare_2v4/
"""
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt

HERE = os.path.dirname(os.path.abspath(__file__))
S3 = os.path.dirname(HERE)
ROOT = os.path.dirname(S3)
sys.path.insert(0, os.path.join(ROOT, "Code"))
import run_stage2_pipeline as P  # noqa: E402

GT_CSV = os.path.join(S3, "build", "csv", "joints3d_25.csv")
SCALED = os.path.join(S3, "build", "opensim", "arm26_S3_scaled.osim")   # same model for both
OUT = os.path.join(S3, "compare_2v4")
RATE, ELB_MAX, FPS = 100.0, 148.0, 50.0
SETUPS = {"2cam": os.path.join(S3, "build2", "pose2sim", "pose-3d"),
          "4cam": os.path.join(S3, "build4", "pose2sim", "pose-3d")}


def flex(S, E, W):
    v1, v2 = S - E, W - E
    cs = np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)
    return 180.0 - np.degrees(np.arccos(cs))


def ml_angle(trc_dir):
    trc = sorted(f for f in glob.glob(os.path.join(trc_dir, "*.trc")) if "LSTM" not in f)[-1]
    L = open(trc).read().splitlines()
    mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return flex(g("RShoulder"), g("RElbow"), g("RWrist"))


def hampel(x, k=10, ns=3.0):
    x = x.astype(float).copy(); n = len(x)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1); w = x[lo:hi]; med = np.nanmedian(w)
        mad = 1.4826 * np.nanmedian(np.abs(w - med)) + 1e-9
        if not np.isnan(x[i]) and abs(x[i] - med) > ns * mad: x[i] = np.nan
    idx = np.arange(n); good = ~np.isnan(x); return np.interp(idx, idx[good], x[good])


def vicon():
    rows = list(csv.DictReader(open(GT_CSV)))
    J = np.array([[float(r["J%d_%s" % (j, ax)]) for j in range(25) for ax in "xyz"] for r in rows]).reshape(-1, 25, 3)
    return flex(J[:, 14], J[:, 15], J[:, 16])


def process(name, trc_dir, gt):
    outdir = os.path.join(OUT, name); os.makedirs(outdir, exist_ok=True)
    raw = hampel(medfilt(ml_angle(trc_dir), 7), 10)
    n = min(len(raw), len(gt)); raw = raw[:n]; g = gt[:n]
    # align + affine calibration vs Vicon
    best = (-2, 0)
    for lag in range(0, 8):
        seg = gt[lag:lag + n]
        if len(seg) < n: continue
        r = np.corrcoef(raw, seg)[0, 1]
        if r > best[0]: best = (r, lag)
    lag = best[1]; g = gt[lag:lag + n]
    a, b = np.polyfit(raw, g, 1); cal = a * raw + b
    # resample + 2 Hz
    t = np.arange(n) / FPS; tv = np.arange(0, t[-1] + 1e-9, 1.0 / RATE)
    flx = np.interp(tv, t, cal)
    bb, aa = butter(4, 2.0 / (RATE / 2), btype="low"); flx = np.clip(filtfilt(bb, aa, flx), 0.0, ELB_MAX)
    # write .mot + dataset
    motion = os.path.join(outdir, "curl.mot")
    with open(motion, "w", newline="\n") as f:
        f.write("c\nversion=1\nnRows=%d\nnColumns=3\ninDegrees=yes\nendheader\n" % len(tv))
        f.write("time\tr_shoulder_elev\tr_elbow_flex\n")
        for k in range(len(tv)): f.write("%.4f\t20.000000\t%.6f\n" % (tv[k], flx[k]))
    ds = os.path.join(outdir, "ds.csv")
    open(ds, "w", newline="\n").write("time,rep_index,fatigue_level\n" + "".join("%.4f,1,0\n" % x for x in tv))
    # OpenSim ID/SO/3CC (same Vicon-scaled model)
    for fn in os.listdir(outdir):
        if fn.endswith((".sto", ".xml")): os.remove(os.path.join(outdir, fn))
    t0, t1 = P.motion_range(motion)
    mp = P.prep_model(SCALED, outdir)
    ids = P.run_id(mp, motion, outdir, t0, t1)
    act, frc = P.run_so(mp, motion, outdir, name, t0, t1)
    labels = P.merge_labels(ids, act, frc, ds, outdir)
    fat, fails = P.run_3cc(labels, SCALED, motion, outdir)
    idc, idr = P.read_sto(ids); el = [c for c in idc if "elbow" in c and "moment" in c][0]
    peak = max(abs(v) for v in P.col(idc, idr, el))
    # vs Vicon (on the calibrated angle, resampled to Vicon grid)
    gtv = np.interp(tv, t, g); e = flx - gtv; mae = float(np.mean(np.abs(e)))
    # reps via threshold
    thr = (flx.max() + flx.min()) / 2; ab = flx > thr; nrep = int(np.sum((~ab[:-1]) & (ab[1:])))
    return dict(name=name, r=best[0], mae=mae, rom=(flx.min(), flx.max()), peak=peak,
                fails=fails, nrep=nrep, tv=tv, flx=flx, gtv=gtv, a=a, b=b)


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = vicon()
    res = {k: process(k, d, gt) for k, d in SETUPS.items()}
    print("\n================= 2-CAM vs 4-CAM (full 14 s, identical processing) =================")
    print("%-6s %7s %8s %14s %10s %7s %6s" % ("setup", "r", "MAE", "ROM(deg)", "peakTorque", "fails", "reps"))
    for k in ["2cam", "4cam"]:
        x = res[k]
        print("%-6s %7.3f %6.1f° %6.0f-%-6.0f %8.2f N·m %7d %6d" %
              (k, x["r"], x["mae"], x["rom"][0], x["rom"][1], x["peak"], x["fails"], x["nrep"]))
    print("affine: 2cam a=%.2f b=%.0f | 4cam a=%.2f b=%.0f" % (res["2cam"]["a"], res["2cam"]["b"], res["4cam"]["a"], res["4cam"]["b"]))

    plt.figure(figsize=(13, 5))
    plt.plot(res["2cam"]["tv"], res["2cam"]["gtv"], color="#2ca02c", lw=2.4, label="Vicon GT")
    plt.plot(res["2cam"]["tv"], res["2cam"]["flx"], color="#1f77b4", lw=1.5, label="2-cam (r=%.3f, MAE %.0f°)" % (res["2cam"]["r"], res["2cam"]["mae"]))
    plt.plot(res["4cam"]["tv"], res["4cam"]["flx"], color="#d62728", lw=1.5, ls="--", label="4-cam (r=%.3f, MAE %.0f°)" % (res["4cam"]["r"], res["4cam"]["mae"]))
    plt.xlabel("time (s)"); plt.ylabel("elbow flexion (deg)"); plt.title("2-cam vs 4-cam vs Vicon — full 14 s")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "compare_2v4.png"), dpi=130); plt.close()
    print("plot -> compare_2v4/compare_2v4.png")


if __name__ == "__main__":
    main()
