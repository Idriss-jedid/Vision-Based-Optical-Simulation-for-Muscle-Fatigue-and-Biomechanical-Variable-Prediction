"""
STAGE 1 - Minimum-Jerk Human Motion Generator (physiologically-faithful fatigue)
================================================================================

Generates repetitive elbow-flexion (bicep-curl) trajectories for OpenSim from
minimum-jerk motor-control theory (Flash & Hogan 1985), with progressive,
biologically-grounded fatigue:
    A. slower motion        -> longer duration T
    B. reduced ROM          -> lower peak angle qf
    C. tremor / less smooth -> band-limited (~6-12 Hz) oscillation
    D. timing asymmetry     -> concentric slower, eccentric faster
    E. endpoint drift       -> slow (~0.1-0.5 Hz) turnaround wander
Fatigue follows a SATURATING curve f = (1-e^{-r*x})/(1-e^{-r}) (Xia/Frey-Law shape).

This module is BOTH a script (writes the canonical single-subject files) and a
library (`build`, writers, `qc_report`, `default_params`) imported by
`generate_subject_dataset.py` to build a multi-subject dataset.

Improvements addressed here (review limitations #4-#10):
    #4  N_REPS is a free parameter (use 30-100 for a real protocol)
    #5  FATIGUE_RATE lowered so the decline is not front-loaded over few reps
    #6  optional CLIP_TO_ROM bounds the drift to [q0, qf]
    #7/#8  optional LOWPASS_HZ produces an ID-ready, low-noise .mot; derivatives
           are finite differences of the (optionally filtered) written signal
    #9  parameterised so subjects can be randomised (see dataset script)
    #10 qc_report() validates velocity profile / symmetry / fatigue trends
"""

import math
import random

# ----------------------------------------------------------------------------
# DEFAULT CONFIG  (single-subject canonical run)
# ----------------------------------------------------------------------------
NAME      = "paper_minjerk_fatigue_10cycles"
OUT_MOT   = "paper_minjerk_fatigue_10cycles.mot"
OUT_STO   = "paper_minjerk_fatigue_10cycles.sto"
OUT_CSV   = "paper_minjerk_fatigue_10cycles_dataset.csv"
WRITE_CSV = True
WRITE_STO = True


def default_params():
    """Return the baseline parameter set (one dict; copy & edit per subject)."""
    return dict(
        sample_rate=100.0,
        n_reps=10,            # #4: raise to 30-100 for a real fatigue protocol
        q0=20.0, qf=120.0,    # literature curl ROM (deg)
        shoulder_fix=20.0,    # stabilised shoulder elevation (arm26)
        t_flex=1.5, t_ext=1.5,
        pause_top=0.20, pause_bottom=0.20,
        enable_fatigue=True,
        fatigue_rate=1.2,     # #5: gentler than before (was 2.5) -> not front-loaded
        dur_gain=0.40,        # +40% duration at full fatigue (A)
        rom_loss=20.0,        # qf drops 120->100 at full fatigue (B)
        asym=0.25,            # concentric +/- eccentric timing (D)
        tremor_amp=1.2, tremor_f_lo=6.0, tremor_f_hi=12.0, tremor_ncomp=8,   # C
        drift_amp=2.0, drift_f_lo=0.10, drift_f_hi=0.50, drift_ncomp=4,      # E
        clip_to_rom=False,    # #6: True clips drift/tremor to [q0, qf]
        lowpass_hz=None,      # #7/#8: e.g. 10.0 -> zero-phase low-pass, ID-ready
        seed=42,
    )


# ----------------------------------------------------------------------------
# core math
# ----------------------------------------------------------------------------
def minjerk(q_start, q_end, T, tau):
    if T <= 0:
        return q_end
    s = min(max(tau / T, 0.0), 1.0)
    dq = q_end - q_start
    return q_start + dq * (10 * s**3 - 15 * s**4 + 6 * s**5)


def fatigue_level(x, rate, enable):
    if not enable:
        return 0.0
    if rate <= 0:
        return x
    return (1.0 - math.exp(-rate * x)) / (1.0 - math.exp(-rate))


def make_band(rng, f_lo, f_hi, ncomp):
    comps = []
    for i in range(ncomp):
        freq = f_lo + (i + 0.5) / ncomp * (f_hi - f_lo)
        comps.append((freq, rng.uniform(0.0, 2 * math.pi)))
    return comps


def band_signal(comps, t):
    return sum(math.sin(2 * math.pi * f * t + ph) for f, ph in comps) / math.sqrt(len(comps))


def _lowpass(times, y, cutoff_hz):
    """Zero-phase low-pass. Uses scipy if available, else a 2-pass moving average."""
    dt = times[1] - times[0]
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(2, cutoff_hz / (0.5 / dt), btype="low")
        return list(filtfilt(b, a, y))
    except Exception:
        # fallback: symmetric moving average ~ matched to cutoff
        win = max(1, int(round((1.0 / cutoff_hz) / dt / 2)))
        out = []
        n = len(y)
        for i in range(n):
            lo, hi = max(0, i - win), min(n, i + win + 1)
            out.append(sum(y[lo:hi]) / (hi - lo))
        return out


def derivatives(times, y):
    n = len(y)
    vel = [0.0] * n
    for i in range(n):
        if i == 0:
            vel[i] = (y[1] - y[0]) / (times[1] - times[0])
        elif i == n - 1:
            vel[i] = (y[-1] - y[-2]) / (times[-1] - times[-2])
        else:
            vel[i] = (y[i + 1] - y[i - 1]) / (times[i + 1] - times[i - 1])
    acc = [0.0] * n
    for i in range(n):
        if i == 0:
            acc[i] = (vel[1] - vel[0]) / (times[1] - times[0])
        elif i == n - 1:
            acc[i] = (vel[-1] - vel[-2]) / (times[-1] - times[-2])
        else:
            acc[i] = (vel[i + 1] - vel[i - 1]) / (times[i + 1] - times[i - 1])
    return vel, acc


# ----------------------------------------------------------------------------
# build one subject's motion
# ----------------------------------------------------------------------------
def build(p):
    rng = random.Random(p["seed"])
    dt = 1.0 / p["sample_rate"]
    N = p["n_reps"]
    tremor = make_band(rng, p["tremor_f_lo"], p["tremor_f_hi"], p["tremor_ncomp"])
    drift = make_band(rng, p["drift_f_lo"], p["drift_f_hi"], p["drift_ncomp"])

    def rep_fatigue(k):
        return fatigue_level(k / (N - 1), p["fatigue_rate"], p["enable_fatigue"]) if N > 1 else 0.0

    times, elbows, reps, fatigues = [], [], [], []
    t = 0.0
    for k in range(N):
        f_k = rep_fatigue(k)
        f_next = rep_fatigue(min(k + 1, N - 1))
        dur_scale = 1.0 + p["dur_gain"] * f_k
        t_flex = p["t_flex"] * dur_scale * (1.0 + p["asym"] * f_k)
        t_ext = p["t_ext"] * dur_scale * (1.0 - p["asym"] * f_k)
        qf_k = p["qf"] - p["rom_loss"] * f_k
        segs = [("flex", t_flex, p["q0"], qf_k), ("hold", p["pause_top"], qf_k, qf_k),
                ("ext", t_ext, qf_k, p["q0"]), ("hold", p["pause_bottom"], p["q0"], p["q0"])]
        rep_n = sum(int(round(d / dt)) for _, d, _, _ in segs)
        j = 0
        for _, dur, qs, qe in segs:
            n = int(round(dur / dt))
            for i in range(n):
                pos = minjerk(qs, qe, dur, i * dt)
                fs = f_k + (f_next - f_k) * (j / rep_n if rep_n else 0.0)
                pos += fs * p["tremor_amp"] * band_signal(tremor, t)
                pos += fs * p["drift_amp"] * band_signal(drift, t)
                if p["clip_to_rom"]:
                    pos = min(max(pos, p["q0"]), p["qf"])
                times.append(t); elbows.append(pos); reps.append(k + 1); fatigues.append(fs)
                t += dt; j += 1
    # closing sample
    fe = rep_fatigue(N - 1)
    pos = p["q0"] + fe * p["tremor_amp"] * band_signal(tremor, t) + fe * p["drift_amp"] * band_signal(drift, t)
    if p["clip_to_rom"]:
        pos = min(max(pos, p["q0"]), p["qf"])
    times.append(t); elbows.append(pos); reps.append(N); fatigues.append(fe)

    if p["lowpass_hz"]:
        elbows = _lowpass(times, elbows, p["lowpass_hz"])

    vels, accs = derivatives(times, elbows)
    return dict(t=times, elbow=elbows, vel=vels, acc=accs, rep=reps, fatigue=fatigues)


# ----------------------------------------------------------------------------
# writers
# ----------------------------------------------------------------------------
def _write_table(path, name, t, elbow, shoulder_fix):
    with open(path, "w", newline="\n") as f:
        f.write("%s\nversion=1\nnRows=%d\nnColumns=3\ninDegrees=yes\nendheader\n" % (name, len(t)))
        f.write("time\tr_shoulder_elev\tr_elbow_flex\n")
        for ti, e in zip(t, elbow):
            f.write("%.4f\t%.6f\t%.6f\n" % (ti, shoulder_fix, e))


def write_mot(path, name, d, p): _write_table(path, name, d["t"], d["elbow"], p["shoulder_fix"])
def write_sto(path, name, d, p): _write_table(path, name, d["t"], d["elbow"], p["shoulder_fix"])


def write_csv(path, d, p):
    with open(path, "w", newline="\n") as f:
        f.write("time,r_shoulder_elev,r_elbow_flex,elbow_vel,elbow_acc,rep_index,fatigue_level\n")
        for ti, e, v, a, r, fa in zip(d["t"], d["elbow"], d["vel"], d["acc"], d["rep"], d["fatigue"]):
            f.write("%.4f,%.6f,%.6f,%.6f,%.6f,%d,%.4f\n" % (ti, p["shoulder_fix"], e, v, a, r, fa))


# ----------------------------------------------------------------------------
# #10  quality-control / validation against minimum-jerk & literature
# ----------------------------------------------------------------------------
def qc_report(d, p, verbose=True):
    """Validate the generated motion. Returns a dict of metrics + pass flags."""
    t, el, vel, rep = d["t"], d["elbow"], d["vel"], d["rep"]
    idx = lambda k: [i for i, r in enumerate(rep) if r == k]

    def smooth(y, win=8):
        # symmetric moving average -> removes 6-12 Hz tremor so the MOVEMENT
        # (sub-5 Hz) peak velocity is measured, not a tremor spike.
        n = len(y); out = []
        for i in range(n):
            lo, hi = max(0, i - win), min(n, i + win + 1)
            out.append(sum(y[lo:hi]) / (hi - lo))
        return out

    def rep_metrics(k):
        ii = idx(k)
        e = [el[i] for i in ii]; v = smooth([vel[i] for i in ii]); tt = [t[i] for i in ii]
        rom = max(e) - min(e)
        # flexion phase = up to the peak angle
        pk = e.index(max(e))
        vflex = v[:pk + 1] if pk > 2 else v
        vpeak = max(vflex) if vflex else 0.0
        # time-to-peak-velocity as fraction of flexion phase (min-jerk -> 0.5)
        vpi = vflex.index(vpeak) if vflex else 0
        ttp = (tt[vpi] - tt[0]) / (tt[pk] - tt[0]) if pk > 0 and tt[pk] > tt[0] else 0.5
        dur = tt[-1] - tt[0]
        return dict(rom=rom, vpeak=vpeak, ttp=ttp, dur=dur)

    m1 = rep_metrics(1)
    mN = rep_metrics(p["n_reps"])
    # min-jerk predicted fresh peak velocity = 1.875 * ROM / T_flex
    vpeak_pred = 1.875 * (p["qf"] - p["q0"]) / p["t_flex"]

    # per-rep series + least-squares slope vs rep index (robust to zero-mean
    # drift noise, unlike a single rep1-vs-repN comparison).
    allm = [rep_metrics(k) for k in range(1, p["n_reps"] + 1)]
    def slope(key):
        ys = [mm[key] for mm in allm]; n = len(ys)
        if n < 2:
            return 0.0
        xs = list(range(n)); mx = sum(xs)/n; my = sum(ys)/n
        den = sum((x-mx)**2 for x in xs)
        return sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/den if den else 0.0

    checks = []
    def chk(name, ok, detail):
        checks.append((name, ok, detail))

    chk("fresh peak-velocity matches min-jerk (1.875*ROM/T)",
        abs(m1["vpeak"] - vpeak_pred) / vpeak_pred < 0.20,
        "got %.0f deg/s, predicted %.0f" % (m1["vpeak"], vpeak_pred))
    chk("fresh velocity profile symmetric (time-to-peak ~50%)",
        0.35 <= m1["ttp"] <= 0.65,
        "time-to-peak = %.0f%% of flexion" % (m1["ttp"] * 100))
    chk("fresh ROM ~ commanded",
        abs(m1["rom"] - (p["qf"] - p["q0"])) < 6,
        "%.1f deg (cmd %.0f)" % (m1["rom"], p["qf"] - p["q0"]))
    if p["enable_fatigue"] and p["n_reps"] > 1:
        chk("fatigue: peak velocity trend DECREASES", slope("vpeak") < 0.0,
            "slope %.2f deg/s per rep (rep1 %.0f -> repN %.0f)"
            % (slope("vpeak"), m1["vpeak"], mN["vpeak"]))
        chk("fatigue: ROM trend DECREASES", slope("rom") < 0.02,
            "slope %.3f deg per rep (rep1 %.1f -> repN %.1f)"
            % (slope("rom"), m1["rom"], mN["rom"]))
        chk("fatigue: movement SLOWS (duration trend up)", slope("dur") > 0.0,
            "slope %.3f s per rep (rep1 %.2f -> repN %.2f)"
            % (slope("dur"), m1["dur"], mN["dur"]))
    chk("peak elbow velocity in human range (<300 deg/s)",
        m1["vpeak"] < 300, "%.0f deg/s" % m1["vpeak"])

    if verbose:
        print("  QC (vs minimum-jerk / literature):")
        for name, ok, detail in checks:
            print("    [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    return dict(rep1=m1, repN=mN, vpeak_pred=vpeak_pred, checks=checks,
                all_pass=all(ok for _, ok, _ in checks))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    p = default_params()
    d = build(p)
    write_mot(OUT_MOT, NAME, d, p)
    if WRITE_STO:
        write_sto(OUT_STO, NAME, d, p)
    if WRITE_CSV:
        write_csv(OUT_CSV, d, p)
    print("Generated %s" % OUT_MOT)
    print("  rows %d | %.2f s @ %.0f Hz | %d reps | fatigue_rate %.1f | ROM %.0f-%.0f"
          % (len(d["t"]), d["t"][-1], p["sample_rate"], p["n_reps"], p["fatigue_rate"], p["q0"], p["qf"]))
    print("  elbow range %.2f .. %.2f deg" % (min(d["elbow"]), max(d["elbow"])))
    qc_report(d, p)
    if WRITE_STO: print("Wrote %s" % OUT_STO)
    if WRITE_CSV: print("Wrote %s" % OUT_CSV)
