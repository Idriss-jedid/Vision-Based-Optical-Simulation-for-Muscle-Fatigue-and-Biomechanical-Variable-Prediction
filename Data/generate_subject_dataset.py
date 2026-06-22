"""
STAGE 1 - Multi-subject synthetic dataset generator  (review limitation #9).

Builds a population of virtual subjects by randomising the minimum-jerk + fatigue
parameters (ROM, movement speed, tremor, fatigue resistance, asymmetry), so the
downstream AI trains on inter-subject variability instead of one trajectory.

For each subject it writes:
    Subjects/subject_<ID>.mot              (OpenSim coordinates)
    Subjects/subject_<ID>_dataset.csv      (time, angle, vel, acc, rep, fatigue)
and a master index `subjects_index.csv` with every subject's parameters, a
fatigue-resistance label, and whether it passed the physiological QC.

Run:  python generate_subject_dataset.py [n_subjects] [n_reps]
"""
import os
import sys
import random

import generate_minjerk_motion as g

OUTDIR = "Subjects"
INDEX = "subjects_index.csv"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def random_subject(rng, sid, n_reps):
    """Draw a physiologically-plausible parameter set for one subject."""
    p = g.default_params()
    p["seed"] = 1000 + sid
    p["n_reps"] = n_reps
    # --- anatomy / task ---
    p["q0"] = clamp(rng.gauss(20, 3), 10, 30)
    p["qf"] = clamp(rng.gauss(120, 8), 105, 135)
    p["shoulder_fix"] = clamp(rng.gauss(20, 5), 5, 40)
    # --- movement speed (correlated flex/ext, slight asymmetry) ---
    base_T = clamp(rng.gauss(1.8, 0.4), 1.0, 3.0)
    p["t_flex"] = base_T
    p["t_ext"] = clamp(base_T * rng.gauss(1.0, 0.1), 1.0, 3.2)
    # --- fatigue resistance (key inter-subject axis) ---
    p["fatigue_rate"] = clamp(rng.gauss(1.2, 0.5), 0.5, 2.5)
    p["rom_loss"] = clamp(rng.gauss(20, 8), 5, 35)
    p["dur_gain"] = clamp(rng.gauss(0.40, 0.15), 0.10, 0.70)
    p["asym"] = clamp(rng.gauss(0.25, 0.10), 0.0, 0.45)
    # --- tremor / motor noise (band centre varies per subject) ---
    fc = clamp(rng.gauss(9.0, 1.5), 5.0, 13.0)
    p["tremor_f_lo"], p["tremor_f_hi"] = fc - 2.0, fc + 2.0
    p["tremor_amp"] = clamp(rng.gauss(1.2, 0.5), 0.3, 2.5)
    p["drift_amp"] = clamp(rng.gauss(2.0, 0.8), 0.5, 3.5)
    return p


def fatigue_resistance_label(p):
    """Composite 0(low)-1(high) resistance: slow onset + small ROM loss = high."""
    r = (1 - (p["fatigue_rate"] - 0.5) / 2.0) * 0.5 + (1 - (p["rom_loss"] - 5) / 30.0) * 0.5
    return round(clamp(r, 0, 1), 3)


def main():
    n_subjects = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rng = random.Random(2026)
    os.makedirs(OUTDIR, exist_ok=True)

    cols = ["subject_id", "n_reps", "q0", "qf", "shoulder_fix", "t_flex", "t_ext",
            "fatigue_rate", "rom_loss", "dur_gain", "asym", "tremor_fc", "tremor_amp",
            "drift_amp", "fatigue_resistance", "qc_pass", "mot_file", "csv_file"]
    rows = []
    n_pass = 0
    print("Generating %d subjects x %d reps ...\n" % (n_subjects, n_reps))
    for sid in range(1, n_subjects + 1):
        p = random_subject(rng, sid, n_reps)
        d = g.build(p)
        mot = os.path.join(OUTDIR, "subject_%02d.mot" % sid)
        csv = os.path.join(OUTDIR, "subject_%02d_dataset.csv" % sid)
        g.write_mot(mot, "subject_%02d" % sid, d, p)
        g.write_csv(csv, d, p)
        qc = g.qc_report(d, p, verbose=False)
        n_pass += int(qc["all_pass"])
        fc = (p["tremor_f_lo"] + p["tremor_f_hi"]) / 2
        rows.append([sid, n_reps, p["q0"], p["qf"], p["shoulder_fix"], p["t_flex"], p["t_ext"],
                     p["fatigue_rate"], p["rom_loss"], p["dur_gain"], p["asym"], fc,
                     p["tremor_amp"], p["drift_amp"], fatigue_resistance_label(p),
                     int(qc["all_pass"]), mot, csv])
        print("  subject %02d | ROM %.0f-%.0f | T %.1f/%.1f s | f_rate %.1f rom_loss %.0f | "
              "resist %.2f | QC %s"
              % (sid, p["q0"], p["qf"], p["t_flex"], p["t_ext"], p["fatigue_rate"],
                 p["rom_loss"], fatigue_resistance_label(p), "ok" if qc["all_pass"] else "FAIL"))

    with open(INDEX, "w", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(("%.4f" % v if isinstance(v, float) else str(v)) for v in r) + "\n")

    print("\nWrote %d subjects to %s/  +  index %s" % (n_subjects, OUTDIR, INDEX))
    print("QC passed: %d/%d subjects" % (n_pass, n_subjects))


if __name__ == "__main__":
    main()
