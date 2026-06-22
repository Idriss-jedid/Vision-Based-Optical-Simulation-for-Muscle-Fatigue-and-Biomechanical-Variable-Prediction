# Stage 1 — Improvements Addressing Limitations #4–#10

How each reviewed limitation of the minimum-jerk generator was handled. Code:
`generate_minjerk_motion.py` (library + canonical single subject) and
`generate_subject_dataset.py` (multi-subject dataset).

| # | Limitation | Status | What changed |
|---|------------|--------|--------------|
| 4 | "Only 10 reps" | ✅ fixed | `n_reps` is a free parameter; dataset uses **30** (`python generate_subject_dataset.py 8 30`), scale to 100 freely. |
| 5 | "Saturating fatigue too aggressive early" | ✅ fixed | `fatigue_rate` lowered **2.5 → 1.2**, so the decline is no longer front-loaded; per-subject randomised 0.5–2.5. |
| 6 | "Endpoint drift exceeds ROM" | ✅ optional | New `clip_to_rom` flag bounds drift/tremor to `[q0, qf]` when strict limits are needed (off by default to preserve realism). |
| 7 | "Tremor affects torques" | ✅ optional | New `lowpass_hz` applies a zero-phase Butterworth low-pass (scipy) → an **ID-ready, low-noise `.mot`**; default keeps the physiological tremor. |
| 8 | "Finite differences sensitive to noise" | ✅ handled | Derivatives are finite differences of the (optionally low-passed) **written** signal, so they stay consistent with the `.mot`; `lowpass_hz` removes the high-freq amplification when needed. |
| 9 | "No subject variability" | ✅ **new feature** | `generate_subject_dataset.py` builds a population with randomised ROM, speed, tremor (amplitude + band centre), fatigue rate, ROM loss, duration gain and asymmetry → a **multi-subject dataset** with a labelled index. |
| 10 | "No experimental validation" | ✅ **new feature** | `qc_report()` validates every motion against minimum-jerk / literature: bell-shaped symmetric velocity, peak velocity = 1.875·ROM/T, ROM, and fatigue **trend slopes** (peak-velocity↓, ROM↓, duration↑) measured by regression over all reps (robust to drift). |

## Validation (QC) — what it checks

Run automatically on every generated motion (`qc_report`). For the canonical
subject all checks PASS:

```
[PASS] fresh peak-velocity matches min-jerk (1.875*ROM/T)  124 vs 125 deg/s
[PASS] fresh velocity profile symmetric (time-to-peak ~50%) 46%
[PASS] fresh ROM ~ commanded                                100.4 deg
[PASS] fatigue: peak velocity trend DECREASES               -6.2 deg/s per rep
[PASS] fatigue: ROM trend DECREASES                         -1.8 deg per rep
[PASS] fatigue: movement SLOWS (duration trend up)          +0.13 s per rep
[PASS] peak elbow velocity in human range (<300 deg/s)      124 deg/s
```

Note: fatigue trends are checked as the least-squares **slope across all reps**,
not rep1-vs-repN — because the endpoint drift (a deliberate realism feature) is
zero-mean and would otherwise mask a small per-rep ROM change in a 2-point test.

## Multi-subject dataset

`generate_subject_dataset.py N_SUBJECTS N_REPS` writes:

```
Subjects/subject_NN.mot            OpenSim coordinates (one virtual subject)
Subjects/subject_NN_dataset.csv    time, angle, vel, acc, rep, fatigue
subjects_index.csv                 per-subject params + fatigue_resistance + qc_pass
```

Randomised per subject (Gaussian, clamped to plausible ranges): `q0, qf,
shoulder_fix, t_flex, t_ext, fatigue_rate, rom_loss, dur_gain, asym, tremor band
centre, tremor_amp, drift_amp`. A composite **`fatigue_resistance`** label
(0 = fatigues fast / large ROM loss, 1 = resistant) is exported for supervised
learning. Demo run: 8 subjects × 30 reps, **8/8 passed QC**.

## Still open (require external data / future work)

- #10 *experimental* validation against real mocap/EMG is only partial here: the
  QC compares against minimum-jerk theory and literature ranges, not a measured
  dataset. True validation needs comparison to recorded elbow kinematics
  (velocity profiles, torque ranges) — a future step.
- Subject variability is parametric, not learned from a real population.
