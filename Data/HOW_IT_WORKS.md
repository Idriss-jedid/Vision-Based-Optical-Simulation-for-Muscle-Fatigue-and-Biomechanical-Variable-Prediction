# Stage 1 — Minimum-Jerk Motion Generator: How It Works

This document explains the updated Stage-1 motion generator after the reviewed
improvements in `STAGE1_IMPROVEMENTS.md`.

Scope boundary: this document is about movement generation (kinematics) only.
Vision/camera modeling belongs to Stage 3 and is referenced here only as a
downstream data-consumer context.

Main code files:

| File | Purpose |
|------|---------|
| `generate_minjerk_motion.py` | canonical single-subject minimum-jerk + fatigue generator |
| `generate_subject_dataset.py` | multi-subject synthetic dataset generator |
| `paper_minjerk_fatigue_10cycles.mot` | canonical OpenSim motion file |
| `paper_minjerk_fatigue_10cycles_dataset.csv` | per-frame labels for the canonical motion |
| `paper_minjerk_fatigue_10cycles.sto` | OpenSim Storage-format copy of the canonical motion |
| `Subjects/subject_XX.mot` | multi-subject OpenSim motion files |
| `Subjects/subject_XX_dataset.csv` | multi-subject per-frame datasets |
| `subjects_index.csv` | subject parameters, fatigue-resistance labels, and QC pass flags |

---

## 1. Scientific Goal

The goal is to generate physiologically plausible repetitive elbow-curl motion
without motion capture.

Instead of recording a person, the elbow angle is generated from a motor-control
law: **minimum jerk**. This gives a clean human-like baseline. Then controlled
fatigue effects are added so the motion becomes slower, smaller in range,
shakier, and more asymmetric over repetitions.

The resulting `.mot` file drives the OpenSim model in Stage 2, and the matching
CSV files provide labels for the AI/data stage.

```text
minimum-jerk law
    -> clean elbow curl
    -> progressive fatigue perturbation
    -> OpenSim .mot/.sto + AI-ready CSV
```

---

## 2. Minimum-Jerk Motion

For one movement from `q_start` to `q_end` over duration `T`, the normalized time
is:

```text
s = tau / T
```

The elbow angle is:

```text
q(tau) = q_start + (q_end - q_start) * (10*s^3 - 15*s^4 + 6*s^5)
```

This fifth-order polynomial is used because it gives:

| Property | Meaning |
|----------|---------|
| zero velocity at endpoints | no sudden start/stop |
| zero acceleration at endpoints | no artificial acceleration spike |
| one bell-shaped velocity profile | close to experimentally observed human reaching/curl motion |
| smooth position curve | suitable baseline for OpenSim kinematics |

The clean analytic derivatives are:

```text
q_dot(tau)  = (q_end - q_start) * (30*s^2 - 60*s^3 + 30*s^4) / T
q_ddot(tau) = (q_end - q_start) * (60*s - 180*s^2 + 120*s^3) / T^2
```

Important detail: after tremor and drift are added, these analytic derivatives
are no longer the derivatives of the final written signal. Therefore the script
computes velocity and acceleration using finite differences of the final angle
that is actually written to the `.mot` file. This keeps the CSV labels
consistent with the OpenSim trajectory.

---

## 3. One Repetition

One elbow-curl repetition is built from four segments:

```text
20 deg -> 120 deg     flexion / concentric phase
120 deg hold          top pause
120 deg -> 20 deg     extension / eccentric phase
20 deg hold           bottom pause
```

The default canonical settings are:

| Parameter | Default |
|-----------|---------|
| sampling rate | 100 Hz |
| repetitions | 10 for the canonical demo; 30-100 for production datasets |
| start angle `q0` | 20 deg |
| fresh target `qf` | 120 deg |
| shoulder angle | 20 deg fixed |
| flexion duration | 1.5 s |
| extension duration | 1.5 s |
| top/bottom pause | 0.2 s |

The 20 to 120 degree range avoids full extension and gives a realistic controlled
curl range for the simplified arm26 model.

---

## 4. Fatigue Model

Fatigue is represented by a normalized level `f` from 0 to 1.

The current generator uses a **saturating fatigue curve**, not a straight line:

```text
f(x) = (1 - exp(-rate*x)) / (1 - exp(-rate))
```

where `x` is repetition progress from 0 to 1.

The fatigue rate was improved from the earlier aggressive value to:

```text
fatigue_rate = 1.2
```

This makes the decline less front-loaded over short protocols while still
allowing faster/slower fatigue in the multi-subject dataset.

The fatigue level controls five effects:

| Effect | Code parameter | What happens |
|--------|----------------|--------------|
| slower movement | `dur_gain` | movement duration increases up to 40% |
| reduced range of motion | `rom_loss` | target angle drops from 120 deg toward 100 deg |
| timing asymmetry | `asym` | flexion becomes slower and extension relatively faster |
| tremor / loss of smoothness | `tremor_amp`, `tremor_f_lo`, `tremor_f_hi` | 6-12 Hz band-limited tremor is added |
| endpoint drift | `drift_amp`, `drift_f_lo`, `drift_f_hi` | slow 0.1-0.5 Hz drift creates realistic turnaround variability |

The perturbations are band-limited sums of sinusoids, not white noise. This is
important because white noise would create unrealistic jerk and unstable
finite-difference derivatives.

---

## 5. Improvements Added After Review

The generator was reviewed against limitations #4-#10. The current version
addresses them as follows.

| # | Previous limitation | Current fix |
|---|---------------------|-------------|
| 4 | Only 10 repetitions | `n_reps` is now a free parameter; production datasets use 30 or more reps. |
| 5 | Fatigue too aggressive early | `fatigue_rate` lowered to 1.2; multi-subject mode randomizes 0.5-2.5. |
| 6 | Endpoint drift can exceed ROM | optional `clip_to_rom=True` bounds the final angle to `[q0, qf]`. |
| 7 | Tremor can affect inverse-dynamics torque | optional `lowpass_hz` produces an ID-ready lower-noise `.mot`. |
| 8 | Finite differences amplify noise | derivatives are computed from the written signal, after optional low-pass filtering. |
| 9 | No subject variability | `generate_subject_dataset.py` creates randomized virtual subjects. |
| 10 | No validation | `qc_report()` checks minimum-jerk velocity, symmetry, ROM, fatigue trends, and human velocity range. |

Two modes are therefore available:

| Mode | Purpose | Suggested settings |
|------|---------|--------------------|
| physiological variability mode | preserve tremor and drift for AI/fatigue signatures | `clip_to_rom=False`, `lowpass_hz=None` |
| inverse-dynamics-safe mode | reduce high-frequency oscillations before ID/SO | `lowpass_hz=10.0`, optionally `clip_to_rom=True` |

The default keeps physiological variability. For sensitive inverse-dynamics runs,
use the optional low-pass setting so tremor does not dominate acceleration and
torque.

---

## 6. Quality-Control Validation

The function `qc_report()` validates each generated motion against minimum-jerk
theory and broad literature expectations.

It checks:

- fresh peak velocity is close to the minimum-jerk prediction;
- fresh velocity profile is approximately symmetric;
- fresh ROM matches the commanded ROM;
- fatigue causes peak velocity to decrease over repetitions;
- fatigue causes ROM to decrease over repetitions;
- fatigue causes duration to increase over repetitions;
- peak elbow velocity remains in a plausible human range.

For the canonical subject, the intended output is like:

```text
[PASS] fresh peak-velocity matches min-jerk (1.875*ROM/T)
[PASS] fresh velocity profile symmetric (time-to-peak ~50%)
[PASS] fresh ROM ~ commanded
[PASS] fatigue: peak velocity trend DECREASES
[PASS] fatigue: ROM trend DECREASES
[PASS] fatigue: movement SLOWS (duration trend up)
[PASS] peak elbow velocity in human range (<300 deg/s)
```

The fatigue checks use regression slopes across all repetitions, not just rep 1
versus rep N. This is more robust because endpoint drift is deliberate and can
mask small differences if only two repetitions are compared.

---

## 7. Multi-Subject Dataset

The script `generate_subject_dataset.py` creates a population of virtual
subjects. Each subject receives randomized but clamped physiological parameters:

- start angle and target angle;
- shoulder stabilization angle;
- flexion and extension duration;
- fatigue rate;
- ROM loss;
- duration gain;
- timing asymmetry;
- tremor frequency center and amplitude;
- endpoint drift amplitude.

It also writes a composite `fatigue_resistance` label:

```text
0 = fatigues faster / larger ROM loss
1 = more fatigue resistant
```

Example production-style command from the `Data` folder:

```powershell
cd Data
python generate_subject_dataset.py 8 30
```

Outputs:

```text
Subjects/subject_01.mot
Subjects/subject_01_dataset.csv
...
Subjects/subject_08.mot
Subjects/subject_08_dataset.csv
subjects_index.csv
```

The current demo dataset contains 8 subjects x 30 repetitions, with QC status
stored in `subjects_index.csv`.

---

## 8. Output File Formats

The OpenSim `.mot` and `.sto` files contain only the coordinate columns OpenSim
needs:

```text
time    r_shoulder_elev    r_elbow_flex
0.0000  20.000000          20.000000
0.0100  20.000000          20.000235
...
```

The dataset CSV contains richer per-frame values:

```text
time,r_shoulder_elev,r_elbow_flex,elbow_vel,elbow_acc,rep_index,fatigue_level
```

The velocity and acceleration columns always describe the final written elbow
angle, including tremor, drift, clipping, or low-pass filtering if enabled.

---

## 9. How To Run

Run from the `Data` directory so generated outputs stay beside the data files:

```powershell
cd Data
python generate_minjerk_motion.py
```

Generate a multi-subject dataset:

```powershell
cd Data
python generate_subject_dataset.py 8 30
```

To create an ID-ready variant, edit the parameter dictionary in
`generate_minjerk_motion.py` or in a small wrapper script:

```python
p = default_params()
p["lowpass_hz"] = 10.0
p["clip_to_rom"] = True
```

Use the unfiltered physiological version for AI/fatigue-signature analysis, and
use the filtered ID-ready version when high-frequency tremor makes inverse
dynamics too sensitive.

---

## 10. What Changed From The Original Motion

| Feature | Original `paper_minjerk_10cycles.mot` | Improved Stage-1 generator |
|---------|---------------------------------------|----------------------------|
| elbow ROM | 0 -> 80 deg | 20 -> 120 deg fresh, declining with fatigue |
| shoulder | 0 deg | 20 deg fixed stabilization |
| repetitions | 10 identical cycles | configurable; 30-100 recommended for datasets |
| movement model | minimum jerk only | minimum jerk + physiological fatigue perturbation |
| fatigue | none | slower, smaller ROM, asymmetric, tremor, endpoint drift |
| noise | none | band-limited tremor/drift, not white noise |
| derivatives | not provided | consistent finite differences of written signal |
| validation | manual inspection | automatic QC report |
| subject variability | none | multi-subject parameter randomization |
| OpenSim formats | `.mot` | `.mot` + `.sto` |

---

## 11. Remaining Honest Limitations

The improvements make Stage 1 much stronger, but it is still synthetic data.

- The QC compares against theory and literature ranges, not measured motion
  capture or EMG.
- Subject variability is parametric, not learned from a real population.
- The motion is elbow-focused and does not include full trunk, scapula, wrist,
  or hand compensation.
- True experimental validation would require real elbow-curl kinematics,
  torques, and preferably EMG or fatigue measurements.

So the correct claim is:

```text
Stage 1 generates controlled, physiologically plausible synthetic elbow-curl
motion for OpenSim and AI dataset construction. It is not a replacement for
experimental motion capture, but it is reproducible, validated against
minimum-jerk theory, and suitable as a synthetic simulation foundation.
```

---

## 12. Pipeline Role

```text
Stage 1: joint angles over time
    -> Stage 2: OpenSim torque, muscle force, activation estimates
    -> Stage 3: AI fatigue prediction / classification dataset
```

Stage 1 is the foundation. If the motion is not physiologically plausible, every
downstream OpenSim and AI result inherits that weakness. The current improved
generator gives a stronger foundation by combining minimum-jerk theory,
controlled fatigue degradation, optional ID-safe filtering, automatic QC, and
multi-subject variability.

---

## References

### Paper-to-Generator Mapping

| Generator component | Main supporting papers |
|---------------------|------------------------|
| minimum-jerk baseline and bell-shaped velocity | Hogan (1984); Flash & Hogan (1985) |
| saturating fatigue state | Xia & Frey Law (2008); Frey-Law et al. (2012) |
| fatigue resistance and task variability | Frey-Law & Avin (2010); Potvin (2012) |
| dynamic fatigue signatures during repeated contractions | Potvin (1997); Potvin & Bent (1997) |
| endpoint drift and movement variability under fatigue | Savin et al. (2021); Yang et al. (2018); Sheikhhoseini et al. (2025) |

The entries below list the paper titles used for those components.

### Minimum-Jerk Motion Generation

- Hogan, N. (1984). **An organizing principle for a class of voluntary
  movements.** Journal of Neuroscience.  
  Used here as the theoretical basis for smooth, jerk-minimizing human movement.

- Flash, T., & Hogan, N. (1985). **The coordination of arm movements: an
  experimentally confirmed mathematical model.** Journal of Neuroscience.  
  Used here as the main reference for the fifth-order minimum-jerk trajectory and
  bell-shaped velocity profile.

### Fatigue Curve / Capacity Model Inspiration

- Xia, T., & Frey Law, L. A. (2008). **A theoretical approach for modeling
  peripheral muscle fatigue and recovery.** Journal of Biomechanics.  
  Used here as inspiration for a saturating fatigue state instead of a purely
  linear fatigue curve.

- Frey-Law, L. A., Avin, K. G. (2010). **Endurance time is joint-specific: a
  modelling and meta-analysis investigation.** Ergonomics.  
  Used here to justify subject/task variability in fatigue resistance.

- Frey-Law, L. A., Looft, J. M., & Heitsman, J. (2012). **A three-compartment
  muscle fatigue model accurately predicts joint-specific maximum endurance times
  for sustained isometric tasks.** Journal of Biomechanics.  
  Used here as a 3CC-style reference for fatigue accumulation and plateau-like
  endurance behavior.

### Dynamic EMG / Repetitive-Task Fatigue Signatures

- Potvin, J. R. (1997). **Effects of muscle kinematics on surface EMG amplitude
  and frequency during fatiguing dynamic contractions.** Journal of Applied
  Physiology.  
  Used here to justify dynamic fatigue signatures during elbow flexion-extension:
  altered EMG amplitude/frequency, speed effects, and fatigue-dependent motion
  changes.

- Potvin, J. R., & Bent, L. R. (1997). **A validation of techniques using surface
  EMG signals from dynamic contractions to quantify muscle fatigue during
  repetitive tasks.** Journal of Electromyography and Kinesiology.  
  Used here to justify using dynamic/repetitive-task fatigue markers rather than
  relying only on static fatigue assumptions. Some project notes previously
  referred to this line as Potvin & Bent 2010; bibliographic databases list this
  dynamic-EMG validation paper as 1997.

- Potvin, J. R. (2012). **Predicting maximum acceptable efforts for repetitive
  tasks: an equation based on duty cycle.** Human Factors.  
  Used here as additional support for treating repetition count, duty cycle, and
  work-rest structure as important fatigue-design variables.

### Kinematic Fatigue / Movement Variability

- Savin, J., Gaudez, C., Gilles, M. A., Padois, V., & Bidaud, P. (2021).
  **Evidence of movement variability patterns during a repetitive pointing task
  until exhaustion.** Applied Ergonomics.  
  Used here to justify fatigue-related endpoint drift and movement-variability
  patterns during repetitive upper-limb tasks. This is the paper title that
  matches the project note about successive coordination/variability patterns in
  repetitive pointing until exhaustion.

- Sheikhhoseini, R., Abdollahi, S., Salsali, M., & Anbarian, M. (2025).
  **Biomechanical coordination and variability alters following repetitive
  movement fatigue in overhead athletes with painful shoulder.** Scientific
  Reports.  
  Used here to justify post-fatigue changes in coordination, shoulder/trunk
  strategy, and movement variability.

- Yang, C., Bouffard, J., Srinivasan, D., Ghayourmanesh, S., Cantu, H., Begon,
  M., & Cote, J. N. (2018). **Changes in movement variability and task
  performance during a fatiguing repetitive pointing task.** Journal of
  Biomechanics.  
  Used here as additional support for evaluating fatigue trends through movement
  variability and task-performance changes.