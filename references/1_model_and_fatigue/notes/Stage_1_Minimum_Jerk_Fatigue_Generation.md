# Stage 1 - Minimum-Jerk Fatigue Motion Generation

## Goal

Generate physiologically plausible repetitive elbow-flexion trajectories without real motion capture. The output is used as the synthetic motion driver for the Arm26 OpenSim workflow and as the first layer of the fatigue dataset.

## Existing Baseline

The existing file at `Data/paper_minjerk_10cycles.mot` is a clean baseline:

- 100 Hz sampling.
- 10 repetitions.
- `r_shoulder_elev = 0 deg` fixed.
- `r_elbow_flex` moves from `0 deg` to `80 deg` and back.
- Each cycle uses fixed timing and has no fatigue degradation.

That is useful for validating OpenSim, but it does not yet represent progressive fatigue.

## New Stage-1 Version

The new generator is `Code/generate_stage1_minjerk_fatigue.py`.

Default output files:

- `Data/paper_minjerk_fatigue_30cycles.mot`
- `Data/paper_minjerk_fatigue_30cycles_full.csv`
- `Data/paper_minjerk_fatigue_30cycles_reps.csv`
- `Data/paper_minjerk_fatigue_30cycles_meta.json`

The `.mot` file is kept OpenSim-compatible with only:

```text
time    r_shoulder_elev    r_elbow_flex
```

The full `.csv` stores the richer Stage-1 variables: velocity, acceleration, jerk, repetition index, phase, target angle, perturbation, and fatigue level.

## Core Equation

Each movement segment uses the classical fifth-order minimum-jerk polynomial:

```text
q(t) = q0 + (qf - q0) * (10*s^3 - 15*s^4 + 6*s^5)
s = t / T
```

where:

- `q0` is the start elbow angle.
- `qf` is the target elbow angle.
- `T` is the segment duration.
- `s` is normalized time from 0 to 1.

This shape gives zero endpoint velocity and acceleration, smooth acceleration/deceleration, and a bell-shaped velocity profile.

## Repetition Structure

Each repetition contains four phases:

```text
20 deg -> target deg    flexion, minimum jerk
target deg             top pause
target deg -> 20 deg    extension, minimum jerk
20 deg                 bottom pause
```

Default fresh movement:

- Initial elbow angle: `20 deg`
- Fresh target angle: `120 deg`
- Flexion duration: `2.0 s`
- Extension duration: `2.0 s`
- Top pause: `0.2 s`
- Bottom pause: `0.2 s`
- Shoulder stabilization: `20 deg`
- Sampling rate: `100 Hz`
- Repetitions: `30`

## Fatigue Model

Fatigue is represented by a normalized value:

```text
fatigue_level = (rep - 1) / (repetitions - 1)
```

So the first repetition is `0.0` and the last repetition is `1.0`.

Progressive changes:

- Slower movement: flexion duration increases from `2.0 s` to `3.0 s`.
- Reduced range of motion: target angle decreases from `120 deg` to `100 deg`.
- Asymmetry: flexion slows more than extension, so the up phase becomes longer than the down phase.
- Reduced smoothness: tremor and small random perturbations increase with fatigue.

Perturbations are multiplied by `sin(pi*s)^2`, so they fade to zero at each movement endpoint. This preserves stable OpenSim boundary behavior while still reducing smoothness inside each movement.

## Why Two Minimum-Jerk Segments

A full elbow curl is not one polynomial. It is better represented as:

- one minimum-jerk segment for concentric flexion,
- one minimum-jerk segment for eccentric extension.

This allows fatigue to affect the phases differently, which is important because fatigued motion often becomes asymmetric.

## How To Run

From the workspace root:

```powershell
".venv/Scripts/python.exe" Code/generate_stage1_minjerk_fatigue.py
```

Optional examples:

```powershell
".venv/Scripts/python.exe" Code/generate_stage1_minjerk_fatigue.py --reps 50 --out-stem paper_minjerk_fatigue_50cycles
".venv/Scripts/python.exe" Code/generate_stage1_minjerk_fatigue.py --q0 20 --qf-fresh 120 --qf-fatigued 95 --shoulder 20
```

## Pipeline Meaning

Stage 1 outputs joint angles over time. These angles can then be used by OpenSim stages to estimate kinematics, inverse dynamics, static optimization, muscle activation, and fatigue-related dataset variables.

In thesis language: minimum-jerk trajectories provide the fresh physiological baseline, and fatigue perturbations provide controlled degradation of timing, range of motion, smoothness, and movement symmetry.