# Minimum-Jerk Fatigue Motion — Version Comparison

Three versions of the synthetic elbow-flexion motion were generated while
developing the Stage-1 pipeline. This document analyses the differences and
states the recommended file.

All three: 100 Hz, 10 reps, elbow 20°→120° base ROM, shoulder fixed at 20°,
columns `time, r_shoulder_elev, r_elbow_flex`, reproducible seed = 42.

| File | Short name |
|------|-----------|
| `paper_minjerk_fatigue_10cycle_first.mot`   | **FIRST**  |
| `paper_minjerk_fatigue_10cycles_seconde.mot`| **SECOND** |
| `paper_minjerk_fatigue_10cycles.mot`        | **FINAL** ✅ recommended |

---

## 1. Measured differences

| Metric | FIRST | SECOND | FINAL |
|--------|------:|-------:|------:|
| Rows / duration | 3999 / 40.0 s | 3999 / 40.0 s | 4202 / 42.0 s |
| Elbow min (deg) | 18.31 | 19.83 | 17.89 |
| Elbow max (deg) | 120.00 | 120.00 | 120.10 |
| Samples below 20° | 140 | 73 | 221 |
| Samples above 120° | 0 | 0 | 9 |
| Max single-step jump (deg) | 2.23 | 2.23 | 2.43 |
| RMS-jerk **fresh** (deg/s³) | 25 637 | 19 176 | 35 825 |
| RMS-jerk **fatigued** (deg/s³) | 489 730 | 311 927 | 174 792 |
| Fresh→fatigued jerk ratio | ×19.1 | ×16.3 | **×4.9** |

Per-rep **structural** peak angle (the commanded ROM, before drift):

| Rep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-----|--|--|--|--|--|--|--|--|--|--|
| FIRST / SECOND (linear)   | 120 | 117.8 | 115.6 | 113.3 | 111.1 | 108.9 | 106.7 | 104.4 | 102.2 | 100 |
| FINAL (saturating)        | 120 | 114.7 | 110.7 | 107.7 | 105.4 | 103.6 | 102.3 | 101.3 | 100.6 | 100 |

---

## 2. What each version is

### FIRST — raw, un-faded
- **Fatigue curve:** linear (`f = k/(N−1)`).
- **Noise:** a single 6 Hz tremor sinusoid **+ white Gaussian noise** (σ up to 0.3°), added **only while moving**.
- **Pros:** looks convincingly "human" — jagged, endpoints drift below the start angle (140 samples < 20°).
- **Cons (real bugs):**
  - **Discontinuity at every move→hold boundary** — tremor is present while moving but switches off during the holds, so the angle jumps when a stroke ends.
  - **White noise is unphysical** — uncorrelated per-sample noise means effectively infinite-bandwidth jerk (note the ×19 jerk explosion).
  - **Its companion CSV was inconsistent** — velocity/acceleration were the *clean* polynomial values, mismatching the noisy angle by up to 113 deg/s.

### SECOND — enveloped
- **Fatigue curve:** linear.
- **Noise:** same 6 Hz + white noise, but multiplied by an **endpoint-fade envelope** `sin(πs)`, added only while moving.
- **Derivatives:** recomputed by finite difference (CSV now consistent).
- **Pros:** clean, bounded — angle never overshoots [20°,120°] (only 73 samples < 20°, all shallow), no boundary spikes, consistent labels.
- **Cons:**
  - The envelope **boxes the motion into the nominal ROM** and kills the endpoint/turnaround variability that real fatigue produces — arguably *too* clean.
  - Still uses **white noise** (unphysical spectrum, ×16 jerk explosion).
  - Linear fatigue is a crude progression model.

### FINAL — physiologically faithful ✅
- **Fatigue curve:** **saturating** `f = (1−e^(−r·x))/(1−e^(−r))`, echoing the Xia & Frey-Law (2008) 3-compartment endurance shape — fatigue rises then plateaus.
- **Noise:** **band-limited tremor (6–12 Hz)** + a separate **slow endpoint drift (0.1–0.5 Hz)**, each a sum of sinusoids, amplitude scaled by `f`, applied **continuously over the whole timeline (holds included, like real postural tremor)**.
- **Derivatives:** finite difference of the final angle — CSV matches the `.mot` exactly (0.0000 mismatch).
- **Pros:**
  - **No discontinuities** — tremor is continuous everywhere; the 2.43°/step max is *real* 12 Hz tremor slope, not a jump.
  - **Endpoint drift restored, but modeled honestly** — the turnaround wander the FIRST version had by accident is now an explicit, controllable component (221 samples drift below 20°).
  - **Physiological spectrum** — band-limited noise gives a realistic ×4.9 jerk increase instead of the white-noise ×16–19 artifact.
  - **Rep 1 is exactly fresh** (`f = 0`, pure min-jerk) and degradation grows smoothly.
  - **Consistent CSV + `.sto`** output for the AI / OpenSim stages.
- **Cons:**
  - Over only 10 reps the saturating curve **front-loads** the ROM loss (big drop by rep 5). For a gentler decline lower `FATIGUE_RATE`; for a real dataset raise `N_REPS` to 30–100.
  - Drift can push the commanded peak a little past 120° (9 samples) — realistic variability, but disable `DRIFT_AMP_DEG` if a hard ROM cap is required.

---

## 3. Key technical differences at a glance

| Aspect | FIRST | SECOND | FINAL |
|--------|-------|--------|-------|
| Fatigue progression | linear | linear | **saturating (3CC-like)** |
| Tremor model | 6 Hz + **white noise** | 6 Hz + white noise | **band-limited 6–12 Hz** |
| Endpoint drift | accidental | removed | **explicit 0.1–0.5 Hz** |
| Tremor during holds | no | no | **yes (postural)** |
| Boundary continuity | ❌ jumps | ✅ | ✅ |
| CSV vel/acc consistent | ❌ | ✅ | ✅ |
| Endpoint variability | yes (uncontrolled) | none (boxed) | **yes (controlled)** |
| Jerk realism | white→×19 | white→×16 | **band→×4.9** |
| `.sto` output | no | no | **yes** |

---

## 4. Recommendation

**Use FINAL — `paper_minjerk_fatigue_10cycles.mot`.**

It keeps the human-like quality you liked in FIRST (endpoint drift, visible
tremor, degrading smoothness) but fixes the three things that were actually
wrong with the earlier versions:

1. it has **no artificial discontinuities** (FIRST jumped at every stroke end);
2. its tremor is **band-limited and physiological**, not white noise (which made
   FIRST/SECOND unrealistically jerky and would inject noise straight into the
   OpenSim inverse-dynamics torques);
3. its **dataset CSV exactly matches the motion**, so the Stage-3 AI never trains
   on labels that contradict the trajectory.

It also uses a **saturating fatigue curve** grounded in the Xia & Frey-Law
endurance model rather than a straight line, and additionally exports a `.sto`.

**When you might pick another:**
- **SECOND** if you need a strictly ROM-bounded signal (no overshoot past
  20°/120°) for a sensitivity test — at the cost of realistic endpoint drift.
- **FIRST** only as a historical reference of the initial draft; its CSV is not
  trustworthy and it has boundary discontinuities.

**Before generating the production dataset:** in the FINAL generator
(`generate_minjerk_motion.py`) set `N_REPS = 30–100` and consider lowering
`FATIGUE_RATE` (~1.5) for a more gradual decline.
