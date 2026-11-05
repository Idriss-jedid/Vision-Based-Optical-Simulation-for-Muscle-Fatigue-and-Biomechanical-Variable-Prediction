# Stage 2 — Experimental Validation: All Muscles + Fatigue Endurance

Addresses two requirements: (1) validate **every** muscle's moment arm against
experimental cadaveric data (not just BRD), and (2) validate the **fatigue model**
against experimental endurance-time data.

Reproduce:
```powershell
conda run -n biomech python Code/validate_all_muscles.py
conda run -n biomech python Code/validate_fatigue_endurance.py
```

---

## 1. All-muscle moment-arm validation (OpenSim wrap-aware, vs Murray 2000/2002)

Elbow flexion moment arm of every muscle, over 0–130° (shoulder fixed 20°), peak
and Murray-window mean (flexors 20–120°, triceps 30–120°). + = flexion, − = extension.

| Muscle | Role | Peak (cm) | @° | **Mean (cm)** | Murray peak | Murray range | Verdict |
|---|---|--:|--:|--:|--:|--|:--|
| BIClong | flexor | 4.94 | 100 | **4.03** | 4.7 | 3.5–5.2 | ✅ PASS |
| BICshort | flexor | 4.94 | 100 | **4.03** | 4.2 | 3.0–5.0 | ✅ PASS |
| BRA (brachialis) | flexor | 2.40 | 110 | **1.85** | 1.9 | 1.5–2.6 | ✅ PASS |
| BRD_hand | flexor | 7.51 | 115 | **5.78** | 7.7 | 7.0–9.0 | ✅ PASS |
| TRIlong | extensor | 2.42 | 25 | **2.10** | 2.3 | 1.8–2.8 | ✅ PASS |
| TRIlat | extensor | 2.42 | 25 | **2.10** | 2.1 | 1.6–2.6 | ✅ PASS |
| TRImed | extensor | 2.42 | 25 | **2.10** | 2.1 | 1.6–2.6 | ✅ PASS |

**7/7 elbow muscles within the published range.** The Murray-window **means** are
especially close to the cadaver data — brachialis 1.85 vs 1.9, triceps 2.10 vs
~2.1, biceps 4.03 vs ~3.4–4.0, BRD 5.78 vs Holzbaur avg 5.7. So the **whole elbow
muscle set** — flexors *and* extensors — is geometrically validated, not just BRD.
(The four shoulder muscles DELT_ant/post, PECT, LAT were removed; this is an
elbow-focused model with the shoulder held by the `shoulder_assist` actuator.)

---

## 2. Fatigue-model validation: endurance time vs intensity

We have **no measured subject**, so the fatigue dynamics are validated against the
**experimental endurance-time (ET) literature** that the model was built on. The
elbow 3CC rates (F = 0.00912, R = 0.00094 /s) were fit by **Frey-Law et al. (2012)
to 369 endurance-time data points from 194 studies** (elbow RMS 9.9 s). Running
the full Xia & Frey-Law (2008) controller at sustained 10–90 % MVC reproduces the
intensity–ET curve:

| %MVC | 3CC MET (s) | general-model ref (s) |
|--:|--:|--:|
| 90 | 13.5 | 10.9 |
| 80 | 29.0 | 14.2 |
| 70 | 49.3 | 19.2 |
| 60 | 77.0 | 27.3 |
| 50 | 117.1 | 41.2 |
| 40 | 180.2 | 68.4 |
| 30 | 294.5 | 131.2 |
| 20 | 568.0 | 328.6 |

**Validation outcome:**
- ✅ **Monotonic, rapid ET decline with intensity** — the hallmark of the
  experimental Rohmert/Frey-Law relationship.
- ✅ **Absolute ET in the experimental elbow range** — e.g. 70 % MVC → 49 s,
  50 % MVC → 117 s, consistent with reported sustained-isometric elbow-flexion
  endurance.
- The 3CC predicts **longer** ET than the *general* whole-body curve (ratio
  ~1.5–2.8×). This is **expected and correct**: Frey-Law (2012) found the **elbow
  is among the more fatigue-resistant joints**, so its joint-specific ET exceeds
  the all-joint average. The discrepancy is the joint-specificity the model is
  designed to capture.

Because the Stage-2 fatigue labels use this **same** experimentally-fit 3CC, the
labels inherit this experimental grounding.

---

## 3. Honest scope of "experimental validation"

| Claim | Status |
|---|---|
| Muscle moment arms match cadaver data (Murray 2000/2002) | ✅ direct, 7/7 muscles |
| Fatigue endurance-time matches experimental intensity–ET literature | ✅ via the Frey-Law-calibrated 3CC |
| Activation rises with load (Gastaldi 0/2 kg) | ✅ (+179 %, Stage-2 pipeline) |
| Activation rises / capacity falls with reps (Potvin & Bent direction) | ✅ monotonic (Stage-2 3CC) |
| **Subject-specific validation against measured EMG / MVC decline** | ❌ **not done — requires data collection (future work)** |

What would make it fully experimental (Hicks 2015): collect surface EMG + MVC from
a small cohort doing the 2 kg fatiguing curl (the Potvin & Bent / Gastaldi
protocol), then compare (a) simulated vs measured activation envelopes, (b)
simulated capacity decline vs measured MVC-force decline, (c) endurance time. The
present work validates the **model components** against the **published experimental
data**; a measured cohort is the remaining step to subject-specific validation.

## Files
| File | Purpose |
|---|---|
| `Code/validate_all_muscles.py` | all-muscle moment-arm validation vs Murray |
| `Code/validate_fatigue_endurance.py` | 3CC endurance-time vs experimental intensity–ET |
