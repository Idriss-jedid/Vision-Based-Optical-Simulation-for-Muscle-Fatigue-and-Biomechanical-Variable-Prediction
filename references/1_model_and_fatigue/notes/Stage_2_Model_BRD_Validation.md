# Stage 2 — OpenSim Model: Audit, BRD Fix & Validation

This documents the Stage-2 musculoskeletal model used as the physics-based label
generator: what the existing model contained, the one defect that was found and
fixed, and how it was validated.

- **Input model:** `Model/arm26_paper_loaded.osim`
- **Validated full model:** `Model/arm26_paper_loaded_brd.osim`
- **Recommended elbow-study model:** `Model/arm26_paper_loaded_brd_elbow_research.osim` ✅ (shoulder muscles removed; elbow-focused)
- **Validator:** `Code/validate_brd_moment_arm.py` and `Code/validate_model_opensim.py`
- **Motion driver:** `Data/paper_minjerk_fatigue_10cycles.mot`

---
## 1. Audit of the existing model

`arm26_paper_loaded.osim` (OpenSim 4.6) was inspected component by component
against the Stage-2 requirements. Most of it was already correct:

| Requirement | Status in existing model |
|---|---|
| Base arm26: 2 DOF (`r_shoulder_elev`, `r_elbow_flex`), Thelen2003 muscles | ✅ present |
| 6 stock muscles (TRIlong/lat/med, BIClong, BICshort, BRA) | ✅ present |
| Shoulder muscles (DELT_ant/post, PECT, LAT) | ⊘ removed in the elbow-focused model (shoulder held by `shoulder_assist`) |
| **2 kg dumbbell** welded to the hand | ✅ `load_2kg` (mass 2 kg) via `load_weld` WeldJoint on `r_ulna_radius_hand` at (0.028, −0.305, 0.082) — correct, gravity/inertia propagate through ID/SO |
| CoordinateActuators for residual support | ✅ `shoulder_assist` (50), `elbow_assist` (30) |
| Markers (acromion, epicondyle, radial styloid) | ✅ present |
| **Brachioradialis (BRD)** added & validated vs Murray | ⚠️ **present but NOT valid** — see §2 |

So the dumbbell and actuator extensions were already done correctly. The only
real problem was the brachioradialis.

---

## 2. The BRD defect

The model contained a `BRD_hand` muscle, but it failed validation on two counts.

**(a) Moment arm was wrong.** Using the virtual-work identity
`r(θ) = −dL/dθ` (reconstructed from the elbow joint axis and the muscle path —
see the validator), the existing BRD moment arm was:

```
PEAK 2.46 cm @ 70°   (average 1.91 cm)     VERDICT: NEEDS TUNING
```

That is ~3× too small and peaks ~40° too early — it behaved like a second
brachialis (BRA peaks 2.4 cm @ 110°), not a brachioradialis. The whole point of
adding BRD (Murray 2000/2002: it has the **largest** elbow-flexor moment arm,
7.7 cm) was therefore not achieved, so it could not relieve the biceps/brachialis
the way the thesis argues.

**(b) Physiology parameters did not match the literature.**

| Parameter | Old value | Holzbaur 2005 | Used now |
|---|---|---|---|
| max isometric force | 320 N | 261.3 N | **261.3 N** |
| optimal fiber length | 0.11 m | 0.173 m | **0.173 m** |
| tendon slack length | 0.17 m | 0.133 m | **0.165 m** (calibrated, see §3) |
| pennation angle | 0.12 rad | 0° | **0°** |

*Method check:* the same calculator reproduces the known arm26 **BIClong
4.94 cm @ 100°** and **BRA 2.40 cm @ 110°** moment arms, confirming it is correct.

---

## 3. The fix

Only the BRD muscle was changed; nothing else in the model was touched.

**Geometry.** The moment arm of BRD depends only on the *crossing* segment
(origin on humerus → first forearm point), because all remaining path points are
rigid on the forearm. Those two points were re-optimised to reproduce the Murray
curve, within anatomically constrained boxes (origin on the lateral supracondylar
ridge; first forearm point anterolateral):

| Point | Frame | New location (m) | Anatomy |
|---|---|---|---|
| `BRD_hand-P1` | r_humerus | `0.0088, −0.1806, 0.0004` | lateral supracondylar ridge (~11 cm proximal to elbow) |
| `BRD_hand-P2` | r_ulna_radius_hand | `0.0234, −0.0744, 0.0440` | anterolateral proximal forearm (bowstring point that sets the moment arm) |

The remaining points P3–P6 route the muscle down to the radial styloid; they are
cosmetic for the moment arm but needed so the tendon physically reaches the
insertion.

**Tendon slack length calibration.** With the literature `Lts = 0.133 m`, the
long arm26 humerus→styloid path overstretched the fiber (1.42·Lopt at 20°). Tendon
slack length is the standard parameter to calibrate per model, so it was set to
`Lts = 0.165 m`, which centres the fiber at its optimal length near mid-ROM. Peak
force, optimal fiber length and pennation are kept at the Holzbaur values.

---

## 4. Validation result

`python Code/validate_brd_moment_arm.py Model/arm26_paper_loaded_brd_elbow_research.osim`

```
 elbow_deg   moment_arm_cm
     20        2.85
     40        4.28
     60        5.56
     80        6.60
     90        7.01
    100        7.32
    110        7.49
    120        7.47
 PEAK 7.51 cm @ 115°   |   average 5.31 cm
 Murray target: peak 7.7 cm (7.0-9.0) @ ~108 deg (100-118), avg ~5.4-5.7 cm
 VERDICT: VALIDATED
```

The curve rises monotonically from ~2.9 cm at 20° to a 7.5 cm peak at ~110–115°,
inside Murray's reported range (7.0–9.0 cm at 100–118°), with an average
(5.3 cm) close to the Holzbaur model average (5.7 cm).

**Confirmed with the OpenSim API** (`biomech` env, OpenSim 4.5.1): the model
loads and `initSystem()` succeeds, and OpenSim's own
`GeometryPath.computeMomentArm` (which accounts for wrap objects) reproduces the
identical BRD curve — **peak 7.51 cm @ 115°, VALIDATED** — and the textbook
flexor ordering BRD (7.5) > biceps (4.9) > brachialis (2.4). Reproduce with
`conda run -n biomech python Code/validate_model_opensim.py`.

| | Old BRD | New BRD | Murray target |
|---|---|---|---|
| Peak moment arm | 2.46 cm | **7.51 cm** | 7.7 cm (7–9) |
| Angle of peak | 70° | **115°** | ~108° (100–118) |
| Verdict | FAIL | **VALIDATED** | — |

---

## 5. Honest caveats

- **Best model choice for the current thesis scope.** The validated full model is
  preserved as `Model/arm26_paper_loaded_brd.osim`. For the loaded elbow-flexion
  study, use `Model/arm26_paper_loaded_brd_elbow_research.osim`: it is identical
  in elbow flexors, BRD path, dumbbell mass/inertia/weld, and coordinates, but the
  four shoulder muscles (DELT_ant/post, PECT, LAT) are **removed**. This is more
  defensible than retuning their non-physiological paths, because none cross the
  elbow and the fixed shoulder is already supported by `shoulder_assist`.
- **Moment-arm validation is exact; the wrap object is a modeling choice.** The
  thesis text mentions a *wrapping cylinder*; here the same validated moment-arm
  curve is achieved with a tuned via-point path (a standard, fully legitimate
  OpenSim technique). A wrap cylinder could be substituted later, but it would
  have to be re-tuned to preserve this curve. Update the thesis wording to
  "via-point path validated against Murray" or add a wrap cylinder in the GUI.
- **OpenSim API validation is available in the `biomech` environment.** Run
  `conda run -n biomech python Code/validate_model_opensim.py` to reproduce the
  wrap-aware OpenSim check. In this workspace it loaded the model, built the
  system, and confirmed the BRD curve. OpenSim did warn that the original Arm26
  `.vtp` mesh files are not in the workspace geometry search path; that affects
  GUI visualization, not model dynamics. Before production, open
  `arm26_paper_loaded_brd_elbow_research.osim` in the OpenSim GUI / OpenSim
  Creator and:
  1. confirm the BRD path looks anatomically clean in 3D once geometry assets are
     available,
  2. run Tools → Analyze → MuscleAnalysis over `r_elbow_flex` and confirm the
     moment-arm plot matches §4,
  3. run Inverse Dynamics + Static Optimization with the `.mot` and check that
     BRD now carries a share of the flexion load (biceps/brachialis activation
     should drop relative to the old model).
- Fiber operating range under the calibrated `Lts` spans ~0.6–1.2·Lopt across
  20–120°, which is wide but physiologically appropriate for BRD's long fascicles.

---

## 6. Files

| File | Purpose |
|---|---|
| `Model/arm26_paper_loaded_brd_elbow_research.osim` | recommended model for the loaded elbow-flexion study |
| `Model/arm26_paper_loaded_brd.osim` | validated full BRD+dumbbell model kept for reference/future shoulder retuning |
| `Model/arm26_paper_loaded.osim` | previous model (BRD un-validated) — kept for reference |
| `Code/validate_brd_moment_arm.py` | reproducible moment-arm validator |
| `Code/validate_model_opensim.py` | OpenSim API validator for the recommended model |
