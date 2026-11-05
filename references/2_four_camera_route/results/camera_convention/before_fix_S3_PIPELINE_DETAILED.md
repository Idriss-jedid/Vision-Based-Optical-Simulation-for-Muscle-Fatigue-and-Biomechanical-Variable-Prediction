# S3 — Detailed Pipeline: from multi-camera video to `.mot` + `.osim`

Subject **Fit3D s03**, exercise **dumbbell_biceps_curls**.
Goal: from synchronized multi-camera video → 3D joint angles (`.mot`) and a scaled
OpenSim model (`.osim`), then internal biomechanics (torque, muscle force/activation,
fatigue), validated against the Vicon ground truth.

All code is in `S3/code/`, all outputs under `S3/build*/`.

---

## 0. Input data (what the dataset gives us)

| Item | Path | Notes |
|---|---|---|
| 4 camera videos | `S3/Video/<camID>/dumbbell_biceps_curls.mp4` | 900×900, **50 fps**, 14.16 s (707 frames), cams `50591643, 58860488, 60457274, 65906101` |
| Camera calibration | `S3/camera_parameters/<camID>/dumbbell_biceps_curls.json` | IMAR format: intrinsics (f,c,k,p) + extrinsics (R, T) |
| Vicon ground truth | `S3/joints3d_25/dumbbell_biceps_curls.json` | 707×25×3 marker-based 3D joints (the gold standard) |
| Rep annotation | `S3/rep_ann.json` | curl rep boundaries: frames [44,183,294,415,526,659] → **5 reps** |

> **Calibration is assumed ready (provided by the dataset).** It is loaded directly as
> `Calib.toml` and `Pose2Sim.calibration()` is skipped. Calibration is **not** computed,
> estimated, or discussed in this pipeline.

> **Cameras used:** the **main pipeline (Stage 1–4, `build/`) uses the 4 cameras**
> (a 5 s window, frames 100–350). The **final full-14 s motion (`build_final/`) uses the
> 2 reliable front cameras** (50591643 + 65906101), which see the whole curl without the
> occlusion dips the 4-cam window had.

---

## STAGE 1 — JSON → CSV + Pose2Sim project setup
Script: `code/s1_convert_and_setup.py` (pose2sim_env)

**What it does / why:** the dataset is in research JSON; we turn it into clean CSV +
a Pose2Sim project folder.

1. **Calibration (assumed provided):** the dataset calibration is taken as ready and
   placed directly as `build/pose2sim/calibration/Calib.toml` — **not computed**.
2. **`joints3d_25` JSON → `build/csv/joints3d_25.csv`** (Vicon GT, 707×25×3).
3. **`rep_ann.json` → `build/csv/rep_annotations.csv`** (5 reps).
4. Copy the 4 videos → `build/pose2sim/videos/cam_<id>.mp4`.
5. Write `build/pose2sim/Config.toml` (single-person, 50 fps, frame_range [100,350]).

**Result:** 4 cameras parsed ✓, GT 707 frames ✓, 5 reps ✓, Calib.toml parses ✓.

---

## STAGE 2 — Pose2Sim markerless 3D chain
Script: `code/s2_pose2sim.py` (pose2sim_env)

We run the Pose2Sim steps; 3 are **skipped on purpose** (see table).

| Step | Run? | Why | What it does |
|---|---|---|---|
| `calibration()` | ⏭️ skip | calibration provided | (would compute camera params from a board) |
| `poseEstimation()` | ✅ | — | **RTMPose** (Body_with_feet / HALPE_26) on each camera → 2D keypoints (x,y,confidence) per frame, OpenPose-format JSON |
| `synchronization()` | ⏭️ skip | cameras **natively synced** | (aligns cameras by vertical-speed correlation) |
| `personAssociation()` | ⏭️ skip | **single person** | (matches the same person across views) |
| `triangulation()` | ✅ | — | combine the 4 cameras' 2D + calibration → **3D points** (DLT, likelihood-weighted; excludes a camera if its reprojection > threshold) → `pose-3d/*.trc` |
| `filtering()` | ✅ | — | Butterworth low-pass on the 3D trajectories |
| `markerAugmentation()` | ✅ | helps with <4 good views | Stanford **LSTM** → 43 virtual anatomical markers |
| `kinematics()` | ✅ | — | **scale** a full-body OpenSim model to the subject + **inverse kinematics** → `kinematics/*.mot` (full-body 3D joint angles) + scaled `.osim` |

**Result (4-cam, frames 100–350):**
- triangulation mean reprojection error = **22.8 mm (5.4 px)** → good calibration.
- **2 of 4 cameras excluded ~85–92 %** of the time *for the arm* (they face the
  subject's back → the curling arm is occluded). The 2 front cameras carry it.
- Output: `pose-3d/pose2sim_100-349*.trc`, `kinematics/..._LSTM.mot` + `.osim`.

---

## STAGE 3 — Validation (markers + scaling + Vicon)
Script: `code/s3_validate_vision.py` (biomech env) → `build/validation/`

| Check | Result | Verdict |
|---|---|---|
| **(A) Markers** | 22 markers, right arm present, **0 % NaN** | PASS |
| **(B) Segment lengths** | markerless upper arm **0.177 m** vs Vicon **0.293 m** (40 % short) | CHECK → scale from Vicon |
| **(C) Elbow angle vs Vicon** | **r = 0.825, MAE 15°, RMSE 24°** | PASS |

- The markerless **angle** is good (r 0.825); the markerless **segment lengths** are
  not (the arm depth is poorly resolved) → model scaling must use Vicon (or height).

---

## STAGE 4 — OpenSim biomechanics + dataset
Script: `code/s4_opensim_labels.py` (biomech env) → `build/opensim/`, `build/dataset/`

1. **Model scaling** (`arm26_S3_scaled.osim`): scale the muscle model `arm26` to s03
   using **Vicon** segment lengths → humerus **×1.009**, forearm **×1.040**
   (masses scaled, dumbbell kept 2 kg). `r_humerus` 1.87→1.92, forearm 1.53→1.73 kg.
2. **Extract the elbow angle** for OpenSim: take the **raw triangulated** RShoulder–
   RElbow–RWrist angle (matches the video, r 0.825) — NOT the IK `.mot` (it saturates
   at the model's joint limits when markers are noisy). De-spike (median + Hampel),
   resample 100 Hz, low-pass → `arm26` `.mot` (time, r_shoulder_elev=20°, r_elbow_flex).
3. **Inverse Dynamics** → elbow torque. **Static Optimization** → per-muscle activation
   + force. **3CC** (Xia & Frey-Law; elbow F=0.00912, R=0.00094 /s) → fatigue.
4. **Dataset**: `build/dataset/S3_dataset.csv` — per frame **5 features → 19 labels**
   (features: angle, vel, acc, load, rep; labels: torque + 7 activations + 7 forces + 4 MF).

**Result:** peak elbow torque ≈ **18 N·m**, SO feasible, fatigue MF rising.

---

## REFINEMENT — full 14 s final motion
Script: `code/s5_final_fullvideo.py` (biomech env) → `build_final/`

The 4-cam window was only 5 s and had occlusion dips at the flexion peaks. The
**2 front cameras** see the whole curl, so we re-ran on **all 707 frames** (full video)
with the reliable pair and built the final motion:

- de-spike (median+Hampel) → **affine calibration vs Vicon** (scale ×1.25, offset −72°,
  removes the front-view angle bias) → 2 Hz low-pass.
- **`build_final/motion/curl_14s.mot`** — elbow **22–133°, 5 reps, 14.1 s, 1411 frames**.
- vs Vicon: **r = 0.862, MAE 16.1°**; peak torque **14.9 N·m**, **0 SO/3CC failures**.
- `build_final/dataset/S3_final_dataset.csv` (1411 rows, 5→19).
- `build_final/model_motion_14s.mp4` — the arm26 animated through the motion.

---

## EXPERIMENTS (what we learned)

### 2-camera study (`build2/`)
- The user's first pair (60457274+65906101) **failed to triangulate** — 60457274 is the
  poor view (excluded 84 % in 4-cam). The **reliable pair (50591643+65906101)** worked:
  **0 % camera exclusion**, **r = 0.862**, no occlusion dips, full 14 s.
- But a systematic **affine bias** (front views compress the sagittal arm): raw MAE 45°.

### Markerless-only fix (NO Vicon) — `code/markerless_only.py`
- **Angle:** anatomical normalization (map the curl range to extension 15° → flexion
  145°) → **MAE 17.4°, r 0.862** — almost as good as the Vicon-affine (16.1°), **no GT**.
- **Scaling:** from the markerless **stature** (Winter anthropometry) → upper arm exact
  (×1.015 ≈ Vicon ×1.009), forearm 12 % short (`arm26_S3_markerless_scaled.osim`).
  The forearm cannot be recovered better here (front-view depth collapse).

---

## MODEL SCALING — where it is now (two versions)

The model is scaled so its bone lengths / muscle moment arms / masses match the subject.
Two scaled models exist:

| Model | Source | humerus | forearm | masses (hum / forearm) |
|---|---|--:|--:|---|
| `build/opensim/arm26_S3_scaled.osim` | **Vicon** | ×1.009 | ×1.040 | 1.915 / 1.726 |
| `build_final/arm26_S3_markerless_scaled.osim` ⭐ | **markerless (stature)** | ×1.015 | ×0.910 | 1.948 / 1.157 |

**Comparison (important):**
- **Upper arm:** ×1.015 ≈ ×1.009 → the markerless stature estimate **matches Vicon almost exactly** ✅.
- **Forearm:** markerless ×0.910 vs Vicon ×1.040 → markerless is **~12 % short** (the Winter
  anthropometric ratio under-estimates this subject's forearm) ⚠️. So the markerless forearm
  is lighter (1.157 vs 1.726 kg) → its torque is a little lower than the Vicon-scaled model.

**Current state:** the pipelines (`build/`, `build_final/`) **use the VICON-scaled model**.
The markerless-scaled model is built but **not yet wired into the pipeline**. To make the
pipeline **100 % markerless** (zero Vicon inside), swap to:
- angle = anatomical de-bias (already have, no Vicon), and
- model = `arm26_S3_markerless_scaled.osim` (stature-based),
then ID/SO/3CC → a fully markerless dataset (Vicon kept only for validation).
Caveat: forearm ~12 % short → elbow torque ~10 % lower than the Vicon-scaled result.

---

## 2-CAM vs 4-CAM COMPARISON (full 14 s, identical processing)
Script: `code/run_4cam_full.py` (Pose2Sim 4-cam full) + `code/compare_2v4.py`

Both setups go through the **exact same chain** (raw TRC elbow → de-spike → affine
calibration vs Vicon → 2 Hz → arm26 Vicon-scaled → ID/SO/3CC), so the only difference
is the triangulation source. Outputs in `S3/compare_2v4/` (+ `compare_2v4.png`).

| Setup | r vs Vicon | MAE | ROM | peak torque | reps detected | 3CC fails |
|---|--:|--:|--:|--:|--:|--:|
| **2-cam** (front pair 50591643+65906101) | **0.862** | **16.1°** | 22–133° | 14.9 N·m | **5** ✓ | 0 |
| **4-cam** (all four) | 0.770 | 18.7° | 22–139° | 13.3 N·m | 10 ⚠️ noisy | 0 |

**The 2-camera setup BEATS the 4-camera one.** Even with 4 cameras, only the 2 front
cameras see the arm (cam 60457274 excluded 87 %, cam 58860488 74 %); when the two back
cameras are intermittently included they **inject noise** → double-bumps at the flexion
peaks → lower correlation (0.770) and over-counted reps (10 vs the real 5). The two clean
front cameras alone give a smoother, more accurate curl.

> **Take-away: more cameras ≠ better when the extra cameras don't see the target limb.**
> Camera **viewpoint > camera count** for a planar elbow curl — confirmed numerically.

---

## RESULTS SUMMARY

| Quantity | Value |
|---|--:|
| Triangulation reprojection (4-cam) | 22.8 mm (5.4 px) |
| Elbow angle vs Vicon (4-cam window) | r 0.825, MAE 15° |
| Elbow angle vs Vicon (2-cam full, calibrated) | **r 0.862, MAE 16°** |
| Elbow angle vs Vicon (markerless-only, no GT) | r 0.862, MAE 17° |
| Model scaling — Vicon | humerus ×1.009, forearm ×1.040 |
| Model scaling — markerless (stature) | humerus ×1.015, forearm ×0.910 |
| Peak elbow torque (final 14 s) | 14.9 N·m |
| SO / 3CC failures (final) | 0 |
| Dataset (final) | 1411 rows, 5 features → 19 labels |

---

## OUTPUTS (the `.mot` and `.osim`)

| File | What |
|---|---|
| `build/pose2sim/calibration/Calib.toml` | converted calibration (used directly, no `calibration()`) |
| `build/pose2sim/pose-3d/*.trc` | triangulated 3D markers |
| `build/pose2sim/kinematics/*_LSTM.mot` + `.osim` | Pose2Sim full-body 3D angles + scaled model |
| `build/opensim/arm26_S3_scaled.osim` | **arm26 scaled (Vicon)** ⭐ |
| `build_final/arm26_S3_markerless_scaled.osim` | arm26 scaled (markerless stature) |
| `build_final/motion/curl_14s.mot` | **final 3D joint angle motion (full 14 s)** ⭐ |
| `build_final/dataset/S3_final_dataset.csv` | features → labels dataset |
| `build_final/opensim/` | ID / SO / 3CC labels |

---

## HONEST LIMITATIONS
- Markerless **angle** is good (r 0.86); markerless **segment lengths** are not (front-view
  depth collapse) → scaling needs Vicon, a known height, or a **side-view camera**.
- The angle de-bias needs either Vicon (affine) or an anatomical prior (markerless-only).
- Forearm length markerless ≈ 12 % short → torque ≈ 10 % low vs Vicon-scaled.
- Load assumed 2 kg (Fit3D records none). Torque/force/fatigue are physics-computed,
  not validated against EMG/force plates (future work).
- **Viewpoint > camera count** for a planar curl: a single well-placed side view (2D)
  gives the cleanest elbow angle; front/back multi-camera adds occlusion or bias.
