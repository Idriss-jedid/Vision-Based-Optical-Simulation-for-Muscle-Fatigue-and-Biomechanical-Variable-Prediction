# References — the code and results behind the paper

This folder collects, in one place, the code and the result files behind every table, figure and number of
the paper on video-based surrogate modelling of simulated upper-limb muscle activation, force and fatigue
(Arm26 with a 2 kg load, coupled three-compartment fatigue, a four-camera route adapted from Pose2Sim, a
single-camera route adapted from OpenCap Monocular, validation against Vicon, LightGBM and BiGRU surrogates,
and the web and mobile prototype).

- Every file was **copied unchanged** from the project; nothing was recomputed. The one exception is a
  docstring in `4_application/care/security.py`, where an e-mail address used as an example was replaced by
  an `example.com` address.
- [`MANIFEST.csv`](MANIFEST.csv) lists every file with the project path it was copied from, its size and its
  SHA-256 checksum.
- Paths inside the scripts are the original project paths (`Code/`, `Data/`, `Model/`, `Results/`, `batch/`,
  …), so the scripts document what was run rather than running from this folder as they are. The figure
  scripts in `5_paper_figures/` are the exception: they run from here (see below).
- The two notebooks are the executed originals, with their outputs. Most surrogate results, and the whole
  single-camera validation, are read directly from those outputs; cell numbers below count from 0.

## Layout

| Folder | Contents |
|---|---|
| [`1_model_and_fatigue/`](1_model_and_fatigue) | Arm26 model construction, the 11-path screening, moment arms, inverse dynamics, static optimisation, the coupled 3CC fatigue stage and the minimum-jerk fatigue experiment |
| [`2_four_camera_route/`](2_four_camera_route) | Pose2Sim route: calibration convention, batch processing, labels, validation, the four-camera notebook (approaches A, B, C), LightGBM models, timings |
| [`3_single_camera_route/`](3_single_camera_route) | OpenCap Monocular route: the single-camera notebook (validation, MM-Fit augmentation, Levels 1–3, BiGRU), code listings, result tables |
| [`4_application/`](4_application) | The web and mobile prototype: analysis service, care service, patient app, deployed BiGRU (PyTorch and ONNX) |
| [`5_paper_figures/`](5_paper_figures) | The scripts that draw the paper figures, their inputs, and the figures as used in the paper |

## Where each result comes from

### Musculoskeletal model and fatigue stage

| Result in the paper | Code | Result files |
|---|---|---|
| Model construction: Arm26 → 2 kg load and 11 candidate paths → corrected brachioradialis → retained flexors (model figure, panels a–c) | `code/build_arm26_paper.py` (adds the four shoulder paths, the brachioradialis and two coordinate actuators), `code/build_arm26_paper_loaded.py` (welds the 2 kg body), `code/test_load_2kg.py` | `models/arm26_base.osim`, `models/arm26_paper_loaded.osim` (11 paths), `models/arm26_paper_loaded_brd.osim` (corrected brachioradialis), `models/arm26_paper_loaded_brd_elbow_research.osim` (model used for the labels), `models/HOW_IT_WORKS.md`, `models/Stage2_Model_Build.tex`, `renders/` |
| Static-optimisation screening of the eleven paths (model figure, panel d): shoulder-only paths at zero, triceps about 1 % each, flexors BRA 41.7 %, BIClong 39.9 %, BICshort 11.5 %, BRD 6.9 % | `code/so_on_first.py`, `code/run_cmc_subset.py` | `results/screening_11_paths/` (OpenSim static optimisation with all eleven paths), `notes/Stage_2_CMC_CrossCheck.md` (flexor force shares) |
| Brachioradialis moment arm corrected to a peak of about 7.51 cm | `code/validate_brd_moment_arm.py`, `code/validate_all_muscles.py`, `code/validate_model_opensim.py` | `notes/Stage_2_Model_BRD_Validation.md`, `notes/Stage_2_Experimental_Validation.md` (moment arms of all seven elbow muscles) |
| Inverse dynamics, static optimisation and the coupled three-compartment stage (F = 0.00912 s⁻¹, R = 0.00094 s⁻¹, re-solved load sharing) | `code/run_id_so.py`, `code/run_fatigue_so.py`, `code/run_stage2_pipeline.py` | `notes/Stage_2_ID_SO_Results.md`, `notes/Stage_2_Fatigue_3CC.md` |
| Minimum-jerk biceps curl used for the fatigue experiment (q₀ = 20°, q_f = 120° falling to 100°, shoulder at 20°, 100 Hz, slower movement and tremor over the set) | `code/generate_minjerk_motion.py` (10 repetitions), `code/generate_stage1_minjerk_fatigue.py` (30 repetitions) | `motions/paper_minjerk_fatigue_10cycles.mot`, `motions/paper_minjerk_fatigue_30cycles.mot` (+ per-frame tables, per-repetition parameters and metadata), `notes/Stage_1_Minimum_Jerk_Fatigue_Generation.md` |
| Fatigue sensitivity experiment (fatigue figure, panel b): end-of-bout fatigued fraction of BIClong 2.7 % (0 kg, 10 rep.), 5.8 % (2 kg, 10), 17.3 % (2 kg, 30), 28.5 % (4 kg, 30) | `code/run_stage2_pipeline.py` (the four conditions, with the 0 kg and 4 kg model variants in `models/`) | `results/fatigue_sensitivity/<condition>/stage2_labels.csv` and `stage2_fatigue_labels.csv`, summary table in `notes/Stage_2_Biomechanical_Analysis.md` |

### Four-camera route (Pose2Sim)

| Result in the paper | Code | Result files |
|---|---|---|
| Calibration convention t = −R·T; the correction raised the elbow-angle correlation from 0.86 to 0.99 | `code/fix_calib_tri.py`, `code/batch_calib_all.py` | `results/camera_convention/before_fix_S3_PIPELINE_DETAILED.md` (r = 0.862 before the correction), `results/camera_convention/after_fix_PIPELINE_4CAM_PRESENTATION.md` (r = 0.99 after) |
| Processing of the eight Fit3D subjects (RTMPose, triangulation, 2 Hz filter, angles, scaling, labels) | `code/run_subject.py` (end-to-end driver), `code/batch_all_subjects.py`, `code/batch_arm26_all.py`, `code/scale_from_trc.py`, `code/extract_labels_all.py`, `config/Config.toml` (Pose2Sim configuration) | `data/ml_dataset_A.csv` (12,640 frames: angles, anthropometry and the thirteen simulated signals of the eight subjects) |
| Validation against Vicon, four tests (summary and per-subject tables, per-subject figure) | Test 1 (2D reprojection) and Test 4 (scale factors): notebook cells 9–10 and 16; Tests 2–3 (segment direction, elbow correlation, elbow and shoulder MAE): `code/final_metrics.py`; corpus 3D RMSE: `code/batch_arm26_all.py` | `results/final_metrics.csv` (Tests 2–3 per subject), `results/RESULTS_ALL_SUBJECTS.md` (3D RMSE, 27.8 mm), `results/report_appendix_tables.tex` (all four tests per subject, as tables) |
| Visual check in Blender (Blender figure) | `code/blender_check/` | renders in `5_paper_figures/inputs/report (1)/Figures/nb/` |
| Benchmark of ten models, raw and processed inputs (approaches A, B, C) | notebook cells 29, 32–33 (A); 59, 62, 65 (B); 85, 88 (C) | `results/report_appendix_tables.tex` (complete benchmarks) |
| Optuna studies of the four best models, 40 trials each | notebook cells 36–37 (A), 68–69 (B), 91–92 (C) | `results/report_appendix_tables.tex` |
| Final leave-one-subject-out scores (main surrogate table, A 0.952, B 0.893, C 0.847), per target and per subject | notebook cells 40–41 (A), 71–72 (B), 94–95 (C); configurations in cells 25, 57, 83 | `models/approach_A/`, `models/approach_B/`, `models/approach_C/` (LightGBM checkpoints with the configurations of the paper) |
| Held-out subject s11 (four-camera held-out figure) | notebook cells 42 and 45; `5_paper_figures/fig_heldout_multi.py` | `5_paper_figures/output/fig_heldout_multi.*` |
| Neural sequence models on the approach-A features (fatigue 0.786–0.845 against 0.936) | notebook cells 104 and 106 | — |
| Stage times of the classical chain (232.7 s: calibration file 1.0 s, RTMPose 103.0 s, triangulation and filter 40.0 s, angles and scaling 0.09 s, OpenSim 88.7 s of which static optimisation 84.4 s) | `code/timing_compare.py` | `results/timing_pipeline.csv` |
| LightGBM inference (0.36 s, 0.10 s, 0.17 s for A, B, C) and model size (21 MB); end-to-end times 144.5, 144.1 and 104.2 s (the stages each approach still runs, plus its inference) | — | `results/results_3_approaches.csv`, with the stage times of `results/timing_pipeline.csv` |

Notes. The four-camera notebook was executed with Python 3.9.25, NumPy 1.26.4 and LightGBM 4.6.0. It
recomputes Tests 2 and 3 live with its own alignment code, so the values it prints there differ from the batch
values used in the paper; the paper's values for these two tests are the ones in `results/final_metrics.csv`. The
model cards of approaches B and C, and the R² columns of `results/results_3_approaches.csv`, come from an
earlier evaluation (B 0.904, C 0.860); the paper's 0.893 and 0.847 are the notebook's final runs (cells 71
and 94). Approach A can be retrained from `data/ml_dataset_A.csv`; approaches B and C need the triangulated
3D joints and the 2D keypoints of the Fit3D videos, which are not included.

### Single-camera route (OpenCap Monocular)

All results below are outputs of `notebook/arm26_pipeline_report.ipynb`, executed on a laptop with an NVIDIA
RTX 2000 Ada Generation GPU (Python 3.9.25, NumPy 1.26.4, PyTorch 2.1.2 with CUDA 11.8, OpenSim 4.5).
The notebook imports helper modules from that workstation; the functions it prints are in `code_listings/`.

| Result in the paper | Notebook cells | Code listing |
|---|---|---|
| WHAM output, gravity alignment, 2 Hz filter, shoulder from the mean of the COCO and Human3.6M regressors, joint angles | 9–14 | `arm26_from_video_excerpt.py` (`load_from_wham`, `gravity_align`, `angles`) |
| Subject scaling; inverse dynamics and static optimisation; reserve actuator peak 0.031 | 15–23 (reserve: cell 22) | `arm26_from_video_excerpt.py` (`scale_arm26`, `run_id`, `run_so`) |
| Validation against Vicon, four tests, both cameras of each subject (summary and per-subject tables, per-subject figure) | 25–33 | `four_tests_excerpt.py` (`r_mae`, `umeyama`, `run_one`) |
| MM-Fit augmentation: labels of the 21 sessions (MM-Fit table) | 49–54 (table: cell 53) | `mmfit_run_excerpt.py` |
| Corpus of 37 recordings, 25,615 frames, 29 subject groups; windows, channels and grouped folds | 56–59 | `ts_harness_excerpt.py` (`load_level`) |
| Input representation with a fixed encoder (single-camera representation table) | 63 (L1), 89 (L2), 106 (L3) | — |
| Benchmark at default hyper-parameters (eighteen architectures) | 65 (L1), 90 (L2), 107 (L3) | — |
| Per-architecture Optuna studies, 8 × 40 trials (benchmark table, tuned columns; main surrogate table, 0.984, 0.956, 0.938) | 72–76 (L1), 94–98 (L2), 111–115 (L3); group scores in 136 | — |
| Joint 45-trial search, configurations and five-fold re-validation (configuration table) | 66–70 and 77–82 (L1), 91–93 and 99–101 (L2), 108–110 and 116–118 (L3) | `ts_hpo_excerpt.py` (`objective`) |
| Joint subset, 3 against 17 joints | 88 (L2), 105 (L3) | — |
| Held-out subject s10 at Level 3 (single-camera held-out figure) | 119; redrawn by `5_paper_figures/fig_heldout_single.py` | — |
| Model size (BiGRU checkpoint 11.7 MB) and speed | 130 | — |
| Classical OpenSim solve 26.7 s against a 15.9 ms BiGRU inference; stage times ViTPose 195 s, WHAM 260 s, alignment 3 s, angles and scaling 0.2 s (total 485 s; Levels 1–3: 458, 455 and 195 s) | 140–144 | — |

`results/report_appendix_tables.tex` gathers the complete single-camera tables (four tests per subject, all
eighteen architectures, the 8 × 40 studies, the top trials of each joint search and the retained trials). The
WHAM working-memory figure follows from the SMPL mesh: 6,890 vertices × 3 coordinates × 4 bytes = 82.7 kB per
processed frame (about 48 MB for a 35 s trial at stride 3). The 624 MB is the size of the OpenSim 4.5
installation on disk.

### Application

| Result in the paper | Files |
|---|---|
| Two-service prototype: analysis service (2D arm keypoints, 22 input channels, thirteen predicted signals) and care service (accounts, prescriptions, sessions, patient and clinician roles, ownership checks) | `4_application/analysis/`, `4_application/care/`, `4_application/patient_app/`, `4_application/tests/`, `4_application/README.md` |
| Deployed Level-3 BiGRU: 11.7 MB as a PyTorch checkpoint, 11.3 MB in ONNX | `4_application/analysis/weights/single_camera_bigru.pt`, `4_application/patient_app/models/l3.onnx` and `l3_meta.json` |
| Screenshots (application figure) | `5_paper_figures/inputs/figures/app/` |

The two third-party ONNX models of the deployed stack (YOLOX-tiny detection, 20.3 MB; RTMPose-s 2D pose,
21.9 MB) are not included; `patient_app/` expects them next to `l3.onnx` as `yolox-tiny.onnx` and
`rtmpose-s.onnx`. The demo database, the sample videos (Fit3D clips) and the sample calibration file are not
included either.

## Paper figures

The scripts draw the figures from the inputs in this folder:

```bash
cd references/5_paper_figures
python make_paper_figures.py inputs inputs/osim_s04.json out   # model, fatigue, signals, validation, Blender, cost, application
python fig_heldout_multi.py inputs out                          # four-camera held-out subject s11 (refits the fold)
python fig_heldout_single.py <L3_pred_example.npz> out          # single-camera held-out subject s10
```

| Figure | Drawn by | Data |
|---|---|---|
| Overview and pipeline adaptation diagrams | TikZ, in the manuscript source | — |
| Construction of the task-specific model | `make_paper_figures.py` (`fig_arm26`) | `inputs/demo/assets/` (model renders); screening shares from `1_model_and_fatigue` |
| Three-compartment stage and its response to load and repetitions | `make_paper_figures.py` (`fig_3cc`) | `1_model_and_fatigue/results/fatigue_sensitivity/` |
| The thirteen simulated signals of subject s04 | `make_paper_figures.py` (`fig_signals`) | `inputs/osim_s04.json` (the s04 OpenSim signals of the project demo data) |
| Per-subject validation of both routes | `make_paper_figures.py` (`fig_validation`, `fig_validation_short` for the short version) | the per-subject tables of both routes |
| Blender visual check | `make_paper_figures.py` (`fig_blender`) | `inputs/report (1)/Figures/nb/` |
| Held-out subjects s11 (four cameras) and s10 (one camera) | `fig_heldout_multi.py`, `fig_heldout_single.py` | `inputs/dataset/ml_dataset_A.csv`; `L3_pred_example.npz` saved by the single-camera notebook (cell 119), kept on the GPU workstation |
| Classical against AI solution, time and agreement | `make_paper_figures.py` (`fig_cost`) | the timing tables above |
| Web and mobile prototype | `make_paper_figures.py` (`fig_app`) | `inputs/figures/app/` |

`output/` holds the figures as they appear in the paper (PDF and high-resolution PNG). The values drawn by
`make_paper_figures.py` are written in the script, copied from the result files listed above.

## Not included

- The manuscript itself.
- The Fit3D and MM-Fit datasets (videos, Vicon joints, camera parameters). They are third-party resources,
  available from <https://fit3d.imar.ro/> and <https://mmfit.github.io/> under their providers' terms, and
  are not redistributed here.
- The licensed SMPL body-model files used by WHAM.
- The intermediate per-frame files of the four-camera runs (2D keypoints, triangulated trajectories) and the
  helper modules and `ml_data/` folder of the single-camera workstation; the notebooks keep their outputs.
