<h1 align="center">Vision-Based Optical Simulation</h1>
<h3 align="center">Upper-limb muscle forces and fatigue from markerless video</h3>

<p align="center">
  <b><a href="https://idriss-jedid.github.io/Vision-Based-Optical-Simulation-for-Muscle-Fatigue-and-Biomechanical-Variable-Prediction/">Project page</a></b>
  &nbsp;·&nbsp; <a href="media/demo.mp4">Demo video</a>
  &nbsp;·&nbsp; <a href="media/linkedin_demo.mp4">Short clip</a>
  &nbsp;·&nbsp; <a href="https://fit3d.imar.ro/fit3d">Fit3D</a>
  &nbsp;·&nbsp; <a href="https://zenodo.org/records/7607736">MM-Fit</a>
</p>

<p align="center">
  <video src="https://idriss-jedid.github.io/Vision-Based-Optical-Simulation-for-Muscle-Fatigue-and-Biomechanical-Variable-Prediction/demo.mp4" controls muted width="860" poster="media/banner.gif"></video>
</p>

## Abstract

A camera records how the arm moves, not how hard its muscles work. This work estimates elbow torque
and the activation, force and fatigue of the four elbow flexors from ordinary markerless video of a
biceps curl. Two front ends recover the motion: a four-camera route built on Pose2Sim and a
single-camera route built on OpenCap Monocular, both validated against Vicon motion capture (elbow
angle error of 4.2° and 6.94°). The motion drives a subject-scaled OpenSim Arm26 model carrying a 2 kg
dumbbell; inverse dynamics, static optimisation and a three-compartment fatigue model yield 13
time-resolved signals. Learned surrogates then reproduce these signals without the solver: LightGBM
reaches R² = 0.952 with four cameras, and a BiGRU reaches R² = 0.938 from single-camera 2D keypoints
alone, about 1680 times faster than OpenSim and without a GPU. This model runs in a web and mobile
application in which a patient records an exercise at home and a clinician reviews the resulting
signals. Data come from the Fit3D and MM-Fit datasets.

**Keywords:** markerless motion capture · musculoskeletal simulation · muscle fatigue · OpenSim ·
pose estimation · surrogate modelling · rehabilitation

---

## Contents

1. [Overview](#overview)
2. [Motion capture](#1-motion-capture)
3. [Musculoskeletal model](#2-musculoskeletal-model)
4. [AI surrogate](#3-ai-surrogate)
5. [Results](#4-results)
6. [Application](#5-application)
7. [Datasets](#datasets)

---

## Overview

The pipeline is assembled from four existing building blocks. Each was adapted to an upper-limb task
and extended with new layers.

<p align="center"><img src="figures/diagrams/blocks.png" width="900" alt="The four building blocks: what was taken, changed and added"/></p>

| Block | Taken | Changed | Added |
|---|---|---|---|
| **Pose2Sim** (4 cameras) | RTMPose 2D pose, triangulation, filtering | calibration rebuilt from Fit3D (`t = −R·T`) | arm kinematics, Arm26, LightGBM surrogate |
| **OpenCap Monocular** (1 camera) | ViTPose 2D pose, WHAM 3D body | optimisation replaced by gravity alignment and a 2 Hz filter | arm kinematics, Arm26, BiGRU surrogate, application |
| **OpenSim Arm26** | 2-DOF arm model | 2 kg dumbbell; 11 muscle paths tested, 4 flexors kept | scaling to each subject |
| **3CC fatigue** | Xia & Frey-Law model | elbow rates, coupled to static optimisation | fatigue for each flexor |

---

## 1. Motion capture

**Four cameras.** Pose2Sim's pose estimation and triangulation are kept. Its calibration is replaced by
a conversion of the Fit3D camera parameters, where the translation must be `−R·T`; this single fix
raised the elbow-angle correlation from 0.86 to 0.99. Its full-body marker augmentation and inverse
kinematics are replaced by three joint centres and two analytical joint angles.

<p align="center"><img src="figures/diagrams/fork_pose2sim.png" width="860" alt="Pose2Sim: shared front end, own biomechanics"/></p>

**One camera.** OpenCap Monocular's ViTPose and WHAM stages are kept. Its two-stage optimisation,
designed for walking (feet, floor, contact), is replaced by aligning the torso with gravity and a
zero-phase 2 Hz filter; its 38 markers and 33-DOF model by three joint centres and a 2-DOF arm.

<p align="center"><img src="figures/diagrams/fork_opencap.png" width="860" alt="OpenCap Monocular: shared front end, own biomechanics"/></p>

**Visual check.** The reconstructed skeleton is laid back on each calibrated camera in Blender.

<p align="center"><a href="media/blender.mp4"><img src="figures/blender_check.jpg" width="640" alt="Blender: reconstructed skeleton over the camera view"/></a></p>

---

## 2. Musculoskeletal model

The OpenSim Arm26 model is loaded with a 2 kg dumbbell and scaled to each subject. Static optimisation
shows that the load is carried by four flexors (brachialis 41.7 %, biceps long head 39.9 %, short
head 11.5 %, brachioradialis 6.9 %), which are retained.

<p align="center"><img src="figures/diagrams/arm26.png" width="860" alt="Arm26 adapted for a loaded curl"/></p>

Inverse dynamics gives the elbow torque, static optimisation the muscle activations and forces, and
the three-compartment model the fatigue of each muscle. Fatigue grows with load and repetitions as
expected (2.7 % to 28.5 % across four conditions).

<p align="center"><img src="figures/diagrams/fatigue_3cc.png" width="860" alt="Three-compartment fatigue model"/></p>

---

## 3. AI surrogate

The simulated signals become training labels for a model that predicts them directly. The model can
start at three depths of the pipeline; the earlier it starts, the more computation it saves. The same
idea is applied to both routes.

<table>
<tr>
<th align="center">Four cameras · LightGBM</th>
<th align="center">One camera · BiGRU</th>
</tr>
<tr>
<td align="center"><img src="figures/diagrams/ai_multi.png" width="430" alt="Multi-camera pipeline and its three AI entry points"/></td>
<td align="center"><img src="figures/diagrams/ai_single.png" width="430" alt="Single-camera pipeline and its three AI entry levels"/></td>
</tr>
<tr>
<td valign="top">

| Entry point | Input | R² |
|---|---|:---:|
| A | joint angles + scaled model | **0.952** |
| B | 3D joints | 0.904 |
| C | 2D keypoints | 0.860 |

OpenSim 89 s → LightGBM 0.1 s (≈ 890×).

</td>
<td valign="top">

| Entry point | Input | R² | Time | GPU |
|---|---|:---:|:---:|:---:|
| Level 1 | joint angles | 0.984 | 458 s | yes |
| Level 2 | 3D joints | 0.956 | 455 s | yes |
| **Level 3** | **2D keypoints** | **0.938** | **195 s** | **no** |

OpenSim 26.7 s → BiGRU 15.9 ms (≈ 1680×).

</td>
</tr>
</table>

**Deployed model: single camera, Level 3.** It removes the 3D reconstruction step, cuts the
processing time by more than half and runs on a CPU, for 0.046 less R² than Level 1.

---

## 4. Results

**Validation against Vicon** (Fit3D, 8 subjects)

| | Four cameras | One camera |
|---|:---:|:---:|
| 3D joint error | 27.8 mm | 49.4 mm |
| Elbow angle error | 4.2° (r = 0.99) | 6.94° (r = 0.993) |

**AI against OpenSim** on subjects never seen in training — solid: OpenSim, dashed: prediction.

<p align="center"><img src="figures/singlecam/pred_L3.png" width="860" alt="Level-3 BiGRU against OpenSim on a held-out subject"/></p>
<p align="center"><em>Single camera, Level 3, from 2D keypoints only.</em></p>

<p align="center"><img src="figures/fig_ts_s11.png" width="860" alt="LightGBM against OpenSim on a held-out subject"/></p>
<p align="center"><em>Four cameras, LightGBM.</em></p>

**Cost.** The OpenSim solve takes 26.7 s per trial on the single-camera route and 89 s on the
four-camera route; the surrogates take 15.9 ms and 0.1 s respectively (about 1680× and 890× faster),
from model files of a few megabytes instead of a 624 MB to 1 GB solver installation.

---

## 5. Application

<p align="center"><img src="figures/app/app_flow.jpg" width="860" alt="Patient phone application and clinician dashboard"/></p>

The clinician prescribes an exercise from a web dashboard; the patient records it with a phone; the
Level-3 model returns the 13 signals; the clinician reviews the session. The peak of the torque and
activation curves marks the hardest moment of the lift.

<p align="center"><img src="figures/app/peak_effort.jpg" width="760" alt="The recorded curl beside its predicted torque and activation"/></p>

---

## Datasets

| Dataset | Content | Use |
|---|---|---|
| [Fit3D](https://fit3d.imar.ro/fit3d) | biceps curls, 8 subjects, 4 calibrated cameras at 50 fps, Vicon 3D reference | validation and training |
| [MM-Fit](https://zenodo.org/records/7607736) | 21 subjects, uncalibrated video, no reference | training only (single camera) |

The single-camera training set totals 37 recordings from 29 subjects. Dataset media are not
redistributed in this repository.

---

**Author** — Idriss Abderraouf Jedid
