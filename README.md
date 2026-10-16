<h1 align="center">Vision-Based Optical Simulation</h1>
<h3 align="center">Upper-limb muscle forces and fatigue from markerless video</h3>

<p align="center">
  <b><a href="https://idriss-jedid.github.io/Vision-Based-Optical-Simulation-for-Muscle-Fatigue-and-Biomechanical-Variable-Prediction/">Project page</a></b>
  &nbsp;·&nbsp; <a href="media/demo.mp4">Narrated demo</a>
  &nbsp;·&nbsp; <a href="https://fit3d.imar.ro/fit3d">Fit3D</a>
  &nbsp;·&nbsp; <a href="https://zenodo.org/records/7607736">MM-Fit</a>
</p>

<p align="center">
  <video src="https://idriss-jedid.github.io/Vision-Based-Optical-Simulation-for-Muscle-Fatigue-and-Biomechanical-Variable-Prediction/demo.mp4" controls width="860" poster="media/banner.gif"></video>
</p>

## Abstract

Muscle activation, force and fatigue are central to rehabilitation, yet they are usually measured
with electromyography, force sensors or marker-based motion capture: equipment that is costly,
intrusive and confined to the laboratory. An ordinary RGB camera, by contrast, is available
everywhere, but it only records how a limb moves, not how its muscles work.

This work presents a vision-based framework that estimates the internal biomechanics of the upper
limb from markerless RGB video, demonstrated on the dumbbell biceps curl. Two routes recover the
motion of the arm. The multi-camera route extends Pose2Sim: RTMPose detects 2D keypoints in four
calibrated views, which are triangulated into 3D joints. The single-camera route extends OpenCap
Monocular: ViTPose and WHAM estimate the 3D body from a single video, which is then aligned with
gravity and filtered. In both routes, the shoulder, elbow and wrist give the two joint angles of the
arm. Validated against Vicon motion capture, the elbow angle follows the reference with a correlation of
0.993 in both routes, with a mean error of 6.6° with four cameras and 6.94° with one camera.

The recovered motion drives a subject-scaled OpenSim Arm26 musculoskeletal model carrying a 2 kg
dumbbell. Inverse dynamics, static optimisation and a three-compartment fatigue model produce
thirteen time-resolved signals: the elbow torque and the activation, force and fatigue of the four
elbow flexors (biceps long and short heads, brachialis and brachioradialis).

Because this physics-based pipeline takes several minutes per recording, machine-learning surrogates
are trained to reproduce its outputs directly. With four cameras, LightGBM reaches R² = 0.952
against OpenSim. With a single camera, a bidirectional GRU predicts the thirteen signals from 2D
keypoints alone with R² = 0.938 on unseen subjects, about 1,680 times faster than the OpenSim solve
and without a GPU.

This model is integrated into a web and mobile application: the patient records an exercise at home
with a phone camera, and the clinician reviews the resulting muscle signals remotely. The framework
offers a low-cost, non-contact way to monitor muscle effort and fatigue in rehabilitation. Data are
drawn from the Fit3D and MM-Fit datasets.

**Keywords:** RGB camera · markerless motion capture · human pose estimation · 3D reconstruction · musculoskeletal modelling · OpenSim · muscle activation · muscle force · muscle fatigue · surrogate model · deep learning · tele-rehabilitation

---

## Contents

1. [Overview](#1-overview)
2. [Building the two pipelines](#2-building-the-two-pipelines)
3. [Validation against Vicon](#3-validation-against-vicon)
4. [Biomechanical outputs](#4-biomechanical-outputs)
5. [Why an AI surrogate](#5-why-an-ai-surrogate)
6. [AI surrogate: approaches, benchmark and prediction](#6-ai-surrogate-approaches-benchmark-and-prediction)
7. [Optimisation, ONNX export and deployment](#7-optimisation-onnx-export-and-deployment)
8. [Datasets](#8-datasets)

---

## 1. Overview

Two pipelines take a video of a biceps curl all the way to the muscle signals, one with **four
cameras** and one with a **single camera**. Neither was built from scratch: both start from an
existing open-source architecture, which was adapted to the upper limb and extended with new layers.

| Step | What was done |
|---|---|
| **Architectures** | took Pose2Sim (4 cameras) and OpenCap Monocular (1 camera), and changed them for an arm task |
| **Musculoskeletal model** | took OpenSim Arm26, and changed it for a loaded curl |
| **Validation** | tested each stage of both pipelines against Vicon motion capture |
| **Outputs** | extracted elbow torque and muscle activation, force and fatigue (3CC) |
| **AI** | replaced the slow pipeline with a learned surrogate: 3 input approaches, model benchmark, fine-tuning |
| **Deployment** | optimised the chosen model, exported it to ONNX, and deployed it in a web and mobile app |

<p align="center"><img src="figures/diagrams/blocks.png" width="900" alt="The four building blocks: what was taken, changed and added"/></p>

---

## 2. Building the two pipelines

### 2.1 Four cameras — from Pose2Sim

Pose2Sim's 2D pose estimation (RTMPose), triangulation and filtering are kept. Its calibration is
replaced by a conversion of the Fit3D camera parameters, where the translation must be `−R·T`; this
fix alone raised the elbow-angle correlation from 0.86 to 0.99. Its full-body marker augmentation and
inverse kinematics are replaced by three joint centres (shoulder, elbow, wrist) and two analytical
joint angles.

<p align="center"><img src="figures/diagrams/fork_pose2sim.png" width="860" alt="Pose2Sim: shared front end, own biomechanics"/></p>

### 2.2 One camera — from OpenCap Monocular

OpenCap Monocular's ViTPose and WHAM stages are kept. Its two-stage optimisation, designed for walking
(feet, floor, ground contact), is replaced by aligning the torso with gravity and a zero-phase 2 Hz
filter; its 38 markers and 33-DOF body by three joint centres and a 2-DOF arm. The shoulder centre is
taken between WHAM's two joint regressors, which err in opposite directions.

<p align="center"><img src="figures/diagrams/fork_opencap.png" width="860" alt="OpenCap Monocular: shared front end, own biomechanics"/></p>

### 2.3 Musculoskeletal model — from OpenSim Arm26

The 2-DOF Arm26 model receives a 2 kg dumbbell welded to the hand and is scaled to each subject.
Eleven muscle paths were tested; static optimisation shows that four flexors carry the load
(brachialis 41.7 %, biceps long head 39.9 %, short head 11.5 %, brachioradialis 6.9 %), so these four
are kept.

<p align="center"><img src="figures/diagrams/arm26.png" width="860" alt="Arm26 adapted for a loaded curl"/></p>

---

## 3. Validation against Vicon

Each stage of both pipelines was tested on Fit3D (8 subjects) against Vicon motion capture, with four
tests whose limits were fixed before measuring. Both pipelines pass all four.

<p align="center"><img src="figures/validation_single.png" width="860" alt="Single-camera validation: the four tests"/></p>
<p align="center"><img src="figures/validation_multi.png" width="860" alt="Multi-camera validation: the four tests"/></p>

As a visual check, the reconstructed skeleton is laid back over every camera view in Blender: an error
in calibration or triangulation would make it drift away from the person.

<p align="center"><a href="media/blender.mp4"><img src="figures/blender_check.jpg" width="760" alt="Blender: reconstructed skeleton over a camera view"/></a></p>

---

## 4. Biomechanical outputs

From the validated motion, OpenSim computes the internal biomechanics frame by frame:

| Solver | Output |
|---|---|
| Inverse dynamics | elbow torque |
| Static optimisation | activation and force of the 4 flexors |
| Three-compartment fatigue model (3CC) | fatigue of the 4 flexors |

This gives **13 time-resolved signals** per recording.

<p align="center"><img src="figures/demo/opensim_signals.jpg" width="860" alt="The 13 OpenSim signals for one biceps curl"/></p>
<p align="center"><img src="figures/diagrams/fatigue_3cc.png" width="860" alt="Three-compartment fatigue model"/></p>

---

## 5. Why an AI surrogate

The pipeline is accurate but heavy: on a single camera, one recording takes **485 s**, needs a
**GPU**, and depends on a full OpenSim installation. A learned model is therefore trained to produce
the same 13 signals directly, without running OpenSim.

<p align="center"><img src="figures/demo/time_memory.jpg" width="860" alt="What the AI saves in time and memory"/></p>

---

## 6. AI surrogate: approaches, benchmark and prediction

The surrogate can start at three depths of the pipeline — **joint angles**, **3D joints** or **2D
keypoints**. The earlier it starts, the more computation it removes. Every model is evaluated on
subjects never seen in training.

### 6.1 Four cameras — LightGBM

<p align="center"><img src="figures/diagrams/ai_multi.png" width="700" alt="Multi-camera pipeline and its three AI entry points"/></p>
<p align="center"><img src="figures/demo/multi_approaches.jpg" width="860" alt="Approaches A, B and C"/></p>
<p align="center"><img src="figures/demo/multi_benchmark.jpg" width="860" alt="Nine models compared"/></p>
<p align="center"><img src="figures/demo/multi_pred.jpg" width="860" alt="LightGBM against OpenSim on subject s11"/></p>

### 6.2 One camera — BiGRU

<p align="center"><img src="figures/diagrams/ai_single.png" width="700" alt="Single-camera pipeline and its three AI entry levels"/></p>
<p align="center"><img src="figures/demo/single_levels.jpg" width="860" alt="Which level goes into the app"/></p>
<p align="center"><img src="figures/demo/single_benchmark.jpg" width="860" alt="Eight architectures, three input levels"/></p>
<p align="center"><img src="figures/demo/single_pred.jpg" width="860" alt="Level-3 BiGRU against OpenSim on subject s10"/></p>

---

## 7. Optimisation, ONNX export and deployment

**Choosing the model.** Level 3 of the single camera is deployed: it needs only 2D keypoints, so it
takes 195 s instead of 458 s and runs without a GPU, for only −0.046 R².

**Exporting.** The BiGRU is exported to **ONNX** and runs on a CPU. The OpenSim solve of 26.7 s
becomes a 15.9 ms inference (about **1,680× faster**), and what must be installed drops from 624 MB to
11.7 MB (about **53× lighter**).

**Deploying.** The model runs in a web and mobile application:

1. The clinician prescribes an exercise on the web dashboard.
2. The patient records one set at home with a phone camera.
3. The 13 signals are computed and saved with the session.
4. The clinician reviews the session remotely.

<p align="center"><img src="figures/app/app_flow.jpg" width="860" alt="Patient phone application and clinician dashboard"/></p>

The peak of the torque and activation curves marks the hardest moment of the lift in the video.

<p align="center"><img src="figures/app/peak_effort.jpg" width="760" alt="The recorded curl beside its predicted torque and activation"/></p>

---

## 8. Datasets

| Dataset | Content | Use |
|---|---|---|
| [Fit3D](https://fit3d.imar.ro/fit3d) | biceps curls, 8 subjects, 4 calibrated cameras at 50 fps, Vicon 3D reference | pipelines, validation and training |
| [MM-Fit](https://zenodo.org/records/7607736) | biceps curls, 21 subjects, uncalibrated video, no reference | single-camera training only |

Dataset media are not redistributed in this repository.

---

**Author** — Idriss Abderraouf Jedid
