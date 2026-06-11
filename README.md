# Vision-Based Optical Simulation for Muscle Fatigue and Biomechanical Variable Prediction

Pipeline markerless : **vidéo → 3D (Pose2Sim) → modèle musculo-squelettique arm26 (OpenSim)
→ angles articulaires + biomécanique (couple, activation, fatigue) → Blender**, validé contre
le ground-truth Vicon du dataset **Fit3D** (8 sujets, exercice *dumbbell_biceps_curls*, 4 caméras).

## Deux pipelines
1. **Multi-caméras synchronisées** (réalisé, validé) — 4 caméras → triangulation → arm26.
2. **Caméra unique (monoculaire)** — *en cours* (cadre simulation-driven : OpenSim + caméra
   pinhole virtuelle + ML).

## Résultats clés (8 sujets, 4-cam, vs Vicon)
- Coude : **r = 0.99, MAE 4.2°** · Épaule : **MAE 1.2°, dir3D 5.5°** · **100 % frames valides**
- Découverte clé : la convention de calibration **`translation = −R·T`** (tvec OpenCV, pas le
  centre caméra T) débloque toute la triangulation (r passé de ~0.86 à 0.99).

## Structure du dépôt
```
Code/            scripts OpenSim (Stage 1 motion min-jerk, ID, SO, CMC, 3CC fatigue, comparaisons)
Model/           modèles arm26 (.osim) : 6 → 11 muscles (dumbbell) → 7 (sans muscles d'épaule)
Data/            motions générées (.mot), datasets (.csv), docs (HOW_IT_WORKS, théorie)
S3/, s04/        pipeline vision Fit3D (code + docs PIPELINE/FIT3D/RESULTS)
batch/           métriques 8 sujets (metrics_all, results_arm26_all, RESULTS_ALL_SUBJECTS.md)
Docs/            documentation additionnelle
```

## Méthodes (références)
- **Minimum-jerk** : Flash & Hogan (1985), J Neurosci — `q = q0+(qf−q0)(10s³−15s⁴+6s⁵)`.
- **Fatigue 3CC** : Xia & Frey-Law (2008) ; rates coude Frey-Law et al. (2012) — `dMF/dt = F·MA − R·MF`.
- **Pose 2D** : RTMPose (HALPE-26) via Pose2Sim ; **triangulation** multi-vues ; **SO** : Crowninshield-Brand.

## Données non incluses (voir `.gitignore`)
Les **vidéos Fit3D** (dataset tiers, licence), les keypoints 2D par frame, les meshes Geometry
et les sorties `.sto` volumineuses sont exclus du dépôt (régénérables / téléchargeables séparément).
