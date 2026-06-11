# Résultats 4-caméras — tous les sujets Fit3D (s03 → s11)

Pipeline identique à s04 (calib **−R·T**, RTMPose lightweight, triangulation 4-cam,
arm26 **2 DOF data-driven** : épaule + coude), exercice **dumbbell_biceps_curls**.
Sources : `batch/metrics_all.csv`, `batch/results_arm26_all.csv`.

---

## 1. Précision angulaire vs Vicon (8 sujets)

| Sujet | %valid | **r coude** | MAE coude | **r épaule** | std épaule | MAE épaule | dir3D | combiné | RMSE 3D |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| s03 | 100% | 0.991 | 4.5° | 0.847 | 2.5° | 1.1° | 5.5° | 2.8° | 24.2 mm |
| s04 | 100% | 0.990 | 4.6° | **0.988** | 8.9° | 1.0° | 4.3° | 2.8° | 28.1 mm |
| s05 | 100% | 0.993 | 4.5° | 0.884 | 3.0° | 1.1° | 6.5° | 2.8° | 29.6 mm |
| s07 | 100% | **0.996** | 3.7° | 0.965 | 4.2° | 0.9° | 5.9° | 2.3° | 27.3 mm |
| s08 | 100% | 0.992 | 4.0° | **0.321** | **1.8°** | 1.4° | 7.1° | 2.7° | 36.9 mm |
| s09 | 100% | 0.991 | 5.3° | 0.870 | 5.0° | 2.0° | 5.5° | 3.7° | 30.7 mm |
| s10 | 100% | 0.995 | 3.8° | 0.973 | 5.9° | 1.0° | 5.7° | 2.4° | 24.9 mm |
| s11 | 100% | 0.994 | **3.4°** | 0.930 | 2.8° | 0.8° | 3.5° | 2.1° | 20.8 mm |
| **Moyenne** | **100%** | **0.993** | **4.2°** | 0.85 | 4.1° | **1.2°** | **5.5°** | **2.7°** | **27.8 mm** |

**Métriques :**
- **r coude / MAE coude** = corrélation et erreur absolue de l'angle du coude vs Vicon.
- **r épaule** = corrélation de l'élévation d'épaule vs Vicon ; **std épaule** = amplitude réelle
  du mouvement d'épaule (Vicon). **r épaule suit directement std épaule** (voir note).
- **MAE épaule** = erreur absolue de l'élévation d'épaule ; **dir3D** = erreur de direction 3D
  du bras (robuste, valable quel que soit le ROM).
- **combiné** = (MAE coude + MAE épaule)/2 ; **RMSE 3D** = erreur 3D RMS des 3 marqueurs (Procrustes).

> **Lecture du r épaule (et le « 0.32 » de s08) :** le coude bouge beaucoup (20→160°) → **r = 0.99
> partout**. L'épaule est **stabilisée** pendant la curl ; **r épaule suit l'amplitude (std)** :
> std élevé → r élevé (s04 : std 8.9° → r 0.99 ; s10 : 5.9° → 0.97), std faible → r faible
> (**s08 : std 1.8° → r 0.32** ; s03 : 2.5° → 0.85). Ce n'est **pas une erreur** : quand
> l'articulation est quasi-immobile, le r de Pearson devient non significatif (variance ≈ bruit).
> L'exactitude réelle est donnée par la **MAE (0.8–2.0°)** et **dir3D (3.5–7.1°)**, bonnes pour
> **tous** (s08 : MAE 1.4°, dir3D 7.1°). Calibration vérifiée OK pour tous (cf. §5).

---

## 2. Mise à l'échelle (scaling) par sujet — pas seulement s04

Facteurs d'échelle arm26 (Vicon) = longueur segment Vicon / longueur modèle par défaut
(humérus 0.291 m, avant-bras 0.254 m). **Ils varient selon la morphologie du sujet.**

| Sujet | Humérus Vicon (m) | ×humérus | Avant-bras Vicon (m) | ×avant-bras |
|---|--:|--:|--:|--:|
| s03 | 0.293 | ×1.009 | 0.264 | ×1.040 |
| s04 | 0.292 | ×1.006 | 0.256 | ×1.007 |
| s05 | 0.276 | ×0.950 | 0.254 | ×0.999 |
| s07 | 0.259 | ×0.891 | 0.246 | ×0.968 |
| **s08** | 0.248 | **×0.853** (le plus petit) | 0.241 | ×0.949 |
| s09 | 0.262 | ×0.901 | 0.247 | ×0.973 |
| **s10** | 0.303 | **×1.044** (le plus grand) | 0.266 | ×1.048 |
| s11 | 0.290 | ×0.998 | 0.249 | ×0.981 |

→ La morphologie change (humérus ×0.85 à ×1.04) ; le scaling **s'adapte par sujet**
(s08 = bras le plus court, s10 = le plus long).

---

## 3. Conclusion

1. **Le fix de calibration (−R·T) généralise aux 8 sujets** : coude **r = 0.99** partout,
   **100 % de frames valides**, 0 échec. Ce n'était pas un coup de chance sur s04.
2. **Précision globale** : coude MAE ≈ **4°**, épaule MAE ≈ **1°**, métrique combinée ≈ **2.7°**,
   erreur de pose 3D ≈ **28 mm** — qualité Vicon en markerless 4-caméras.
3. **Scaling par sujet** : le modèle arm26 est mis à l'échelle individuellement (Vicon).
4. **Sorties Blender par sujet** dans `D:\p2s_blender\<subj>_arm26_4cam\` (model.osim,
   motion_world.csv, arm_markers_world.trc, Calib_world.toml, videos, Geometry).

---

## 4. Comparaison des modèles de pose — VIDÉO COMPLÈTE (s04, 4-cam, vs Vicon)

Sur tout le clip (pas une fenêtre de 6 s), 4 caméras, calib −R·T :

| Mode | Modèle | r coude | MAE coude | r épaule | MAE épaule | **temps pose** |
|---|---|--:|--:|--:|--:|--:|
| **lightweight** | RTMPose-s + YOLOX-tiny | 0.988 | 4.9° | 0.986 | 1.1° | **46 s** |
| **balanced** | RTMPose-m + YOLOX-m | **0.995** | **3.6°** | 0.991 | 0.9° | **3124 s (52 min)** |

→ balanced est un peu plus précis (MAE coude 3.6° vs 4.9°), mais **≈ 68× plus lent** (52 min
vs 46 s par sujet). **Notre objectif étant un usage temps réel, balanced (52 min) est
disqualifié.** **Choix : lightweight** — il garde **r = 0.99** au coude et **100 %** de frames
valides, tout en restant rapide (compatible temps réel). Tous les résultats des 8 sujets
ci-dessus utilisent **lightweight**.

---

## 5. Validation de la calibration (les 8 sujets) — pas de bug

Question : le faible r d'épaule de s08 vient-il d'un problème de calibration (comme s04 au
début) ? **Non.** Deux vérifications :

| Sujet | reproj triangulation (markerless, **= qualité calib**) | reproj Vicon→2D (calib + offset anatomique) |
|---|--:|--:|
| s03 | 3.5 px | 8.0 px |
| s04 | 3.7 px | 5.0 px |
| s05 | 3.6 px | 8.6 px |
| s07 | 2.9 px | 10.5 px |
| **s08** | **2.8 px (le meilleur !)** | 13.5 px |
| s09 | 3.3 px | 9.5 px |
| s10 | 4.2 px | 6.2 px |
| s11 | 3.0 px | 5.2 px |

- La **reproj de triangulation markerless (2.8–4.2 px)** est la vraie mesure de qualité de la
  calibration (indépendante de Vicon). **Tous excellents ; s08 = 2.8 px, le meilleur.** → calib OK.
  (Le bug −R·T donnait des **centaines de px** ; aucun sujet n'est dans ce cas.)
- La reproj Vicon→2D plus élevée pour s08 (13.5 px) provient de l'**écart de définition entre
  le joint Vicon et le keypoint HALPE** (anatomie/morphologie), **pas de la calibration**.
- **Conclusion :** ni s08 ni s03 n'ont de problème de calibration. Le faible r d'épaule de s08
  est dû à l'épaule **quasi-statique** (3 preuves : elbow r=0.99, triangulation 2.8 px, et
  aucun signal corrélable après recalage/lissage). Métriques correctes : MAE 1.4°, dir3D 7.1°.

---

## 6. Sorties Blender par sujet

`D:\p2s_blender\<subj>_arm26_4cam\` pour s03…s11, chacun avec :
`model.osim`, `motion_world.csv` (épaule+coude data-driven, world frame, 50 fps),
`arm_markers_world.trc`, `Calib_world.toml` (4 caméras, reproj 3.7–4.5 px), `videos\` (4),
`Geometry\` (601 .stl). Import Blender : Model → Motion (csv) → Markers → cameras → vidéos.

---

## 7. Règles, formules et seuils (quand un résultat est bon / mauvais)

### 7.1 Calibration — reprojection
- **Formule :** `reproj = ‖ proj(K, R, t, X) − x_2D ‖` (px). `proj` = projection pinhole avec
  `t = −R·T` (tvec). Erreur de triangulation = on triangule X depuis toutes les caméras puis on
  reprojette : `reproj_i = ‖ P_i·X − x_i ‖`.
- 🟢 **Bon :** triangulation < **5 px** (≈ 2 cm) ; acceptable < 25 px (< 2.5 cm, Pose2Sim).
- 🔴 **Mauvais :** > 25 px, ou **centaines de px** = bug de convention (`t = T` au lieu de `−R·T`).
- *Nos valeurs :* 2.8–4.2 px → **bon** pour tous.

### 7.2 poseEstimation — confiance des keypoints
- **Formule :** chaque keypoint a une confiance `c ∈ [0,1]` (sortie SimCC de RTMPose).
- 🟢 **Bon :** `c > 0.5` (on garde si `c > 0.3`). *Nos bras : ~0.85–0.88.*
- 🔴 **Mauvais :** `c < 0.3` → keypoint occulté/faux → exclu de la triangulation.

### 7.3 triangulation — DLT + exclusion de caméras
- **Formule :** DLT pondéré par la confiance ; une caméra est exclue si sa `reproj > seuil`
  (30 px) ; si caméras restantes < `min_cameras` (2) → frame abandonnée. **Angle de base
  (baseline)** entre 2 caméras vu du sujet : `θ = arccos( (C₁−S)·(C₂−S) / (|·||·|) )`.
- 🟢 **Bon :** % frames valides = **100 %**, 0 caméra exclue, baseline **> 25–30°** (nous : 47.7°).
- 🔴 **Mauvais :** baseline < 15° (triangulation mal conditionnée) ; « No persons triangulated ».

### 7.4 filtering — Butterworth passe-bas (zéro-phase)
- **Formule :** filtre Butterworth ordre 4, fréquence de coupure `fc`, appliqué en `filtfilt`
  (aller-retour → zéro déphasage).
- **Règle :** `fc` **au-dessus** du mouvement (curl ~0.5–1 Hz) et **en-dessous** du bruit (3–6 Hz).
  On prend **fc = 2 Hz** sur les angles.
- 🟢 **Bon :** enlève le bruit sans écraser la ROM. 🔴 **Mauvais :** `fc` trop bas → pics rabotés
  (ROM perdue) ; trop haut → bruit passe → couple d'inverse-dynamics gonflé.

### 7.5 Métriques de validation (angle / 3D vs Vicon)

| Métrique | Formule | 🟢 Bon | 🔴 Mauvais |
|---|---|---|---|
| **r (Pearson)** | `r = Σ(xᵢ−x̄)(yᵢ−ȳ) / √(Σ(xᵢ−x̄)²·Σ(yᵢ−ȳ)²)` | **> 0.90** (fort), > 0.95 excellent | < 0.7 — **mais non valide si std < ~3°** (variance ≈ bruit) |
| **MAE** | `MAE = (1/N)·Σ|xᵢ−yᵢ|` (°) | **< 5°**, < 2° excellent | > 10° |
| **RMSE** | `RMSE = √((1/N)·Σ(xᵢ−yᵢ)²)` | < 5° | > 10° |
| **dir3D** (bras) | `θ = arccos(û_ml·û_vic)` après recalage rigide | **< 8°** | > 15° |
| **RMSE 3D** | `√((1/N)·Σ‖p_ml−p_vic‖²)` après Procrustes (mm) | **< 30 mm** | > 50 mm |
| **% frames valides** | frames triangulées / total | **100 %** | < 90 % |

**Règle d'or sur le `r` :** le coefficient de corrélation n'est fiable que si le signal **bouge
assez** (std grand). Coude (bouge 20→160°) → r toujours valide (0.99). Épaule **stabilisée**
(std faible) → utiliser **MAE / dir3D** (qui restent valides à faible amplitude), pas le r.

### 7.6 Comparaison des modèles de pose (choix lightweight)
- **Critère :** précision (MAE) **vs** vitesse (temps réel exigé).
- 🟢 **lightweight :** MAE coude 4.9°, r 0.99, **temps réel** (46 s/vidéo) → **choisi**.
- 🟠 **balanced :** MAE 3.6° (meilleur) **mais 52 min/vidéo** → trop lent pour le temps réel.
