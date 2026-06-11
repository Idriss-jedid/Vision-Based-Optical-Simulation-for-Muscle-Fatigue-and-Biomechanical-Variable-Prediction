# Existing Solutions & Our Solution (deux pipelines)

Doc lل presentation : état de l'art (existing solutions + limites) → notre solution
(deux pipelines complémentaires, sur **caméra réelle**) → pourquoi elle est bonne.

---

## 1. Le problème

Estimer les **variables biomécaniques internes** (activation musculaire, force, **fatigue**)
de manière **non-contact**, à partir d'une **caméra**. Application santé : prévenir les
troubles musculo-squelettiques (TMS), surveiller la charge de travail, suivi de
rééducation — sans capteurs intrusifs.

Difficulté clé : ces variables internes **ne sont pas observables directement** par une
caméra, et il y a **rarement de ground-truth** dans des conditions réelles.

---

## 2. Solutions existantes (état de l'art)

Dans la littérature (et dans l'introduction de notre article), l'estimation des variables
biomécaniques et de la fatigue musculaire repose principalement sur **deux grandes catégories** :

| Existing solution | Description | Limites |
|---|---|---|
| **Wearable Sensors** | Capteurs portés sur le corps (EMG, IMU, Xsens, ...) pour mesurer l'activité musculaire ou le mouvement | **Intrusifs** ; nécessitent d'**équiper le sujet** avec des capteurs/électrodes |
| **Laboratory-based Motion Capture Systems** | Systèmes de capture de mouvement en laboratoire (Vicon, Qualisys, OptiTrack, ...) | **Coûteux** ; nécessitent un **environnement contrôlé** ; **peu pratiques en conditions réelles** |

**Le constat :** ces deux familles sont précises, mais **intrusives, coûteuses et confinées au
laboratoire** — donc peu adaptées à un usage réel, sur le terrain ou à domicile.

### 2.1 Notre changement de direction : passer à la vision

Pour dépasser ces limites, **nous passons à une approche basée sur la vision** : estimer les
mêmes variables biomécaniques (et la **fatigue musculaire**) à partir d'une **simple caméra,
sans aucun contact**.

Deux questions ouvertes restent, que notre travail adresse :
- **Pas de ground-truth** des variables musculaires en conditions réelles → on s'appuie sur
  un GT **Vicon** (Pipeline 1) et **simulation OpenSim** (Pipeline 2).
- **L'impact des conditions d'imagerie** (point de vue, bruit, nombre de caméras) est peu
  étudié → on l'analyse explicitement.

C'est l'idée centrale de notre travail — détaillée en section 3 (nos deux pipelines).

---

## 3. Notre solution = deux pipelines complémentaires (caméra réelle)

**Cœur commun :** modèle musculo-squelettique OpenSim **arm26** (7 muscles, haltère 2 kg) +
chaîne **vision → angles → ID / SO → fatigue 3CC**. Les deux pipelines partagent ce cœur ;
ils diffèrent par la **partie acquisition vision**.

### Pipeline 1 — Multi-caméras synchronisées (✅ réalisé, validé)
Doc : `s04/PIPELINE_4CAM_PRESENTATION.md`.
```
video réelle (Fit3D s04, 4 caméras synchro 50 fps)
   → RTMPose (2D) → triangulation (Calib.toml, fix −R·T) → 3D markerless (.trc)
   → arm26 (scaling Vicon) → ID (couple) → SO (activation/force) → 3CC (fatigue)
   → validation vs Vicon : r = 0.99, MAE 4–6°, 0 % caméras exclues
```
- **Avantage :** 3D précise par triangulation, **validée contre un ground-truth Vicon**.
- **Contrainte :** nécessite **plusieurs caméras calibrées et synchronisées**.
- **Rôle :** prouve la **faisabilité réelle** et fournit les labels biomécaniques fiables.

### Pipeline 2 — Caméra unique (monoculaire) — 🚧 **en cours**
Article en cours : *"Vision-Based Optical Simulation for Muscle Fatigue and Biomechanical
Variable Prediction"*.
```
video réelle (UNE seule caméra) → 2D → angles → arm26 → variables musculaires / fatigue
```
- **Idée :** atteindre la même estimation avec **une seule caméra grand public** (un téléphone)
  → encore plus accessible et non-invasif.
- **Défi :** une caméra unique n'a **pas de profondeur ni de ground-truth 3D** → on s'appuie
  sur un **cadre simulation-driven** (OpenSim + modèle de caméra **pinhole** virtuel + ML,
  OpenSim servant de ground-truth) pour générer des données, étudier l'impact du **point de
  vue et du bruit**, et entraîner la prédiction.
- **Statut :** **en cours de développement.**

---

## 4. Pourquoi notre solution est bonne

1. **Non-contact & scalable** — caméras standard, pas d'électrodes ni de marqueurs.
2. **Résout le problème du ground-truth** — Pipeline 1 valide contre **Vicon réel** (r=0.99) ;
   Pipeline 2 utilise la **simulation OpenSim comme ground-truth parfait**.
3. **Va au-delà de la cinématique** — on estime **activation musculaire + force + fatigue (3CC)**,
   là où la plupart des méthodes vision s'arrêtent aux angles.
4. **Étudie les conditions d'imagerie** (point de vue, bruit, nombre de caméras) — un manque
   de la littérature. (Ex. : on a montré que le bug de calibration, pas le nombre de caméras,
   était la vraie cause des erreurs.)
5. **Flexible / deux régimes** : multi-caméras **précis** (Pipeline 1) ou caméra unique
   **pratique** (Pipeline 2) — selon le contexte d'usage.
6. **Les deux pipelines se renforcent** : la simulation conçoit et étudie le système ; les
   données réelles multi-caméras le valident.

---

## 5. Comparatif : état de l'art vs notre solution

| Critère | Wearable Sensors (EMG/IMU) | Lab Motion Capture (Vicon/Qualisys) | **Nous (vision, P1 + P2)** |
|---|---|---|---|
| Non-contact | ❌ (capteurs sur le corps) | ❌ (marqueurs sur le corps) | ✅ |
| Coût / matériel | élevé | **très élevé** | **caméra standard** |
| Conditions réelles | limité (intrusif) | ❌ (labo seulement) | ✅ (terrain / domicile) |
| Sortie | EMG / mouvement | cinématique 3D | **cinématique + muscle + fatigue** |
| Ground-truth | — | référence | **Vicon (P1) + simulation (P2)** |
| Fatigue musculaire | partielle (EMG) | ❌ | ✅ (3CC) |

---

## 6. Résumé (une phrase)

> On propose une estimation **non-contact des variables musculaires et de la fatigue** par
> vision, autour d'un cœur OpenSim (arm26 + 3CC), via **deux pipelines complémentaires sur
> caméra réelle** : un pipeline **multi-caméras synchronisées** déjà **réalisé et validé
> contre Vicon (r=0.99)**, et un pipeline **caméra unique** **en cours**, soutenu par un cadre
> simulation-driven (OpenSim + caméra pinhole virtuelle + ML).
