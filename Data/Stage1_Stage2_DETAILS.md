# Stage 1 + Stage 2 — Détails complets (motion FIRST + muscles 11→7 + fatigue 3CC)

Motion étudiée : **`paper_minjerk_fatigue_10cycle_first.mot`** (version FIRST).
Model : `arm26_paper_loaded*.osim`. Tous les chiffres ci-dessous sont mesurés sur la
motion FIRST (ID/SO réel, biomech env).

---

## 1. Comment la motion FIRST a été générée (Stage 1)

Générateur : `generate_minjerk_motion.py`. On ne filme personne — l'angle du coude est
généré par une **loi de minimum-jerk** (Flash & Hogan 1985).

### 1.1 Loi minimum-jerk (polynôme degré 5)
```
q(τ) = q0 + (qf − q0)·(10·s³ − 15·s⁴ + 6·s⁵) ,   s = τ / T
```
Propriétés : vitesse = 0 et accélération = 0 aux extrémités, **profil de vitesse en
cloche** (proche du mouvement humain réel).

### 1.2 Structure d'une répétition
| Phase | Détail | Durée |
|---|---|---|
| flexion (concentrique) | 20° → 120° | 1.5 s |
| pause haute | tient 120° | 0.2 s |
| extension (excentrique) | 120° → 20° | 1.5 s |
| pause basse | tient 20° | 0.2 s |
Échantillonnage **100 Hz**, **10 reps**, épaule fixée **20°**, seed = 42.

### 1.3 Fatigue (version FIRST = brouillon)
- **Courbe de fatigue : LINÉAIRE** `f = rep/(N−1)` (0 à rep 1, 1 à rep 10).
- Effet ROM : la cible descend **linéairement 120° → 100°** :

| Rep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|--|--|--|--|--|--|--|--|--|--|
| Pic commandé (°) | 120 | 117.8 | 115.6 | 113.3 | 111.1 | 108.9 | 106.7 | 104.4 | 102.2 | 100 |

- **Bruit : 1 sinusoïde tremor 6 Hz + bruit blanc gaussien** (σ jusqu'à 0.3°), ajouté
  **seulement pendant le mouvement**.

### 1.4 Stats mesurées (FIRST)
| Métrique | Valeur |
|---|---|
| Lignes / durée | 3999 / **40.0 s** |
| Coude min / max | **18.31° / 120.00°** |
| Échantillons < 20° | 140 |
| Saut max par pas | 2.23° |
| RMS-jerk frais → fatigué | 25 637 → 489 730 (**×19.1**) |

### 1.5 Bugs de FIRST (pourquoi on a fait FINAL après)
1. **Discontinuité** à chaque fin de course (le tremor s'éteint pendant les pauses → saut).
2. **Bruit blanc non physique** → jerk ×19 (injecte du bruit dans l'inverse dynamics).
3. **CSV incohérent** (vitesse/accél = valeurs propres, pas celles du signal bruité).
→ FINAL (`paper_minjerk_fatigue_10cycles.mot`) corrige tout : fatigue saturante,
tremor band-limité 6–12 Hz, drift 0.1–0.5 Hz, jerk ×4.9, CSV+`.sto` cohérents.

---

## 2. Le modèle : 11 muscles → 7 (pourquoi)

| Modèle | # muscles | Muscles |
|---|---|---|
| `arm26.osim` (origine) | 6 | TRIlong, TRIlat, TRImed, BIClong, BICshort, BRA |
| `arm26_paper_loaded.osim` | **11** | + DELT_ant, DELT_post, PECT, LAT, **BRD_hand** |
| `arm26_paper_loaded_brd_elbow_research.osim` | **7** | on enlève les 4 muscles d'épaule |

### 2.1 Pourquoi enlever 4 muscles ?
On classe chaque muscle par son **bras de levier au coude** `r = −dL/dθ` (virtual work) :
`r > 0` = fléchisseur, `r < 0` = extenseur, `r ≈ 0` = muscle d'épaule (n'agit pas au coude).

| # | Muscle | Rôle | **Bras de levier coude (cm)** | Décision |
|---|---|---|---|---|
| 1 | BIClong | **FLÉCHISSEUR** | 1.99 → 4.94 | ✅ gardé |
| 2 | BICshort | **FLÉCHISSEUR** | 1.99 → 4.94 | ✅ gardé |
| 3 | BRA | **FLÉCHISSEUR** | 0.36 → 2.40 | ✅ gardé |
| 4 | BRD_hand | **FLÉCHISSEUR** | 1.73 → 2.46 | ✅ gardé |
| 5 | TRIlong | extenseur | −2.40 | ✅ gardé |
| 6 | TRIlat | extenseur | −2.40 | ✅ gardé |
| 7 | TRImed | extenseur | −2.40 | ✅ gardé |
| 8 | DELT_ant | épaule | **0.00** | ❌ **enlevé** |
| 9 | DELT_post | épaule | **0.00** | ❌ **enlevé** |
| 10 | LAT | épaule | **0.00** | ❌ **enlevé** |
| 11 | PECT | épaule | **0.00** | ❌ **enlevé** |

→ Les 4 muscles d'épaule (DELT×2, PECT, LAT) ont **bras de levier = 0.00 cm au coude**
(l'épaule est fixée à 20°) → activation 0 dans une flexion du coude → **enlevés**. Restent **7**.

---

## 3. Charge à vaincre + activation des muscles (ID + SO sur FIRST)

### 3.1 Couple gravitationnel de l'haltère 2 kg (la demande)
| Coude (°) | 18 | 38 | 58 | 68 | 88 | 118 |
|---|--|--|--|--|--|--|
| Couple flexion (N·m) | −4.07 | −5.35 | −5.99 | **−6.04** | −5.59 | −4.x |
Pic ≈ **6 N·m** vers 63–68° (bras horizontal). C'est ce que les fléchisseurs doivent tenir.

### 3.2 Inverse Dynamics (couple coude requis sur FIRST)
**|M| moyen = 6.85 N·m, pic = 12.90 N·m** (gravité + dynamique du mouvement).

### 3.3 Static Optimization — répartition sur les 7 muscles (motion FIRST)
SO minimise Σ activation² (Crowninshield-Brand). Résultat **mesuré** :

| Muscle | Rôle | **act moy %** | **act pic %** | force moy (N) | force pic (N) | Travaille ? |
|---|---|--:|--:|--:|--:|---|
| **BIClong** | fléch. | **17.9** | **39.7** | 104.5 | 251.5 | ✅ principal |
| **BRA** | fléch. | **11.1** | **30.1** | 105.2 | 296.0 | ✅ principal |
| **BRD_hand** | fléch. | 7.5 | 17.3 | 17.7 | 38.6 | ✅ |
| **BICshort** | fléch. | 7.2 | 19.5 | 29.4 | 77.6 | ✅ |
| TRIlong | exten. | 1.0 | 1.0 | 6.6 | 8.4 | ❌ ~0 (plancher) |
| TRIlat | exten. | 1.0 | 1.2 | 5.9 | 7.5 | ❌ ~0 |
| TRImed | exten. | 1.0 | 1.2 | 5.7 | 7.3 | ❌ ~0 |

**Lecture :** seuls les **4 fléchisseurs** travaillent ; **BIClong + BRA** portent l'essentiel
(act pic ~40% et ~30%). Les **3 triceps restent au plancher 1%** (bound min SO = 0.01) car
ce sont des extenseurs → inutiles pour lever le poids. → la curl est portée par **4 muscles**.

---

## 4. Fatigue — pourquoi, où, comment (3CC + SO)

### 4.1 Le problème (la comparaison qui a motivé la fatigue)
ID+SO seuls → **activation quasi-plate d'une rep à l'autre** (ne change pas). Raison :
- la fatigue *cinématique* de Stage-1 **réduit la demande** (plus lent, ROM plus petit → couple plus faible) ;
- SO est **sans mémoire** (chaque frame indépendant) → aucune accumulation.

Or la vraie fatigue fait l'inverse : **la capacité chute, la demande reste → l'activation MONTE**
(Potvin & Bent : aEMG ↑, MPF −25…29 %).

### 4.2 La solution : Three-Compartment Controller (Xia & Frey-Law 2008) couplé à SO
Script : `run_fatigue_so.py`. 3 compartiments par muscle : **MA** (actif %), **MR** (repos %),
**MF** (fatigué %).

**Formule (cœur de la fatigue) :**
```
dMF/dt = F·MA − R·MF
capacité(t) = 1 − MF(t)/100
```
**Taux du coude (Frey-Law 2012) :  F = 0.00912 /s   (fatigue),   R = 0.00094 /s   (récup.)**

À chaque frame on **re-résout la répartition avec la borne fatiguée** :
```
min  Σ (F_m / Cap0_m)²
s.c. Σ r_m(θ)·F_m = M(t)
     0 ≤ F_m ≤ capacité_m(t) · Cap0_m      (ne peut pas dépasser le max fatigué)
```
`Cap0_m = F0_m / a0_m` (capacité fraîche issue de SO). La fraction active
`MA = 100·(F_m/Cap0_m)` alimente le `dMF` suivant → **boucle positive** : plus de fatigue
→ moins de capacité → plus d'activation → plus de fatigue. Si Σ capacité < demande →
**échec de tâche** (saturation).

### 4.3 Résultat
L'activation **monte d'une rep à l'autre**, **MF augmente**, **capacité diminue** —
signature de fatigue conforme à Potvin/Bent.

---

## 5. Comparaison : reps ↑ / masse ↑ → qu'est-ce qui augmente ?

| On augmente | Ce qui MONTE | Ce qui BAISSE |
|---|---|---|
| **Répétitions** | **MF (fibres fatiguées %)**, **activation fatiguée** (accumulation 3CC) | **capacité** ; (l'activation SO *fraîche* reste plate = le problème de départ) |
| **Masse** (`paper_scenarios.csv` : 0 → 2 kg) | **demande (couple coude)** → activation de base ↑ → fatigue plus rapide | endurance / temps avant **échec de tâche** (plus tôt) |

→ Résumé : avec reps/masse, **fatigue (MF) + activation augmentent, capacité diminue**.
Plus de masse = plus de demande = fatigue plus rapide = échec plus tôt.

---

## 6. CMC — l'étape de validation (là où on a hésité)

SO **ignore la dynamique d'activation/contraction** (instantané). Pour **valider** les
activations SO, on a essayé **CMC (Computed Muscle Control, Thelen/Anderson/Delp 2003)** :
suivi PD + optimiseur sur un modèle en dynamique directe, qui **respecte la dynamique**.

- Script : `run_cmc_subset.py`. CMC est **beaucoup plus lent** → lancé sur une **fenêtre courte**
  (~2 premières reps, t = 0.1–8.6 s).
- Attendu (Roelker 2020) : activations CMC du **même ordre** que SO mais **plus lisses /
  déphasées** (délai électromécanique), parfois un peu plus hautes (co-contraction).
  L'accord **valide les labels SO**.

→ CMC = **contre-vérification** de SO (pas une nouvelle étape), faite sur un sous-ensemble.

---

## 7. Récap du pipeline (Stage 1 → Stage 2)
```
min-jerk + fatigue cinématique           → motion .mot (FIRST = brouillon, FINAL = retenu)
   → arm26 : 11 muscles → 7 (enlève 4 épaule, bras de levier coude = 0)
   → ID  : couple coude requis (FIRST: |M| moy 6.85, pic 12.90 N·m)
   → SO  : activation/force (4 fléchisseurs travaillent ; BIClong+BRA dominent)
   → [comparaison : activation plate sur les reps]
   → 3CC + SO : dMF/dt = F·MA − R·MF (F=0.00912, R=0.00094 /s) → activation ↑, capacité ↓
   → CMC (sous-ensemble) : validation des activations SO
```
