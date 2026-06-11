# Fit3D Dataset — présentation complète + utilisation (JSON → CSV/Calib)

Source : `D:\Download\fit3d_train\train`. Doc lل presentation : structure complète du
dataset، les 6 types de fichiers (b arqám)، puis **comment on n'utilise que
`dumbbell_biceps_curls`** et **comment on convertit JSON → CSV / Calib.toml**.

---

## 1. Vue d'ensemble (Fit3D, split "train")

**Fit3D** (Fieraru et al. 2021, "AIFit") = dataset 3D human pose & shape pendant des
**exercices de fitness**, multi-vues, avec ground-truth 3D précis.

| Élément | Valeur |
|---|---|
| Sujets (train) | **8** : `s03, s04, s05, s07, s08, s09, s10, s11` |
| Caméras | **4 synchronisées** : `50591643, 58860488, 60457274, 65906101` |
| Exercices | **47** par sujet (band_pull_apart, squat, deadlift, **dumbbell_biceps_curls**, ...) |
| Vidéo | **900 × 900 px, 50 fps** (durée variable, ex. s03 biceps = 709 frames) |
| Taille (par sujet) | videos ≈ 1.8 GB, smplx ≈ 618 MB, gpp ≈ 429 MB, joints3d_25 ≈ 57 MB, calib ≈ 0.8 MB |

> **3lèch Fit3D ?** video + calibration + ground-truth 3D، kol m3a ba3dhom، 3la 8 sujets ×
> 47 exercices × 4 caméras → mثáli bش **nvalidiw** pipeline markerless bla mocap.

---

## 2. Structure d'un sujet

```
s04/
├── camera_parameters/   ← calibration : 4 caméras × 47 exercices  (R, T, intrinsics)
│   ├── 50591643/{band_pull_apart.json, ..., dumbbell_biceps_curls.json, ...}
│   └── ... (4 caméras)
├── joints3d_25/         ← GROUND TRUTH 3D : 25 joints × frames  (1 .json / exercice)
│   └── dumbbell_biceps_curls.json
├── videos/              ← 4 caméras × 47 exercices  (.mp4)
│   ├── 50591643/dumbbell_biceps_curls.mp4
│   └── ... (4 caméras)
├── smplx/               ← SMPL-X body model fits  (1 .json / exercice)
├── gpp/                 ← annotations "posing" AIFit  (1 .json / exercice)
└── rep_ann.json         ← frontières des répétitions، tous les exercices
```

---

## 3. Les 6 types de fichiers (détails + arqám)

### 3.1 `camera_parameters/<cam>/<exercice>.json` — calibration
Un fichier par caméra et par exercice. Exemple (cam 65906101) :
```json
{ "extrinsics": { "R": [[...3×3...]],          ← rotation (monde → caméra)
                  "T": [[3.9024, 2.1214, 1.4836]] },   ← CENTRE caméra (m), ~4.4 m du sujet
  "intrinsics_w_distortion": {
      "f": [[1039.0, 1035.0]],   ← focale (fx, fy) px
      "c": [[476.4, 449.0]],     ← point principal (cx, cy)
      "k": [[-0.145, 0.112, 0.0034]],   ← distorsion radiale (k1,k2,k3)
      "p": [[-0.0072, 0.0062]] },        ← distorsion tangentielle (p1,p2)
  "intrinsics_wo_distortion": { "f": [...], "c": [...] } }
```
Convention IMAR : `X_caméra = R·(X_monde − T)`.

### 3.2 `joints3d_25/<exercice>.json` — el GROUND TRUTH
```json
{ "joints3d_25": [ [ [x,y,z] ×25 joints ] ×T frames ] }   →  shape (T, 25, 3), en mètres
```
- 25 joints 3D précis (mocap-grade). **Indices utilisés pour la curl : 14=RShoulder, 15=RElbow, 16=RWrist.**
- C'est le "Vicon" du dataset → comparaison de notre angle markerless (r, MAE).

### 3.3 `videos/<cam>/<exercice>.mp4`
4 vidéos synchronisées, 900×900, 50 fps → **entrée de la pose estimation** (RTMPose).

### 3.4 `smplx/<exercice>.json` — modèle de corps SMPL-X (non utilisé)
Clés : `transl, global_orient, body_pose, betas, left_hand_pose, right_hand_pose, jaw_pose,
leye_pose, reye_pose, expression`. = paramètres de forme + pose du corps complet (mesh).
**On ne l'utilise pas** (on travaille au niveau articulaire, pas le mesh).

### 3.5 `gpp/<exercice>.json` — annotations "posing" (non utilisé)
Clés : `posing_values, body_code`. = retours de forme / qualité du geste (style coach AIFit).
**Non utilisé** ici.

### 3.6 `rep_ann.json` — frontières des répétitions
```json
{ "dumbbell_biceps_curls": [115, 244, 380, 514, 652, 784], ... }   ← 6 frontières = 5 reps
```

---

## 4. Comment on l'utilise (uniquement `dumbbell_biceps_curls`)

Sur **47 exercices**, on n'en prend **qu'un seul : `dumbbell_biceps_curls`** (flexion du coude
contre 2 kg → idéal pour le modèle arm26 à 2 DOF). Sur les 6 types de fichiers, on en utilise **3** :

| Fichier Fit3D (biceps curls) | Utilisé ? | Rôle |
|---|---|---|
| `camera_parameters/*/dumbbell_biceps_curls.json` | ✅ | → `Calib.toml` (triangulation) |
| `videos/*/dumbbell_biceps_curls.mp4` (4) | ✅ | → entrée RTMPose |
| `joints3d_25/dumbbell_biceps_curls.json` | ✅ | → ground truth (validation + scaling) |
| `rep_ann.json` (clé biceps) | ✅ | → labels `rep_index` |
| `smplx/*.json` | ❌ | non utilisé |
| `gpp/*.json` | ❌ | non utilisé |

On travaille sur le sujet **s04** (4 caméras), vidéo complète ≈ 17.6 s, 5 répétitions.

---

## 5. JSON → CSV / Calib.toml (la conversion)

Script : `s04/code/s1_setup_run.py`. Trois conversions :

### 5.1 Calibration JSON → **Pose2Sim `Calib.toml`** (avec le fix −R·T)
```python
R = np.array(d["extrinsics"]["R"]).reshape(3,3)
T = np.array(d["extrinsics"]["T"]).reshape(3)        # camera centre (IMAR)
tvec = -R @ T                                         # <-- le fix (Pose2Sim veut le tvec OpenCV)
K = [[f[0],0,c[0]],[0,f[1],c[1]],[0,0,1]]
rvec = cv2.Rodrigues(R)[0]                            # rotation -> vecteur Rodrigues
# -> bloc [cam_<id>] : matrix=K, distortions=[k1,k2,p1,p2], rotation=rvec, translation=tvec
```

### 5.2 joints3d_25 JSON → **`joints3d_25.csv`** (ground truth)
```python
J = np.array(d["joints3d_25"])      # (T,25,3)
# header: frame, J0_x, J0_y, J0_z, ..., J24_z   (1 + 75 = 76 colonnes)
# une ligne par frame
```

### 5.3 rep_ann.json → **`rep_annotations.csv`**
```python
bounds = d["dumbbell_biceps_curls"]  # [115,244,380,514,652,784]
# rows: rep_index, start_frame, end_frame  (5 reps)
```

---

## 6. Récap (flux Fit3D → pipeline)

```
Fit3D train (8 sujets × 47 exercices × 4 caméras)
        │  on prend : sujet s04, exercice dumbbell_biceps_curls
        ▼
  videos (4)          → RTMPose (2D) → triangulation (Calib.toml, −R·T) → 3D markerless (.trc)
  camera_parameters   → Calib.toml                          ↑
  joints3d_25         → joints3d_25.csv → VALIDATION (r=0.99) + scaling arm26
  rep_ann             → rep_annotations.csv → labels reps
  (smplx, gpp = non utilisés)
```

**Points-clés présentation :**
1. Fit3D = 8 sujets × 47 exercices × 4 caméras synchro, avec **GT 3D + calibration + vidéo**.
2. On en utilise **un seul exercice** (dumbbell_biceps_curls) et **3 fichiers** (calib, vidéos, GT).
3. La conversion JSON→Calib.toml applique le **fix −R·T** (clé du déblocage de la triangulation).
4. Le `joints3d_25` (joints 14/15/16) sert de **ground truth** pour valider et scaler le modèle.
