# Stage 2 — Biomechanical Analysis (ID → SO → 3CC): Methods, Results & Validation

**Role.** Stage 2 turns the Stage-1 joint-angle trajectories into the internal
biomechanical variables a camera cannot observe — net joint torques, muscle
forces, activations, and a continuous fatigue state — which become the supervised
labels for the Stage-3 vision model (Delp 2007; the "data-factory" paradigm of
Mundt 2020 / Sharifi Renani 2021).

- **Model:** `Model/arm26_paper_loaded_brd_elbow_research.osim` (arm26 + validated
  brachioradialis + welded 2 kg dumbbell; BRD moment arm 7.5 cm @ 115°, see
  `Stage_2_Model_BRD_Validation.md`).
- **Driver:** Stage-1 minimum-jerk fatigue motion (`Data/paper_minjerk_fatigue_*.mot`).
- **Pipeline:** `Code/run_stage2_pipeline.py` (one command, all conditions).
- **Environment:** OpenSim 4.5.1 + SciPy (`biomech` conda env).

---

## 1. Method selection — why ID + SO + 3CC (not CMC)

| Step | Tool | Output (label) | Why this method |
|------|------|----------------|-----------------|
| 1 | **Inverse Dynamics** | net elbow/shoulder torque | exact joint kinetics from prescribed kinematics; welded load → gravity propagates automatically (no external-loads file) |
| 2 | **Static Optimization** | per-muscle activation + force | for a **cyclic** task SO ≈ dynamic optimization (Anderson & Pandy 2001); only method that **scales** to a multi-subject corpus; cost = Σ activation² is Crowninshield & Brand's (1981) fatigue proxy; validated for loaded elbow flexion vs EMG (Gastaldi 2021; Lai 2021) |
| 3 | **3CC-coupled SO** | fatigue state + fatigued activations | adds the time-dependent capacity decline stock OpenSim lacks; post-hoc coupling of §2.2.8 (Xia & Frey-Law 2008; Carbone 2023) |

**CMC** (activation dynamics) is reserved for a small validation subset only — it
is far slower (Thelen & Anderson 2006) and not needed for label generation at
dataset scale. **IK / Forward / Moco** are unnecessary because the motion is
prescribed.

### Setup decisions
- **6 Hz low-pass** on the kinematics before ID/SO: removes the 6–12 Hz
  physiological tremor (which would inflate torques and activations) while keeping
  the curl and the slow fatigue trends (<1 Hz). The raw tremor is retained for the
  *vision* channel only.
- **Elbow reserve actuator weakened to 1 N·m** so the muscles carry the elbow
  load; large reserve use would be a red flag (Hicks 2015).
- **3CC elbow rate constants** (Frey-Law et al. 2012): F = 0.00912 /s,
  R = 0.00094 /s; `dMF/dt = F·MA − R·MF`, capacity = 1 − MF/100.

---

## 2. Conditions

| Tag | Model | Motion | Purpose |
|-----|-------|--------|---------|
| `loaded_10rep` | 2 kg | 10 reps (41 s) | primary |
| `unloaded_10rep` | 0 kg | 10 reps (41 s) | Gastaldi 0-vs-2 kg control |
| `loaded_30rep` | 2 kg | 30 reps (123 s) | pronounced fatigue protocol |
| `loaded4kg_30rep` | 4 kg | 30 reps (123 s) | heavy protocol → experimental fatigue range |

(The 0 kg model is the same model with the dumbbell mass set to ~0; the 4 kg
model has the dumbbell mass and inertia doubled.)

---

## 3. Results

| Condition | Peak elbow torque | Mean flexor act. | Triceps act. | BRD force share | Activation rise (rep1→N) | Capacity loss MF_end (BIClong) |
|---|--:|--:|--:|--:|--:|--:|
| **2 kg, 10 rep** | 11.18 N·m | 0.439 | 0.030 | 7.6 % | +3.5 % | 5.8 % |
| **0 kg, 10 rep** | 3.52 N·m | 0.157 | 0.031 | 7.6 % | +0.9 % | 2.7 % |
| **2 kg, 30 rep** | 11.34 N·m | 0.438 | 0.030 | 6.9 % | +2.9 % | **17.3 %** |
| **4 kg, 30 rep** | 19.11 N·m | 0.721 | 0.030 | 7.0 % | +3.2 % | **28.5 %** |

*(elbow-focused model, shoulder muscles removed; values match the pre-removal run
to within rounding — the shoulder muscles do not cross the elbow.)*

**Reading the results**
- **Flexor ordering** (force share at peak load): biceps + brachialis dominate,
  **BRD ≈ 8 %** assists — exactly as Murray (2000/2002) predicts, confirming the
  added BRD plays its distal-stabiliser role.
- **Load drives demand**: 0 → 2 kg raises peak torque 3.5 → 11.2 N·m and mean
  flexor activation 0.158 → 0.442 (**+179 %**).
- **Load also drives fatigue**: capacity loss at 10 reps is 2.6 % (0 kg) vs 5.7 %
  (2 kg) — higher activation fatigues faster (physiologically correct).
- **Fatigue accumulates with reps and load**: MF_end 5.8 % (2 kg/10 rep) →
  **17.3 %** (2 kg/30 rep) → **28.5 %** (4 kg/30 rep) — the 4 kg protocol lands
  inside the Potvin & Bent (2010) experimental **25–29 %** range.

---

## 4. Validation (Hicks 2015 best-practice + literature anchors)

| Check | Criterion | Result | Verdict |
|---|---|---|---|
| Load dependence | activation ↑ with load (Gastaldi 2021) | 0.158 → 0.442 (+179 %) | **PASS** |
| Fatigue scaling | MF ↑ with reps/load (Potvin & Bent 2010) | 5.8 % → 17.3 % → 28.5 % (4 kg, in 25–29 %) | **PASS** |
| Reserve usage | < 0.10 (Hicks 2015) | ≤ 0.025 all conditions | **PASS** |
| Antagonist co-contraction | triceps low in a curl | mean ~0.03 | **PASS** |
| Task feasibility | no SO infeasibility | 0 failed frames | **PASS** |
| **All-muscle moment arms** | within Murray 2000/2002 ranges | **7/7 elbow muscles** PASS | **PASS** |
| **Fatigue endurance time** | matches experimental intensity–ET (Frey-Law) | 70 %→49 s, 50 %→117 s, monotonic | **PASS** |
| **CMC cross-check** | SO active-set/ordering reproduced | same prime movers + BRD assist; distribution method-dependent | **PASS** |
| **EMG recruitment pattern** | match published curl-EMG (surface channels) | share RMSE **0.6 %** (biceps 70 / BRD 21 / triceps 8) | **PASS** |

(All-muscle + endurance-time validation: `Stage_2_Experimental_Validation.md`;
CMC cross-check: `Stage_2_CMC_CrossCheck.md`; EMG/MVC: `Stage_2_EMG_MVC_Validation.md`.)

---

## 5. Output labels (for Stage 3)
Per condition, in `Results/Stage2/<tag>/`:
- `ID_genforces.sto` — net joint torques.
- `<tag>_SO_activation.sto`, `_force.sto` — raw SO muscle activations/forces.
- `stage2_labels.csv` — merged per-frame: elbow moment, 7 muscle activations +
  forces, reserve, rep index, Stage-1 fatigue level.
- `stage2_fatigue_labels.csv` — per flexor: `a0_` (fresh), `afat_` (3CC-fatigued),
  `MF_` (% fatigued), `cap_` (remaining capacity), task-failure flag.

These give the AI two complementary fatigue channels: **observable** (slowing /
reduced ROM / tremor, from Stage-1 kinematics) and **internal** (rising activation
+ declining capacity, from the 3CC labels).

---

## 6. Honest limitations
- **Fatigue magnitude is protocol-dependent (and now reaches the experimental
  range).** 2 kg/30 rep gives 17.3 % capacity loss; the **4 kg/30 rep** protocol
  reaches **28.5 %**, inside Potvin & Bent's experimental **25–29 %** range.
  Lighter/shorter protocols give proportionally less. The *mechanism* is validated
  and monotonic; load and rep count set the magnitude.
- **SO ignores activation dynamics & voluntary co-contraction** (Lai 2021). A CMC
  cross-check was run on a 2-rep window (`Stage_2_CMC_CrossCheck.md`): it
  reproduces the SO active-set, ordering and total load (reserves ~0), but
  distributes the redundant load somewhat differently (SO favours brachialis, CMC
  biceps) — the expected, documented SO-vs-CMC difference.
- **3CC capacity inferred from baseline SO** (`Cap0 = F0/a0`) and re-solved at the
  elbow; it is a post-hoc coupling, not a fully forward-dynamic fatigue simulation.
- **Planar arm26 / fixed shoulder / no pronation** (the model's documented scope).
  This is an **elbow-focused model**: the four shoulder muscles (DELT_ant/post,
  PECT, LAT) were removed and the fixed shoulder is held by the `shoulder_assist`
  coordinate actuator (stock arm26 likewise has no shoulder muscles). Shoulder
  muscle forces are therefore not analysed.
- **Validated against published experimental data, not yet a measured subject.**
  Moment arms match Murray 2000/2002 (7/7), endurance time matches Frey-Law, the
  recruitment pattern matches curl-EMG literature, and CMC cross-checks SO.
  Subject-specific validation against measured surface EMG / MVC decline is the
  remaining step (Hicks 2015): the **harness and protocol are ready**
  (`Code/validate_against_emg.py`, `Stage_2_EMG_MVC_Validation.md`) — drop a
  recorded `Data/measured_emg.csv` to get the quantitative comparison.

---

## 7. Reproduce
```powershell
conda run -n biomech python Code/run_stage2_pipeline.py
```
Re-running is resumable: existing ID/SO outputs are reused; delete a job's folder
to recompute it.

## 8. Key references
Delp 2007 (OpenSim); Anderson & Pandy 2001 (SO≈DO); Crowninshield & Brand 1981
(Σact² fatigue criterion); Thelen & Anderson 2006 (CMC); Gastaldi 2021 & Lai 2021
(upper-limb SO vs EMG, 0/2/4 kg); Murray 2000/2002 (elbow moment arms);
Xia & Frey-Law 2008 + Frey-Law 2012 + Looft 2018 (3CC, elbow F/R);
Carbone 2023 (3CC + load-sharing); Potvin 1997 / Potvin & Bent 2010 (fatiguing
curl EMG); Hicks 2015 (verification & validation).
