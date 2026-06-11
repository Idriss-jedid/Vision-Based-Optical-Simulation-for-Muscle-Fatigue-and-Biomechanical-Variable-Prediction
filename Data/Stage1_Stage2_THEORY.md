# Stage 1 + Stage 2 — Tafsir théorique (9awa3id + formules + examples)

Tafsir pédagogique mte3 `Stage1_Stage2_DETAILS.md` : théorie + règles + formules +
examples b arqám. Kima cours.

---

# 📚 1. Minimum-Jerk — el théorie

## Chnowa houwa "jerk" ?
- **Position** `q` → **Vitesse** `q̇` (dérivée 1) → **Accélération** `q̈` (dérivée 2) → **Jerk** `q⃛` (dérivée 3).
- Jerk = "kifech tetbaddel el accélération". Jerk 3ali = 7araka metšannja (saccadée). Jerk menxfodh = 7araka na3ma.

> **Mثal:** fl tموبيل، ki el chauffeur ykabbes el frein b3ada → accélération tetbaddel bسرعة → **jerk 3ali** → ت7ess راsek yet7arrek. Ki yfarmel bالشويّة → jerk menxfodh → na3em.

## El 9a3ida: el 7araka el-bachariya t9allel el jerk
El dmagh yextár el trajectoire eli **t9allel** `∫(jerk)² dt` (Flash & Hogan 1985). El 7all el-riyadhi mte3 hadha = **polynôme degré 5** :
```
q(s) = q0 + (qf−q0)·(10s³ − 15s⁴ + 6s⁵),   s = t/T  (0→1)
```

## 3lèch hadha el polynôme bالضبط ?
5ater houwa el wa7id degré-5 eli y7a99e9 **6 šorout** :
- `q(0)=q0`, `q(1)=qf` (yebda w ywsel)
- `q̇(0)=0`, `q̇(1)=0` (vitesse=0 fl bداية w nihaya → ma fammaš entila9a faj2iya)
- `q̈(0)=0`, `q̈(1)=0` (accélération=0 fl el-atráf → ma fammaš choc)

## Mثal b arqám (mte3na) :
ROM = 100° (men 20° l 120°)، T = 1.5s.
- **f noss el wa9t** s=0.5 : `10(0.125) − 15(0.0625) + 6(0.03125) = 1.25 − 0.9375 + 0.1875 = 0.5` → ya3ni f noss el wa9t = noss el zawiya (70°). **Symétrique.**
- **Vitesse max** (f s=0.5) : el formule t3ti `q̇_max = 1.875·ROM/T = 1.875·100/1.5 = 125°/s`.
  - *El 9a3ida 1.875 :* `q̇(s)=30s²−60s³+30s⁴`، f s=0.5 → 7.5−7.5+1.875 = **1.875**.

> **El šakl :** el vitesse tetla3 kima **jaras (bell)** — batí2a fl bداية، asra3 fl wast، batí2a fl nihaya. Hadha kima el bachar 7a9i9i.

---

# 📚 2. Fatigue kinématique — linéaire vs saturante

## El 9a3ida
Kol rep، el 7araka tet3eb : **abta2 + ROM asghar + ra3cha (tremor)**. Nmaththlouha b level `f` men 0 (frais) l 1 (met3eb tomáman).

## FIRST = **linéaire** : `f = rep/(N−1)`
- rep1 → f=0، rep10 → f=1. Kol rep yna99es el ROM b nafs el me9dar.
- *Mثal :* ROM 120°→100° 3la 10 reps → kol rep yna99es **2.2°** (120, 117.8, 115.6, ...).

## FINAL = **saturante** : `f = (1−e^(−r·x))/(1−e^(−r))`
- *3lèch ?* el fatigue el-7a9i9iya tetla3 **bسرعة fl bداية ba3dha testa9err (plateau)** — ma tzidš l'∞ xattiyan.

> **Mثal 7ayet :** ki tjri، awwel da9i9a t7ess b ta3eb bسرعة، ba3dha ywsel l 7ala "thábta" ma tzidš b nafs el sor3a. Hadha saturante، mch linéaire.

---

# 📚 3. Moment arm `r = −dL/dθ` — el nadhariya (mohemma !)

## El su2al : kifech muscle ydir torque ?
Muscle yšodd b **force F** (xatt mosta9im). El 3dham ydour. **9addèch torque ya3ti ?**

## El 9a3ida = **principe des travaux virtuels (virtual work)**
El šoghl eli ya3melou el muscle = el šoghl eli ya3melou el joint :
```
F · δL  =  τ · δθ        (šoghl muscle = šoghl joint)
```
- `δL` = 9addèch el muscle t9asser ki el joint ydour b `δθ`.
- 9smna : `τ = F · (δL/δθ) = F · r`, **m3a r = −dL/dθ** (el sáleb 5ater el flexor y9asser ki el coude yentwi → dL<0 → r>0).

→ **`r` = moment arm = 9addèch el muscle "9wi" 3al joint** (b el mètre).

## Mثal b arqám (mte3na) :
- BIClong y9asser men 41cm l 37cm ki el coude yentwi men 20° l 120° (100° = 1.745 rad).
- `r ≈ ΔL/Δθ = 0.04m / 1.745 = 0.023m = 2.3cm`. (fl table mte3na : r men 1.99 l 4.94 cm 7sab el zawiya).
- ki F = 250N → **torque = 250 × 0.025 = 6.25 N·m**.

## 3lèch el shoulder muscles enlevés (r=0) ?
DELT, PECT, LAT **ya3brou el ktef، mch el coude**. Ki el **coude** ydour، toulhom **ma yetbaddelš** → `dL/dθ_coude = 0` → **r = 0.00 cm** → torque = 0 3al coude → **inutiles lل curl** → enlevés (11→7).

> **Mثal :** law tšodd 7bal marbout fl báb، w t7arrek šay áxer b3id 3al báb — el 7bal ma y2aththerš. El shoulder muscle nafs el 7aja 3al coude.

---

# 📚 4. Inverse Dynamics (ID) — "9addèch torque ?"

## El 9a3ida = **Newton-Euler** (loi mte3 Newton lل rotation)
```
τ = I·α + (couple gravité) + (couple charge externe)
```
- `I·α` = inertie × accélération angulaire (saghir fl curl batí2).
- **couple gravité (el ahamm)** = `m · g · L · sin(angle men el vertical)`.

## Mثal b arqám (el dumbbell 2kg) :
- m=2kg, g=9.81, L≈0.30m (men el coude l el dumbbell).
- **ki el dhrè3 ofo9i** (sin=1) : `τ = 2 × 9.81 × 0.30 × 1 = 5.9 N·m` ≈ **el 6 N·m** eli 9isna ! ✅
- **ki el dhrè3 vertical** (sin≈0) : `τ ≈ 0` (el dumbbell fou9 el coude direct، ma fammaš lever).

> **Mثal 7ayet :** ki tšodd dumbbell w dhrè3ek **ofo9i** → t7ess b thi9el kbir (torque akbar). ki terf3ou **fou9** → axaff. ID y7seb hadha fl kol la7dha.
- *arqám mte3na (FIRST) :* |M| moyen = **6.85 N·m**, pic = **12.90 N·m** (gravité + accélération mte3 el 7araka).

---

# 📚 5. Static Optimization (SO) — "ánna muscle yexdem ?"

## El problème = **redondance musculaire**
3andna **1 mo3adla** (el torque = 6 N·m) ama **4 muscles fléchisseurs** (4 unknowns).
→ **3dad lá nihá2i mte3 7loul !** (tnajjem ta3mel el 6 N·m b alf tari9a). **Ánna wa7da yextár el jism ?**

## El 9a3ida = **Crowninshield-Brand (1981)**
El jism yextár el tawzi3 eli y9allel **majmou3 el activation tarbi3** :
```
min  Σ (F_m / Fmax_m)²  =  Σ (activation_m)²
s.c. Σ r_m · F_m = M(t)     (lázem ya3mlou el torque)
```

## 3lèch **tarbi3** (mch xatti) ?
- **tarbi3** y3á9eb el activation el-3áliya f ayy muscle → **ywazza3 el 7eml** 3al barcha muscles (ma ya3serš wa7ed barka). = a9all ta3eb.
- law **xatti**، el optimizer y7ott koll šay 3al muscle el-akthar efficient (r/Fmax akbar) barka → ghir wá9i3i.

> **Mثal 7ayet :** ki terfa3 kis thi9il، el dmagh ma yesta3melš biceps 100% w el l9rin 0% — ywazza3 bش ma tet3ebš bسرعة. SO = el mo3adla eli t7áki hadha.

## Mثal b arqám (mte3na، SO 3al FIRST) :
- BIClong = **40%** (el principal، r kbir)
- BRA = **30%**، BICshort = 19%، BRD = 17%
- TRIlong/lat/med = **1%** (extensors → ma yexedmouš، y9a3dou 3al plancher 0.01)

→ El curl ma7moula b **4 fléchisseurs**، BIClong + BRA dominent.

---

# 📚 6. Fatigue 3CC — el ODE (équation différentielle)

## El 9a3ida = **3 compartiments** (Xia & Frey-Law 2008)
Kol muscle 3andou 100% mte3 motor units (fibres)، m9assmin l 3 :
```
MA (active، 9á3ed yexdem) + MR (rest، jáhez) + MF (fatigued، met3eb) = 100%   (toujours)
```

## El mo3adla el-asásiya :
```
dMF/dt = F·MA − R·MF
```
- **`F·MA`** = kol ma texdem akthar (MA kbir) → fatigue tzid. `F=0.00912/s` = sor3a el ta3eb.
- **`−R·MF`** = el récupération (el met3eb yerja3 šwaya). `R=0.00094/s` = sor3a el récup، **batí2a bezzáf !**
- **capacité(t) = 1 − MF(t)/100** (kol ma MF yekber، el muscle ynajjem a9all).

## El no9ta el-théorique el-9awiya : **steady state**
ki `dMF/dt = 0` (isti9rár) : `MF_∞ = (F/R)·MA`.
- `F/R = 0.00912/0.00094 ≈ 9.7`.
- *Mثal :* law el muscle yexdem 3la **MA=10%** bástimrár → `MF_∞ = 9.7 × 10 = 97%` fatigué !
- → **3lhكa el effort el-mostamerr yefšel** : 7atta 10% effort mostamerr ywassel l 97% fatigue !

## El boucle positive (3lèch yefšel asra3) :
```
fatigue MF ↑ → capacité ↓ → bش ta3mel nafs el force lázem activation ↑ → MA ↑ → MF ↑↑ → ...
```
ki Σ capacité < demande → **task failure** (ma 3ádš ynajjem yerfa3).

> **Mثal 7ayet = batterie téléphone :**
> - testa3melou (MA) → yen9os bسرعة (F kbir). teš7nou (rest) → yerja3 **bالشويّة** (R saghir، 10× abta2).
> - kol rep el batterie adh3ef → lázem "ta3ser" akthar → tet3eb asra3. ba3d šwaya → **éteint (task failure)**.

## Mثal intégration (xotwa wa7da) :
MF=0, MA=20%, dt=0.01s :
`dMF = 0.01 × (0.00912×20 − 0.00094×0) = 0.01 × 0.1824 = 0.0018%`/xotwa.
3la 40s (4000 xotwa) b activation motazáyda → MF yekber tadrijiyan → el activation tetla3 3al reps.

---

# 📚 7. CMC — 3lèch + el far9 m3a SO

## El no9s mte3 SO :
SO **instantané** — yeftaredh el muscle yša33el/yetfi **fawran**. Ama fl el-wá9e3 el muscle yáxedh **wa9t** bش yša33el (electromechanical delay ~50ms) : `da/dt = (u−a)/τ`.

## CMC (Thelen 2003) :
yzid hadha el **dynamique d'activation** + forward dynamics + PD controller bش yetba3 el 7araka.
- *El natija el-motawa99a3a :* activations CMC **nafs el ordre** mte3 SO ama **an3am + mit2axxra šwaya** (el delay).
- **El accord binhom = validation** mte3 el SO labels.
- CMC **batí2 bezzáf** → 3melnáh 3la window 9sir (2 reps barka).

> **Mثal :** SO = "aša33el el muscle 40% **tawwa**". CMC = "neb3eth el amr، el muscle ywsel l 40% ba3d 50ms" — a9rab lل wá9e3، ama as3eb yet7seb.

---

# 🎯 Récap b mثal wa7ed (el chaîne théorique)
```
min-jerk (jerk min)  → angle na3em
   → r=−dL/dθ        → ánna muscle y2aththr 3al coude (shoulder r=0 → enlevé)
   → ID (Newton)     → torque = m·g·L·sinθ ≈ 6 N·m
   → SO (min Σa²)    → BIClong 40%, BRA 30%, triceps 0%
   → 3CC (dMF/dt=F·MA−R·MF) → batterie tedh3of → activation↑ → task failure
   → CMC             → validation (m3a delay)
```
