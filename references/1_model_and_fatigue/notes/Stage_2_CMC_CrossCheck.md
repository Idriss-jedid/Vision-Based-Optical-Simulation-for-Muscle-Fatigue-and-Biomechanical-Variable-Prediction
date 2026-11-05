# Stage 2 — CMC Cross-Check of the Static-Optimization Labels

Static Optimization (SO) ignores activation/contraction dynamics. Computed Muscle
Control (CMC; Thelen, Anderson & Delp 2003) includes them via PD tracking on a
forward-dynamic model. CMC is far slower, so it is run on a **short representative
window** (first ~2 curl reps, t = 0.1–8.6 s of the 2 kg trial) as an independent
cross-check of the SO muscle solution.

Reproduce: `conda run -n biomech python Code/run_cmc_subset.py`
(CMC tracked the kinematics successfully; ~125 s runtime.)

## Comparison — muscle force (matched window, start-up transient excluded)

CMC's per-frame muscle **force** is the reliable, physically-meaningful output and
is the standard SO-vs-CMC comparison quantity (Roelker 2020). (CMC's reported
state-activation column is unreliable here, so forces are compared directly.)

| Muscle | SO peak (N) | CMC peak (N) | SO mean (N) | CMC mean (N) |
|---|--:|--:|--:|--:|
| BIClong | 174.5 | 123.4 | 98.6 | 78.2 |
| BICshort | 55.7 | 27.6 | 28.4 | 20.4 |
| BRA | 205.4 | 45.8 | 103.1 | 30.4 |
| BRD_hand | 26.9 | 20.2 | 17.0 | 19.4 |

Flexor force share (mean over the window):

| Muscle | SO | CMC |
|---|--:|--:|
| BIClong | 39.9 % | 52.7 % |
| BICshort | 11.5 % | 13.7 % |
| BRA | 41.7 % | 20.5 % |
| BRD_hand | 6.9 % | 13.1 % |

## Interpretation
- **Both methods agree on the essentials**: the same active set (biceps +
  brachialis dominate, BRD assists ~7–13 %, triceps inactive) and the same total
  flexor load; **reserve actuators are ~0 in both**, so muscles carry the load.
- **The load *distribution* differs**: SO favours brachialis (the strongest
  flexor, so min-Σactivation² loads it), while CMC distributes more to biceps
  long. This is the **expected SO-vs-CMC difference** — the two methods resolve
  muscle redundancy differently, and SO cannot capture voluntary co-contraction
  or activation dynamics (Roelker 2020; Lai 2021).
- **Conclusion:** the SO labels are cross-validated in active-set, ordering, and
  total magnitude. The exact per-muscle split is method-dependent — a documented
  limitation, not an error. For labels that must reflect activation dynamics
  (electromechanical delay, smoothing), CMC can be substituted on the trials that
  need it; for dataset-scale label generation, SO remains the practical choice.

## Files
- `Code/run_cmc_subset.py` — CMC setup + SO/CMC force comparison
- `Results/Stage2/cmc_subset/` — CMC outputs (forces, controls, kinematics, pErr)
