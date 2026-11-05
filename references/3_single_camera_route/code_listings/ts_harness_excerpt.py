"""Excerpt of the module `ts_harness`, as printed by the executed single-camera notebook
(notebook/arm26_pipeline_report.ipynb, cell 59, via inspect.getsource).

Only the functions the notebook displays are here; the full module lives on the GPU
workstation that ran the notebook (/home/idriss/arm26_bridge). The code is unchanged.
"""

# ---- notebook cell 59
def load_level(level, aug_cum=False, aug_kin=False, sel_joints=None, aug_scale=False):
    """Minimal, physically-motivated transforms of the raw angle signal (NOT hand-crafted
    features): aug_kin adds velocity+acceleration (inverse dynamics needs q,q',q'');
    aug_cum adds the gravity-load impulse (fatigue = time-integral of muscle effort).
    aug_scale appends the subject's arm26 scaling from the .osim (humerus & forearm
    length + scale factors) as constant channels -> 'L1 = .mot + .osim'.
    sel_joints: for L2 (3D)/L3 (2D), keep only these joint indices (COCO-17 order:
    RSh=6, REl=8, RWr=10, ...) -> exercise-specific channel selection."""
    man = json.load(open(os.path.join(ML, "manifest.json")))
    fps_by = {m["run"]: float(m.get("fps", 50.0)) for m in man["runs"]}
    ncoord = {"L2": 3, "L3": 2}.get(level)
    runs = []
    for m in man["runs"]:
        d = np.load(os.path.join(ML, m["run"] + ".npz"))
        X = d[level].astype(np.float32); L1 = d["L1"].astype(np.float32)
        if sel_joints is not None and ncoord is not None:
            Xj = X.reshape(len(X), -1, ncoord)[:, sel_joints, :]          # keep chosen joints only
            X = Xj.reshape(len(X), -1).astype(np.float32)
        n = len(X); dt = 1.0 / fps_by[m["run"]]; extra = []
        if aug_kin:
            vel = np.gradient(X, axis=0); acc = np.gradient(vel, axis=0)      # q', q'' of the level's own channels
            extra += [vel, acc]
        if aug_cum:
            # gravity-load proxy: forearm+load moment arm ~ sin(shoulder_elev + elbow_flex)
            g = np.abs(np.sin(np.radians(L1[:, 0] + L1[:, 1]))).astype(np.float32)  # instantaneous load
            cumg = (np.cumsum(g) * dt).astype(np.float32)                    # gravity impulse (real seconds, unnormalized)
            elapsed = (np.arange(n) * dt).astype(np.float32)                 # elapsed time (s) -> fatigue is time-driven
            tnorm = (np.arange(n) / max(n-1, 1)).astype(np.float32)          # phase within the run
            extra += [g[:, None], cumg[:, None], elapsed[:, None], tnorm[:, None]]
        if aug_scale and "scale" in d:
            sc = d["scale"].astype(np.float32)                               # [hum_m, fore_m, x_hum, x_fore]
            extra += [np.tile(sc, (n, 1))]                                   # constant over time (subject geometry)
        if extra:
            X = np.concatenate([X] + extra, axis=1)
        runs.append((m["subj"], X.astype(np.float32), d["Y"].astype(np.float32)))
    return runs, man["targets"]
