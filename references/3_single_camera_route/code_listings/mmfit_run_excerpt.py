"""Excerpt of the module `mmfit_run`, as printed by the executed single-camera notebook
(notebook/arm26_pipeline_report.ipynb, cell 50, via inspect.getsource).

Only the functions the notebook displays are here; the full module lives on the GPU
workstation that ran the notebook (/home/idriss/arm26_bridge). The code is unchanged.
"""

# ---- notebook cell 50
def run(subj):
    wham = f"{RES}/mmfit_{subj}_set0/{subj}_set0_bicep_curls/wham_output.pkl"
    if not os.path.exists(wham): return None
    od = os.path.join(OUT, subj); os.makedirs(od, exist_ok=True)
    # Stage 2-4: arm -> angles (30 fps) -> scale arm26 -> .mot
    kp = A.butter3d(A.gravity_align(A.load_from_wham(wham))[0], fs=FPS)
    sh, el, wr = kp[:, A.C["RSh"]], kp[:, A.C["REl"]], kp[:, A.C["RWr"]]
    q_sh, q_el, hl, fl = A.angles(sh, el, wr)
    t = np.arange(len(q_el)) / FPS
    scaled, sfh, sff = A.scale_arm26(hl, fl, os.path.join(od, "arm26.osim"))
    mot = A.write_mot(t, np.clip(q_sh, 0, 90), np.clip(q_el, 2, 128), os.path.join(od, "curl.mot"))
    # Stage 5: ID -> SO -> 3CC
    rep = {"subj": subj, "n_frames": len(q_el), "fps": FPS,
           "scale_hum": round(sfh, 3), "scale_fore": round(sff, 3)}
    try:
        labels, id_sto = A.run_biomech(scaled, mot, od, "curl")
        fat, fails, mf = A.run_3cc(labels, scaled, mot, od)
        lab = list(csv.DictReader(open(labels)))
        def c(n): return np.array([float(r[n]) for r in lab])
        tq = np.abs(c("elbow_moment_Nm"))
        rep["peak_torque_Nm"] = round(float(tq.max()), 2)
        rep["peak_force_N"] = {m: round(float(c("frc_" + m).max()), 1) for m in A.FLEXORS}
        rep["mean_act"] = {m: round(float(c("act_" + m).mean()), 3) for m in A.FLEXORS}
        rep["MF_end_pct"] = {m: round(float(mf[m]), 3) for m in A.FLEXORS}
        rep["biomech_ok"] = True
    except Exception as e:
        rep["biomech_ok"] = False; rep["error"] = str(e)
    # Validation vs pose_3d (elbow angle)
    gt = pose3d_elbow(subj)
    if gt is not None:
        lag = best_lag(q_el, gt); n = min(len(q_el), len(gt))
        if lag >= 0: a, b = q_el[:n-lag], gt[lag:lag+(n-lag)]
        else: a, b = q_el[-lag:n], gt[:n+lag]
        m = min(len(a), len(b)); a, b = a[:m], b[:m]
        msk = ~(np.isnan(a) | np.isnan(b))
        rep["val_elbow"] = {"pearson": round(float(np.corrcoef(a[msk], b[msk])[0, 1]), 3),
                            "mae_deg": round(float(np.mean(np.abs(a[msk] - b[msk]))), 2),
                            "wham_range": [round(float(q_el.min()), 0), round(float(q_el.max()), 0)],
                            "pose3d_range": [round(float(np.nanmin(gt)), 0), round(float(np.nanmax(gt)), 0)],
                            "n": int(msk.sum()), "lag": int(lag)}
    json.dump(rep, open(os.path.join(od, "report.json"), "w"), indent=2)
    return rep
