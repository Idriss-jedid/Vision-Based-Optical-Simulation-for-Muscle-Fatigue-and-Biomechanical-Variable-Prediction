"""Excerpt of the module `arm26_from_video`, as printed by the executed single-camera notebook
(notebook/arm26_pipeline_report.ipynb, cells 12, 16, 21, via inspect.getsource).

Only the functions the notebook displays are here; the full module lives on the GPU
workstation that ran the notebook (/home/idriss/arm26_bridge). The code is unchanged.
"""

# ---- notebook cell 12
def load_from_wham(wham_pkl, person=0):
    """Regress joints (camera frame) from the WHAM SMPL mesh vertices.

    COCO-17 layout is kept (hips/shoulders drive gravity alignment), but the RIGHT
    arm (shoulder/elbow/wrist) is taken from the H36M regressor: the COCO *surface*
    shoulder under-measures the humerus by ~12%, whereas H36M's shoulder-JOINT centre
    gives an accurate humerus (mean err ~+3% vs Vicon instead of -12%).
    """
    import joblib
    d = joblib.load(wham_pkl)
    if person not in d:
        person = sorted(d.keys())[0]
    verts = np.asarray(d[person]["verts_cam"], dtype=float)          # (T,6890,3)
    Jco = np.load(os.path.join(BODY_MODELS, "J_regressor_coco.npy")) # (17,6890)
    kp = np.einsum("jk,tkc->tjc", Jco, verts)                        # (T,17,3) COCO
    # COCO elbow/wrist are accurate (forearm err ~+2%); only the shoulder is the
    # problem. COCO's *surface* shoulder sits lateral/inferior -> humerus ~-12%;
    # H36M's *joint* shoulder sits medial/superior -> humerus ~+6% (up to +16% on
    # short-armed subjects). The true GH-joint centre is between them, so we BLEND
    # the two shoulder points (best worst-case: max humerus err 9% vs 13-16%).
    coco_sh = kp[:, C["RSh"]].copy()
    Jh = np.load(os.path.join(BODY_MODELS, "J_regressor_h36m.npy"))  # H36M Rshoulder = 14
    h36m_sh = np.einsum("jk,tkc->tjc", Jh, verts)[:, 14]
    kp[:, C["RSh"]] = 0.5 * (coco_sh + h36m_sh)
    return kp


def gravity_align(kp):
    """Rotate all joints so torso-up (midShoulder-midHip) -> world +Y (Y-up, like the .trc)."""
    mid_sh = (kp[:, C["RSh"]] + kp[:, C["LSh"]]) / 2.0
    mid_hip = (kp[:, C["RHip"]] + kp[:, C["LHip"]]) / 2.0
    up_raw = unit(np.nanmean(mid_sh - mid_hip, axis=0))
    R = rot_between(up_raw, np.array([0.0, 1.0, 0.0]))
    return kp @ R.T, R


def angles(sh, el, wr):
    """Return q_sh (shoulder elevation, deg) and q_el (elbow flexion, deg).

    q_el uses the ANATOMICAL convention (0 = full extension, ~150 = full flexion),
    matching the arm26 model's r_elbow_flex (range 0..130 deg) AND the Fit3D Vicon GT.
    NOTE: Vision's angles_anthro used (180 - this) for its LightGBM features; that
    inverted form must NOT be fed to the real OpenSim model.
    """
    ua, fa = el - sh, wr - el
    cu = (ua * fa).sum(1) / (np.linalg.norm(ua, axis=1) * np.linalg.norm(fa, axis=1) + 1e-9)
    q_el = np.degrees(np.arccos(np.clip(cu, -1, 1)))        # anatomical elbow flexion
    # shoulder elevation = angle of humerus from the downward vertical (Y-up frame).
    # proper Euclidean form (not the L1 approximation from the LightGBM features).
    q_sh = np.degrees(np.arccos(np.clip(-ua[:, 1] / (np.linalg.norm(ua, axis=1) + 1e-9), -1, 1)))
    hl = float(np.nanmedian(np.linalg.norm(ua, axis=1)))
    fl = float(np.nanmedian(np.linalg.norm(fa, axis=1)))
    return q_sh, q_el, hl, fl


# ---- notebook cell 16
def scale_arm26(hl, fl, out_osim):
    m = osim.Model(ARM26); s = m.initSystem()
    sfh, sff = hl / HUM_DEF, fl / FOR_DEF
    ss = osim.ScaleSet()
    for body, sf in [("r_humerus", sfh), ("r_ulna_radius_hand", sff)]:
        sc = osim.Scale(); sc.setSegmentName(body)
        sc.setScaleFactors(osim.Vec3(sf, sf, sf)); sc.setApply(True)
        ss.cloneAndAppend(sc)
    m.scale(s, ss, False, -1.0)
    m.printToXML(out_osim)
    return out_osim, sfh, sff


# ---- notebook cell 21
def run_id(model_path, motion, outdir, t0, t1):
    out = os.path.join(outdir, "ID_genforces.sto")
    idt = osim.InverseDynamicsTool()
    idt.setModelFileName(model_path); idt.setCoordinatesFileName(motion)
    idt.setLowpassCutoffFrequency(LOWPASS_HZ)
    ex = osim.ArrayStr(); ex.append("Muscles"); idt.setExcludedForces(ex)
    idt.setStartTime(t0); idt.setEndTime(t1)
    idt.setResultsDir(outdir); idt.setOutputGenForceFileName("ID_genforces.sto")
    idt.run()
    return out


def run_so(model_path, motion, outdir, tag, t0, t1):
    model = osim.Model(model_path)
    so = osim.StaticOptimization(); so.setName("SO")
    so.setUseModelForceSet(True); so.setActivationExponent(2.0); so.setUseMusclePhysiology(True)
    so.setStartTime(t0); so.setEndTime(t1)
    model.addAnalysis(so); model.initSystem()
    tool = osim.AnalyzeTool(model); tool.setName(tag); tool.setModel(model)
    tool.setInitialTime(t0); tool.setFinalTime(t1)
    tool.setLowpassCutoffFrequency(LOWPASS_HZ)
    tool.setCoordinatesFileName(motion); tool.setLoadModelAndInput(True)
    tool.setResultsDir(outdir); tool.run()
    return os.path.join(outdir, "%s_SO_activation.sto" % tag), os.path.join(outdir, "%s_SO_force.sto" % tag)
