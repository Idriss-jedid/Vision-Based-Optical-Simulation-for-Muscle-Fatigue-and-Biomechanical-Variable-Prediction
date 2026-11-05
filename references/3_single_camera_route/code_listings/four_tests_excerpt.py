"""Excerpt of the module `four_tests`, as printed by the executed single-camera notebook
(notebook/arm26_pipeline_report.ipynb, cell 25, via inspect.getsource).

Only the functions the notebook displays are here; the full module lives on the GPU
workstation that ran the notebook (/home/idriss/arm26_bridge). The code is unchanged.
"""

# ---- notebook cell 25
def r_mae(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 else float("nan")
    return r, float(np.mean(np.abs(a-b))), float(np.std(b))


def umeyama(src, dst):
    """similarity transform (s,R,t) mapping src->dst (Nx3). Returns aligned src."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src-mu_s, dst-mu_d
    C = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[-1,-1] = -1
    R = U @ S @ Vt
    var = (s0**2).sum() / len(src)
    scale = np.trace(np.diag(D) @ S) / var
    t = mu_d - scale * R @ mu_s
    return (scale * (R @ src.T).T + t), scale, R, t


def run_one(subj, cam):
    wham = os.path.join(RES, f"{subj}_cam{cam}", f"cam_{cam}", "wham_output.pkl")
    if not os.path.exists(wham): return None
    va = vicon_arm(subj)
    if va is None: return None
    Sv, Ev, Wv, Jv = va

    # mono 3D (camera frame) + gravity aligned COCO
    kp_cam = load_from_wham(wham)
    kp = butter3d(gravity_align(kp_cam)[0])
    Sm, Em, Wm = kp[:, COCO_ARM[0]], kp[:, COCO_ARM[1]], kp[:, COCO_ARM[2]]
    n = min(len(Sm), len(Sv))
    Sm,Em,Wm = Sm[:n],Em[:n],Wm[:n]; Sv,Ev,Wv,Jv = Sv[:n],Ev[:n],Wv[:n],Jv[:n]
    R = {"subj": subj, "cam": cam, "frames": n}

    # ---------- Test 1: 2D reprojection error (px) ----------
    try:
        cal = tomllib.loads(open(os.path.join(VIS,"batch",subj,"pose2sim","calibration","Calib.toml"),"rb").read().decode())
        c = cal[f"cam_{cam}"]
        K = np.array(c["matrix"]); Rc = cv2.Rodrigues(np.array(c["rotation"]))[0]
        t = np.array(c["translation"]); dist = np.array(c["distortions"])
        tr = joblib.load(os.path.join(RES,f"{subj}_cam{cam}",f"cam_{cam}","tracking_results.pth"))
        vit = np.asarray(tr[sorted(tr.keys())[0]]["keypoints"])[:n]
        errs = []
        for i, gi in enumerate(GT_ARM):
            Xw = Jv[:, gi]
            Xc = (Rc @ Xw.T).T + t
            pts,_ = cv2.projectPoints(Xc.reshape(-1,1,3), np.zeros(3), np.zeros(3), K, dist)
            pts = pts.reshape(-1,2)
            errs.append(np.linalg.norm(pts - vit[:, COCO_ARM[i], :2], axis=1))
        R["test1_reproj_px"] = round(float(np.nanmean(errs)), 2)
    except Exception as e:
        R["test1_error"] = str(e)

    # ---------- Test 2: 3D RMSE (mm) + dir3D (deg) ----------
    try:
        mono = np.concatenate([Sm, Em, Wm], 0)         # (3n,3)
        vic  = np.concatenate([Sv, Ev, Wv], 0)
        aligned, sc, Rr, tt = umeyama(mono, vic)
        rmse_mm = float(np.sqrt(np.mean(np.sum((aligned-vic)**2,1)))) * 1000
        Sa, Ea, Wa = aligned[:n], aligned[n:2*n], aligned[2*n:]
        def dirdiff(u, v):
            u=u/ (np.linalg.norm(u,axis=1,keepdims=True)+1e-9); v=v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-9)
            return np.degrees(np.arccos(np.clip((u*v).sum(1),-1,1)))
        d_ua = dirdiff(Ea-Sa, Ev-Sv); d_fa = dirdiff(Wa-Ea, Wv-Ev)
        R["test2_rmse_mm"] = round(rmse_mm, 1)
        R["test2_dir3D_deg"] = round(float(np.nanmean(np.concatenate([d_ua,d_fa]))), 1)
    except Exception as e:
        R["test2_error"] = str(e)

    # ---------- Test 3: joint angles vs Vicon ----------
    # elbow (frame-invariant)
    qel_m = flex(Sm, Em, Wm); qel_v = flex(Sv, Ev, Wv)
    r_el, mae_el, _ = r_mae(qel_m, qel_v)
    R["test3_elbow_r"] = round(r_el, 3); R["test3_elbow_mae"] = round(mae_el, 2)
    # shoulder: mono Y-up (axis1), vicon Z-up (axis2)
    qsh_m = elev_from_axis(Sm, Em, 1)
    up_v = int(np.argmax(Jv.reshape(-1,3).max(0) - Jv.reshape(-1,3).min(0)))
    qsh_v = elev_from_axis(Sv, Ev, up_v)
    r_sh, mae_sh, std_sh = r_mae(qsh_m, qsh_v)
    R["test3_shoulder_r"] = round(r_sh, 3); R["test3_shoulder_mae"] = round(mae_sh, 2)
    R["test3_shoulder_std"] = round(std_sh, 2)

    # ---------- Test 4: subject scaling ----------
    hl = float(np.nanmedian(np.linalg.norm(Em-Sm, axis=1)))
    fl = float(np.nanmedian(np.linalg.norm(Wm-Em, axis=1)))
    R["test4_humerus_m"] = round(hl,4); R["test4_forearm_m"] = round(fl,4)
    R["test4_scale_hum"] = round(hl/HUM_DEF,3); R["test4_scale_fore"] = round(fl/FOR_DEF,3)
    return R
