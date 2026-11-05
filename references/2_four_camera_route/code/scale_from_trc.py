# pyright: reportMissingImports=false
"""
Scaling VISION-ONLY (sans Vicon) : les facteurs d'echelle sont calcules directement depuis le
.trc (markerless) -> longueurs medianes RShoulder-RElbow (humerus) et RElbow-RWrist (avant-bras),
divisees par les longueurs par defaut d'arm26. Produit arm26_<subj>_scaled_trc.osim et compare
aux facteurs Vicon (modele scale existant). biomech env.
"""
import os, glob
import numpy as np
import opensim as osim

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
BATCH = os.path.join(ROOT, "batch")
BASE = os.path.join(ROOT, "Model", "arm26_paper_loaded_brd_elbow_research.osim")
MKIDX = {"RShoulder": 17, "RElbow": 18, "RWrist": 19}   # ordre HALPE dans le .trc


def model_lengths_masses(path):
    """retourne (humerus_len, forearm_len, humerus_mass, forearm_mass) d'un modele .osim."""
    m = osim.Model(path); s = m.initSystem()
    def jc(j): return np.array([m.getJointSet().get(j).getChildFrame().getPositionInGround(s).get(i) for i in range(3)])
    def mk0(n): return np.array([m.getMarkerSet().get(n).getLocationInGround(s).get(i) for i in range(3)])
    ua = float(np.linalg.norm(jc("r_shoulder") - jc("r_elbow")))
    fa = float(np.linalg.norm(jc("r_elbow") - mk0("r_radius_styloid")))
    def bmass(b): return float(m.getBodySet().get(b).getMass())
    return ua, fa, bmass("r_humerus"), bmass("r_ulna_radius_hand")


def read_trc_arm(path):
    """retourne RShoulder, RElbow, RWrist (N x 3) depuis le .trc."""
    rows = []
    for ln in open(path).read().splitlines():
        p = ln.split("\t")
        if len(p) > 59:
            try:
                float(p[0]); float(p[1]); rows.append([float(x) if x.strip() else np.nan for x in p])
            except ValueError:
                continue
    d = np.array(rows)
    def mk(name):
        c = 2 + (MKIDX[name] - 1) * 3; return d[:, c:c + 3]
    return mk("RShoulder"), mk("RElbow"), mk("RWrist")


def main():
    # longueurs par defaut d'arm26
    ua_d, fa_d, _, _ = model_lengths_masses(BASE)
    print("arm26 defaut : humerus=%.4f m, avant-bras=%.4f m\n" % (ua_d, fa_d))

    subs = sorted([os.path.basename(p) for p in glob.glob(os.path.join(BATCH, "s*"))
                   if os.path.isdir(p) and os.path.exists(os.path.join(p, "opensim"))])
    print("%-5s | %-22s | %-22s | %-18s" % ("subj", "scale_H (Vicon -> TRC)", "scale_F (Vicon -> TRC)", "hum_mass V->TRC"))
    print("-" * 78)
    diffs = []
    for subj in subs:
        trc = sorted(glob.glob(os.path.join(BATCH, subj, "pose2sim", "pose-3d", "*filt_butterworth.trc")))
        vic_model = os.path.join(BATCH, subj, "opensim", "arm26_%s_scaled.osim" % subj)
        if not trc or not os.path.exists(vic_model):
            print("%-5s  (manque trc/modele)" % subj); continue
        # longueurs markerless (median sur frames)
        sh, el, wr = read_trc_arm(trc[0])
        ua_trc = float(np.nanmedian(np.linalg.norm(sh - el, axis=1)))
        fa_trc = float(np.nanmedian(np.linalg.norm(el - wr, axis=1)))
        sfh, sff = ua_trc / ua_d, fa_trc / fa_d
        # applique le scaling OpenSim -> modele vision-only
        m2 = osim.Model(BASE); s2 = m2.initSystem(); ss = osim.ScaleSet()
        for body, sf in [("r_humerus", sfh), ("r_ulna_radius_hand", sff)]:
            sc = osim.Scale(); sc.setSegmentName(body); sc.setScaleFactors(osim.Vec3(sf, sf, sf)); sc.setApply(True)
            ss.cloneAndAppend(sc)
        out = os.path.join(BATCH, subj, "opensim", "arm26_%s_scaled_trc.osim" % subj)
        m2.scale(s2, ss, False, -1.0); m2.printToXML(out)
        # comparaison avec Vicon (modele existant)
        ua_v, fa_v, hm_v, _ = model_lengths_masses(vic_model)
        _, _, hm_t, _ = model_lengths_masses(out)
        sfh_v, sff_v = ua_v / ua_d, fa_v / fa_d
        diffs.append(abs(sfh - sfh_v)); diffs.append(abs(sff - sff_v))
        print("%-5s | %.3f -> %.3f  (d=%.3f) | %.3f -> %.3f  (d=%.3f) | %.2f -> %.2f kg" %
              (subj, sfh_v, sfh, abs(sfh - sfh_v), sff_v, sff, abs(sff - sff_v), hm_v, hm_t))
    print("\nd= moyen des facteurs d'echelle (Vicon vs TRC) = %.4f  (= %.2f %%)" %
          (np.mean(diffs), 100 * np.mean(diffs)))
    print("-> modeles vision-only ecrits : batch/<subj>/opensim/arm26_<subj>_scaled_trc.osim")


if __name__ == "__main__":
    main()
