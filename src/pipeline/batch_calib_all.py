# -*- coding: utf-8 -*-
"""BATCH Calib_world + vidéos pour tous les sujets : pour chaque sujet, place les 4
caméras dans le frame .trc (Procrustes Vicon->trc) -> D:\\p2s_blender\\<subj>_arm26_4cam\\
Calib_world.toml, et copie les 4 vidéos. Vérifie la reprojection. pose2sim_env."""
import glob, json, os, re, shutil
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__)); S04 = os.path.dirname(HERE); ROOT = os.path.dirname(S04)
SRC = r"D:\Download\fit3d\fit3d_train\train"; BATCH = os.path.join(ROOT, "batch")
EXERCISE = "dumbbell_biceps_curls"
import sys
SUBJECTS = sys.argv[1:] if len(sys.argv) > 1 else ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
CAMS = ["50591643", "58860488", "60457274", "65906101"]; SIZE = (900, 900)
VIDX = {"RShoulder": 14, "RElbow": 15, "RWrist": 16}; HIDX = {"RShoulder": 6, "RElbow": 8, "RWrist": 10}


def num(x): return repr(float(x))
def arr(a): return "[ " + ", ".join(num(v) for v in a) + ",]"
def mat(M): return "[ " + ", ".join("[ %s,]" % ", ".join(num(v) for v in r) for r in M) + ",]"


def trc_arm(proj):
    cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f and "filt" not in f]
    if not cands: cands = [f for f in glob.glob(os.path.join(proj, "pose-3d", "*.trc")) if "LSTM" not in f]
    L = open(sorted(cands)[-1]).read().splitlines(); mk = [m for m in L[3].split("\t") if m.strip()][2:]
    D = np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t")) > 5])
    def g(n): j = mk.index(n); c = 2 + 3 * j; return D[:, c:c + 3]
    return {k: g(k) for k in VIDX}


def procrustes(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    G = U @ S @ Vt; return G, mu_d - G @ mu_s


def do(subj):
    proj = os.path.join(BATCH, subj, "pose2sim"); POSE = os.path.join(proj, "pose")
    DST = os.path.join("D:\\", "p2s_blender", "%s_arm26_4cam" % subj); os.makedirs(DST, exist_ok=True)
    J = np.array(json.load(open(os.path.join(SRC, subj, "joints3d_25", EXERCISE + ".json")))["joints3d_25"])
    trc = trc_arm(proj); n = min(len(trc["RElbow"]), len(J))
    src, dst = [], []
    for k in VIDX:
        v = J[:n, VIDX[k]]; t = trc[k][:n]; good = ~np.isnan(t).any(1)
        src.append(v[good]); dst.append(t[good])
    G, g0 = procrustes(np.vstack(src), np.vstack(dst))
    txt = ""; rep = []
    for cam in CAMS:
        d = json.load(open(os.path.join(SRC, subj, "camera_parameters", cam, EXERCISE + ".json")))
        R = np.array(d["extrinsics"]["R"], float).reshape(3, 3); T = np.array(d["extrinsics"]["T"], float).reshape(3)
        f = np.array(d["intrinsics_w_distortion"]["f"], float).reshape(-1); c = np.array(d["intrinsics_w_distortion"]["c"], float).reshape(-1)
        k = np.array(d["intrinsics_w_distortion"]["k"], float).reshape(-1); p = np.array(d["intrinsics_w_distortion"]["p"], float).reshape(-1)
        Rp = R @ G.T; C = G @ T + g0; tvec = -Rp @ C
        K = [[f[0], 0.0, c[0]], [0.0, f[1], c[1]], [0.0, 0.0, 1.0]]; Km = np.array(K); dist = np.array([k[0], k[1], p[0], p[1], k[2]])
        kp = {}
        for fp in sorted(glob.glob(os.path.join(POSE, "cam_%s_json" % cam, "*.json"))):
            fr = int(re.findall(r"_(\d+)\.json$", fp)[0]); ppl = json.load(open(fp)).get("people", [])
            kp[fr] = np.array(ppl[0]["pose_keypoints_2d"]).reshape(-1, 3) if ppl else None
        rv = cv2.Rodrigues(Rp)[0]
        for key in VIDX:
            for fr in range(n):
                if fr not in kp or kp[fr] is None or kp[fr][HIDX[key], 2] < 0.3 or np.isnan(trc[key][fr]).any(): continue
                pr = cv2.projectPoints(trc[key][fr].reshape(1, 1, 3), rv, tvec, Km, dist)[0].reshape(2)
                rep.append(np.linalg.norm(pr - kp[fr][HIDX[key], :2]))
        txt += ("[cam_%s]\nname = \"cam_%s\"\nsize = %s\nmatrix = %s\ndistortions = %s\nrotation = %s\ntranslation = %s\nfisheye = false\n\n"
                % (cam, cam, arr([SIZE[0], SIZE[1]]), mat(K), arr([k[0], k[1], p[0], p[1]]), arr(cv2.Rodrigues(Rp)[0].reshape(-1)), arr(tvec)))
    txt += "[metadata]\nadjusted = false\nerror = 0.0\n"
    open(os.path.join(DST, "Calib_world.toml"), "w", newline="\n").write(txt)
    vdst = os.path.join(DST, "videos"); os.makedirs(vdst, exist_ok=True)
    for cam in CAMS:
        shutil.copy(os.path.join(SRC, subj, "videos", cam, EXERCISE + ".mp4"), os.path.join(vdst, "cam_%s.mp4" % cam))
    return float(np.median(rep))


def main():
    for subj in SUBJECTS:
        try:
            px = do(subj); print("%s: Calib_world + 4 vidéos OK (reproj médian %.1f px)" % (subj, px))
        except Exception as e:
            print("%s FAILED: %s" % (subj, e))


if __name__ == "__main__":
    main()
