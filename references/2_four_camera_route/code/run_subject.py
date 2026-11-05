# -*- coding: utf-8 -*-
"""
PIPELINE END-TO-END pour UN sujet (Fit3D) — pour ajouter un nouvel échantillon.
Pré-requis : le sujet existe dans D:\\Download\\fit3d_train\\train\\<subj>\\ avec
  camera_parameters/ , videos/ , joints3d_25/  (exercice dumbbell_biceps_curls).

Usage :  python run_subject.py s12
Enchaîne (chaque étape dans le bon env) :
  1) Pose2Sim   (pose2sim_env) : calib -R*T -> RTMPose -> triangulation -> filtering
  2) arm26      (biomech)      : scaling Vicon + motion (épaule+coude) + placement UPRIGHT
  3) Calib_world(pose2sim_env) : caméras dans le frame .trc + vidéos
  4) Geometry   : copie des .stl
Sortie : D:\\p2s_blender\\<subj>_arm26_4cam\\ (prêt pour Blender).
"""
import glob, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
P2S = r"D:\Download\pose2sim_env\Scripts\python.exe"
BIO = r"D:\miniconda3\envs\biomech\python.exe"
GEO_SRC = r"D:\p2s_blender\s04_arm26_4cam\Geometry"   # géométrie standard (601 .stl) déjà convertie


def run(py, script, label):
    print("\n========== %s ==========" % label)
    r = subprocess.run([py, os.path.join(HERE, script), subj], cwd=HERE)
    if r.returncode != 0:
        print("!! ÉCHEC à l'étape : %s" % label); sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: run_subject.py <subj>   (ex: run_subject.py s12)"); sys.exit(1)
    subj = sys.argv[1]
    run(P2S, "batch_all_subjects.py", "1/4 Pose2Sim : calib (-R*T) -> RTMPose -> triangulation -> filtering")
    run(BIO, "batch_arm26_all.py",    "2/4 arm26 : scaling Vicon + motion (epaule+coude) + placement UPRIGHT")
    run(P2S, "batch_calib_all.py",    "3/4 Calib_world (frame .trc) + videos")
    dst = os.path.join("D:\\", "p2s_blender", "%s_arm26_4cam" % subj, "Geometry")
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(GEO_SRC, "*.stl")):
        d = os.path.join(dst, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(d):   # évite SameFileError si subj == source géométrie
            shutil.copy(f, d)
    print("4/4 Geometry copiée (%d .stl)" % len(glob.glob(os.path.join(dst, "*.stl"))))
    out = os.path.join("D:\\", "p2s_blender", "%s_arm26_4cam" % subj)
    print("\n=================================================================")
    print("OK -> %s" % out)
    print("Fichiers : model.osim, motion_world.csv (UPRIGHT), arm_markers_world.trc,")
    print("           Calib_world.toml, videos/, Geometry/")
    print("Blender  : Model -> Motion(motion_world.csv) -> Markers -> cameras(Calib_world) -> videos")
    print("=================================================================")
