# pyright: reportMissingImports=false
"""Comprendre la convention de r_shoulder_elev dans arm26 AVANT de l'extraire des
données : pour une grille d'angles, on calcule la direction du bras (acromion->
epicondyle) et son élévation par rapport à la verticale (bas = -Y). On veut savoir
si r_shoulder_elev == élévation du bras (1:1) ou s'il faut une table de correspondance.
biomech env."""
import os
import numpy as np
import opensim as osim

HERE = os.path.dirname(os.path.abspath(__file__))
S04 = os.path.dirname(HERE)
MODEL = os.path.join(S04, "build4", "opensim", "arm26_s04_scaled.osim")

m = osim.Model(MODEL); s = m.initSystem()
csh = m.getCoordinateSet().get("r_shoulder_elev")
cel = m.getCoordinateSet().get("r_elbow_flex")
cel.setValue(s, np.radians(20.0), False)

def mk(n):
    loc = m.getMarkerSet().get(n).getLocationInGround(s)
    return np.array([loc.get(0), loc.get(1), loc.get(2)])

down = np.array([0.0, -1.0, 0.0])
print("r_shoulder_elev  ->  bras(acromion->epicondyle): élévation vs bas(-Y), et composantes")
print("-" * 78)
for sh in [-30, 0, 20, 30, 45, 60, 90, 120]:
    csh.setValue(s, np.radians(sh)); m.realizePosition(s)
    ua = mk("r_humerus_epicondyle") - mk("r_acromion")
    uan = ua / (np.linalg.norm(ua) + 1e-9)
    elev = np.degrees(np.arccos(np.clip(uan @ down, -1, 1)))
    print("  %4d°  ->  élévation %6.1f°   dir=[% .2f % .2f % .2f]" % (sh, elev, uan[0], uan[1], uan[2]))
