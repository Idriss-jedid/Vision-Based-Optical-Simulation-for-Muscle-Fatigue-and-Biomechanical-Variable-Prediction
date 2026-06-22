# pyright: reportMissingImports=false
"""
Ajoute les features ANTHROPOMÉTRIQUES (constantes par sujet) au dataset ML A.
Lues directement du modèle scalé arm26_<subj>_scaled.osim (pas besoin de relancer OpenSim) :
  humerus_mass, forearm_mass  (kg)  — <Body><mass>
  humerus_len,  forearm_len   (m)   — norme des translations des joints r_elbow / load_weld
Usage :
  python add_morpho.py            -> affiche la table morpho des 8 sujets (dry-run, n'écrit rien)
  python add_morpho.py --apply    -> ajoute les colonnes à batch/ml_dataset_A.csv (+ chaque labels_ml.csv)
"""
import os, sys, math
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
BATCH = os.path.join(ROOT, "batch")
SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
MORPHO_COLS = ["humerus_mass", "forearm_mass", "humerus_len", "forearm_len"]


def _norm(txt):
    v = [float(x) for x in txt.split()]
    return math.sqrt(sum(c * c for c in v))


def morpho(osim):
    root = ET.parse(osim).getroot()
    mass = {}
    for b in root.iter("Body"):
        m = b.find("mass")
        if m is not None:
            mass[b.get("name")] = float(m.text)
    # longueur humérus : frame r_humerus_offset du joint r_elbow
    hum_len = fore_len = None
    for j in root.iter("CustomJoint"):
        if j.get("name") == "r_elbow":
            for f in j.iter("PhysicalOffsetFrame"):
                if f.get("name") == "r_humerus_offset":
                    hum_len = _norm(f.find("translation").text)
    for j in root.iter("WeldJoint"):
        if j.get("name") == "load_weld":
            for f in j.iter("PhysicalOffsetFrame"):
                if f.get("name") == "r_ulna_radius_hand_offset":
                    fore_len = _norm(f.find("translation").text)
    return {"humerus_mass": mass.get("r_humerus", float("nan")),
            "forearm_mass": mass.get("r_ulna_radius_hand", float("nan")),
            "humerus_len": hum_len, "forearm_len": fore_len}


def main():
    table = {}
    print("%-5s %12s %12s %11s %11s" % ("subj", "hum_mass(kg)", "fore_mass(kg)", "hum_len(m)", "fore_len(m)"))
    for s in SUBJECTS:
        osim = os.path.join(BATCH, s, "opensim", "arm26_%s_scaled.osim" % s)
        if not os.path.exists(osim):
            print("%-5s  (pas de .osim)" % s); continue
        m = morpho(osim); table[s] = m
        print("%-5s %12.3f %12.3f %11.4f %11.4f" % (s, m["humerus_mass"], m["forearm_mass"], m["humerus_len"], m["forearm_len"]))

    if "--apply" not in sys.argv:
        print("\n(dry-run) relancer avec --apply pour ajouter les colonnes au dataset.")
        return

    # ajoute les colonnes à chaque labels_ml.csv + au ml_dataset_A.csv concaténé
    def augment(path):
        L = open(path).read().splitlines()
        hdr = L[0].split(","); si = hdr.index("subj")
        if MORPHO_COLS[0] in hdr:
            return  # déjà fait
        out = [",".join(hdr + MORPHO_COLS)]
        for line in L[1:]:
            if not line.strip():
                continue
            cells = line.split(","); s = cells[si]; m = table.get(s)
            if m is None:
                out.append(line); continue
            out.append(",".join(cells + ["%.5f" % m[c] for c in MORPHO_COLS]))
        open(path, "w", newline="\n").write("\n".join(out) + "\n")

    for s in SUBJECTS:
        p = os.path.join(BATCH, s, "labels_ml.csv")
        if os.path.exists(p):
            augment(p)
    ds = os.path.join(BATCH, "ml_dataset_A.csv")
    if os.path.exists(ds):
        augment(ds)
    print("\nAJOUT OK : colonnes %s ajoutées à ml_dataset_A.csv + labels_ml.csv" % MORPHO_COLS)


if __name__ == "__main__":
    main()
