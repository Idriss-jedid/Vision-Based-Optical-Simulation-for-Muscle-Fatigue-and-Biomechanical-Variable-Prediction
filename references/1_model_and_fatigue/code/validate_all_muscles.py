# pyright: reportMissingImports=false
"""
Validate EVERY muscle's elbow moment arm against cadaveric literature
=====================================================================
Not just BRD. Uses OpenSim's wrap-aware computeMomentArm over the elbow range
(shoulder fixed at 20 deg) and compares each muscle's PEAK and MEAN flexion
moment arm to Murray et al. (2000, 2002) / Holzbaur et al. (2005).

Murray averaging windows: flexors 20-120 deg, triceps 30-120 deg.
Sign convention here: + = flexion moment arm, - = extension.

Run:  conda run -n biomech python validate_all_muscles.py [model.osim]
"""
import os
import sys
import math
import opensim as osim

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "Model", "arm26_paper_loaded_brd_elbow_research.osim")
SHOULDER_DEG = 20.0

# Literature peak |moment arm| (cm) and plausible range, elbow flexion(+)/ext(-)
# Murray 2000/2002 (cadaver) and Holzbaur 2005 (model).
LIT = {
    "BIClong":  dict(role="flexor",   peak=4.7, rng=(3.5, 5.2), src="Murray2000 biceps-long 4.7"),
    "BICshort": dict(role="flexor",   peak=4.2, rng=(3.0, 5.0), src="Murray/Holzbaur biceps ~3.5-4.7"),
    "BRA":      dict(role="flexor",   peak=1.9, rng=(1.5, 2.6), src="Murray2000 brachialis ~1.9"),
    "BRD_hand": dict(role="flexor",   peak=7.7, rng=(7.0, 9.0), src="Murray2000/2002 BRD 7.7 (7-9)"),
    "TRIlong":  dict(role="extensor", peak=2.3, rng=(1.8, 2.8), src="Murray2000 triceps-long 2.3"),
    "TRIlat":   dict(role="extensor", peak=2.1, rng=(1.6, 2.6), src="Murray triceps ~2.0-2.3"),
    "TRImed":   dict(role="extensor", peak=2.1, rng=(1.6, 2.6), src="Murray triceps ~2.0-2.3"),
}


def main():
    model = osim.Model(MODEL)
    state = model.initSystem()
    elbow = model.getCoordinateSet().get("r_elbow_flex")
    model.getCoordinateSet().get("r_shoulder_elev").setValue(state, math.radians(SHOULDER_DEG))
    muscles = model.getMuscles()
    names = [muscles.get(i).getName() for i in range(muscles.getSize())]

    angles = list(range(0, 131, 5))
    ma = {n: [] for n in names}
    for deg in angles:
        elbow.setValue(state, math.radians(deg))
        model.assemble(state)
        model.realizePosition(state)
        for i in range(muscles.getSize()):
            m = muscles.get(i)
            ma[m.getName()].append(m.getGeometryPath().computeMomentArm(state, elbow) * 100.0)

    print("Model: %s   (shoulder fixed %g deg)\n" % (model.getName(), SHOULDER_DEG))
    print("%-10s %-9s %8s %6s %9s %8s %14s %s"
          % ("muscle", "role", "peak_cm", "@deg", "mean_cm", "lit_peak", "lit_range", "verdict"))
    print("-" * 92)
    n_elbow = n_pass = 0
    for n in names:
        arr = ma[n]
        mean_abs = sum(abs(v) for v in arr) / len(arr)
        if mean_abs < 0.3:
            print("%-10s %-9s %8.2f %6s %9s %8s %14s   shoulder DOF (no elbow action)"
                  % (n, "shoulder", max(arr, key=abs), "-", "-", "-", "-"))
            continue
        n_elbow += 1
        signed_mean = sum(arr) / len(arr)
        role = "flexor" if signed_mean > 0 else "extensor"
        pk_i = max(range(len(arr)), key=lambda k: abs(arr[k]))
        peak = abs(arr[pk_i]); peak_deg = angles[pk_i]
        # Murray window mean
        lo = 20 if role == "flexor" else 30
        win = [abs(arr[k]) for k, d in enumerate(angles) if lo <= d <= 120]
        meanw = sum(win) / len(win)
        lit = LIT.get(n)
        if lit:
            ok = lit["rng"][0] <= peak <= lit["rng"][1]
            n_pass += int(ok)
            verdict = ("PASS" if ok else "CHECK") + "  vs " + lit["src"]
            litpk = "%.1f" % lit["peak"]; litrng = "%.1f-%.1f" % lit["rng"]
        else:
            verdict = "no lit ref"; litpk = litrng = "-"
        print("%-10s %-9s %8.2f %6d %9.2f %8s %14s   %s"
              % (n, role, peak, peak_deg, meanw, litpk, litrng, verdict))

    print("\nElbow muscles validated: %d/%d within published range." % (n_pass, n_elbow))
    print("(Sign: + flexion, - extension; peak |moment arm| compared to Murray 2000/2002.)")


if __name__ == "__main__":
    main()
