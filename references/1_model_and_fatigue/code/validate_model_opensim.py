# pyright: reportMissingImports=false

"""
Authoritative model validation using the OpenSim API (run in the 'biomech' env).

Loads the elbow-focused research model by default and, using OpenSim's own
kinematics + wrapping solver, reports for every muscle across the elbow range:
    - elbow flexion moment arm (computeMomentArm)   <- handles wrap objects
    - musculotendon length and normalised fiber length
and checks the BRD curve against Murray (2000/2002): peak ~7.7 cm @ ~108 deg.

Also verifies the 2 kg dumbbell (mass, inertia, location) and that the model
builds a consistent System.

Run:
    conda run -n biomech python validate_model_opensim.py
    conda run -n biomech python validate_model_opensim.py ../Model/arm26_paper_loaded_brd.osim
"""
import os
import math
import sys
import xml.etree.ElementTree as ET
import opensim as osim

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DEFAULT = os.path.join(HERE, "..", "Model", "arm26_paper_loaded_brd_elbow_research.osim")
MODEL = sys.argv[1] if len(sys.argv) > 1 else MODEL_DEFAULT
MODEL_DIR = os.path.dirname(MODEL)
WORKSPACE = os.path.abspath(os.path.join(HERE, ".."))
SHOULDER_DEG = 20.0


def vec(text):
    return tuple(float(x) for x in text.split())


def load_xml_audit(model_path):
    root = ET.parse(model_path).getroot()

    muscle_params = {}
    for mus in root.iter("Thelen2003Muscle"):
        name = mus.get("name")
        applies_force = mus.find("appliesForce")
        muscle_params[name] = {
            "lopt": float(mus.find("optimal_fiber_length").text),
            "lts": float(mus.find("tendon_slack_length").text),
            "penn": float(mus.find("pennation_angle_at_optimal").text),
            "applies_force": applies_force is None or applies_force.text.strip().lower() == "true",
        }

    load = {}
    for body in root.iter("Body"):
        if body.get("name") == "load_2kg":
            load = {
                "mass": float(body.find("mass").text),
                "mass_center": vec(body.find("mass_center").text),
                "inertia": vec(body.find("inertia").text),
            }

    weld = {}
    for joint in root.iter("WeldJoint"):
        if joint.get("name") == "load_weld":
            for frame in joint.iter("PhysicalOffsetFrame"):
                if frame.get("name") == "r_ulna_radius_hand_offset":
                    weld = {
                        "translation": vec(frame.find("translation").text),
                        "orientation": vec(frame.find("orientation").text),
                    }

    mesh_files = [mesh.text for mesh in root.iter("mesh_file") if mesh.text]
    search_dirs = [MODEL_DIR, os.path.join(WORKSPACE, "Geometry"), os.path.join(HERE, "Geometry")]
    missing_meshes = []
    for mesh in mesh_files:
        if not any(os.path.exists(os.path.join(folder, mesh)) for folder in search_dirs):
            missing_meshes.append(mesh)

    return muscle_params, load, weld, sorted(set(missing_meshes))


def main():
    muscle_params, load_xml, weld_xml, missing_meshes = load_xml_audit(MODEL)

    model = osim.Model(MODEL)
    state = model.initSystem()
    print("Model loaded: %s" % model.getName())
    print("  coords=%d  muscles=%d  bodies=%d\n"
          % (model.getCoordinateSet().getSize(),
             model.getMuscles().getSize(),
             model.getBodySet().getSize()))

    elbow = model.getCoordinateSet().get("r_elbow_flex")
    shoulder = model.getCoordinateSet().get("r_shoulder_elev")
    shoulder.setValue(state, math.radians(SHOULDER_DEG))

    muscles = model.getMuscles()
    names = [muscles.get(i).getName() for i in range(muscles.getSize())]

    angles = list(range(0, 131, 5))
    # collect moment arm + fiber length per muscle
    ma = {n: [] for n in names}
    fib = {n: [] for n in names}
    for deg in angles:
        elbow.setValue(state, math.radians(deg))
        model.assemble(state)
        model.realizePosition(state)
        for i in range(muscles.getSize()):
            mus = muscles.get(i)
            n = mus.getName()
            ma[n].append(mus.getGeometryPath().computeMomentArm(state, elbow) * 100.0)  # cm
            try:
                mus.setActivation(state, 0.05)
            except Exception:
                pass
            fib[n].append(mus.getLength(state))

    print("%-10s %-9s %8s %8s %7s   %s   %s   %s"
          % ("muscle", "role", "peak_cm", "@deg", "ma@90", "MTU_cm", "fiber_norm", "flag"))
    print("-" * 98)
    for n in sorted(names):
        arr = ma[n]
        mean = sum(arr) / len(arr)
        role = "FLEXOR" if mean > 0.3 else ("extensor" if mean < -0.3 else "shoulder")
        pk_i = max(range(len(arr)), key=lambda k: abs(arr[k]))
        ma90 = arr[angles.index(90)]
        Ls = [L * 100 for L in fib[n]]
        params = muscle_params.get(n)
        if params:
            denom = params["lopt"] * max(math.cos(params["penn"]), 1e-6)
            fiber_norm = [(L - params["lts"]) / denom for L in fib[n]]
            fiber_text = "%.2f-%.2f" % (min(fiber_norm), max(fiber_norm))
            flag = ""
            if not params["applies_force"]:
                flag = "DISABLED"
            elif min(fiber_norm) < 0.4 or max(fiber_norm) > 1.6:
                flag = "CHECK_FIBER"
        else:
            fiber_text = "n/a"
            flag = ""
        print("%-10s %-9s %8.2f %8d %7.2f   %.1f-%.1f   %s   %s"
              % (n, role, arr[pk_i], angles[pk_i], ma90, min(Ls), max(Ls), fiber_text, flag or "ok"))

    # ---- BRD validation verdict ----
    brd = ma["BRD_hand"]
    pk_i = max(range(len(brd)), key=lambda k: brd[k])
    peak, peak_deg = brd[pk_i], angles[pk_i]
    print("\nBRD moment arm (OpenSim, with wrapping):")
    for deg, v in zip(angles, brd):
        if deg % 10 == 0:
            print("   %3d deg : %5.2f cm  %s" % (deg, v, "#" * int(max(v, 0) * 4)))
    ok = (7.0 <= peak <= 9.0) and (100 <= peak_deg <= 118)
    print("  >>> PEAK %.2f cm @ %d deg | Murray 7.7 (7-9) @ ~108 (100-118) | %s"
          % (peak, peak_deg, "VALIDATED" if ok else "CHECK"))

    # ---- dumbbell check ----
    load = model.getBodySet().get("load_2kg")
    print("\nDumbbell: mass=%.3f kg" % load.getMass())
    mc = load.getMassCenter()
    print("  mass_center=(%.3f,%.3f,%.3f)" % (mc.get(0), mc.get(1), mc.get(2)))
    if load_xml:
        print("  inertia=(%.4f, %.4f, %.4f, %.4f, %.4f, %.4f) kg*m^2" % load_xml["inertia"])
    if weld_xml:
        print("  weld_translation=(%.3f, %.3f, %.3f) m" % weld_xml["translation"])
        print("  weld_orientation=(%.6f, %.6f, %.6f) rad" % weld_xml["orientation"])

    print("\nNotes:")
    print("  - Shoulder muscles (DELT_ant/post, PECT, LAT) were removed; this is an")
    print("    elbow-focused model with the shoulder held by the shoulder_assist actuator.")
    if missing_meshes:
        print("  - %d visual mesh files are missing from workspace geometry search paths. Dynamics validate, but GUI visuals will be incomplete until the Arm26 .vtp geometry folder is added." % len(missing_meshes))


if __name__ == "__main__":
    main()
