"""
Build an extended arm26 variant for upper-limb loaded motion studies.

Takes stock arm26 (2 DOF, 6 muscles) and ADDS:
    - 4 shoulder Hill-type muscles:
                DELT_ant, DELT_post, PECT, LAT
    - 1 forearm/elbow-crossing Hill-type muscle:
        BRD_hand
    → 11 Hill-type muscles total
    - 2 CoordinateActuators ('shoulder_assist', 'elbow_assist')
        → enables assistive torque injection from Python

Reads:    Arm26/arm26.osim       (stock 6-muscle arm26)
Writes:   Arm26/arm26_paper.osim (11 muscles + 2 actuators)

Run:
        conda activate biomech
        cd C:\\Users\\21652\\Downloads\\OpenSimOverView\\Arm26
        python build_arm26_paper.py
"""

from pathlib import Path
import sys

try:
    import opensim as osim
except ImportError:
    print("ERROR: opensim Python bindings not installed.")
    print("Activate the 'biomech' env first.")
    sys.exit(1)


# -------- Configuration --------
ROOT = Path(__file__).resolve().parent
SRC_MODEL = ROOT / "arm26.osim"
OUT_MODEL = ROOT / "arm26_paper.osim"


# ============================================================
# 4 NEW shoulder Hill-type muscles (attach base ↔ r_humerus)
# Parameters loosely based on Holzbaur upper-limb model.
# ============================================================
# Each entry:
#   (name, max_iso_force_N, opt_fiber_length_m, tendon_slack_m, pennation_rad,
#    base_origin_xyz, humerus_insertion_xyz)
NEW_SHOULDER_MUSCLES = [
    {
        "name": "DELT_ant",
        "max_iso": 1142.6,
        "opt_fiber": 0.0976,
        "tendon_slack": 0.0930,
        "pennation": 0.3840,
        "base_pt":   ( 0.03, 0.02,  0.02),   # anterior, lateral to shoulder
        "humerus_pt":( 0.012, -0.12, 0.005), # mid-humeral shaft, anterior
        "role": "shoulder flexor (anterior deltoid)",
    },
    {
        "name": "DELT_post",
        "max_iso": 944.7,
        "opt_fiber": 0.1378,
        "tendon_slack": 0.0380,
        "pennation": 0.3142,
        "base_pt":   (-0.03, 0.02,  0.02),   # posterior, lateral to shoulder
        "humerus_pt":(-0.012, -0.12, 0.005), # mid-humeral shaft, posterior
        "role": "shoulder extensor (posterior deltoid)",
    },
    {
        "name": "PECT",
        "max_iso": 444.3,
        "opt_fiber": 0.1442,
        "tendon_slack": 0.0280,
        "pennation": 0.2967,
        "base_pt":   ( 0.05, -0.05,  0.06),   # anterior medial (chest)
        "humerus_pt":( 0.010, -0.08,  0.012), # proximal humerus, medial
        "role": "shoulder flexor + adductor (pec major clavicular)",
    },
    {
        "name": "LAT",
        "max_iso": 290.4,
        "opt_fiber": 0.2542,
        "tendon_slack": 0.0750,
        "pennation": 0.4363,
        "base_pt":   (-0.06, -0.08,  0.05),   # posterior medial (back)
        "humerus_pt":(-0.005, -0.07, 0.005),  # proximal humerus, posterior
        "role": "shoulder extensor + adductor (latissimus)",
    },
]


# 1 NEW forearm/elbow-crossing Hill-type muscle (attach r_humerus ↔ r_ulna_radius_hand)
# User-requested single distal muscle path to keep the hand region clean and realistic.
NEW_FOREARM_MUSCLES = [
    {
        "name": "BRD_hand",
        "max_iso": 320.0,
        "opt_fiber": 0.110,
        "tendon_slack": 0.170,
        "pennation": 0.120,
        # User-validated target: outer forearm bone side (radius/lateral),
        # with very tight bone-hugging placement.
        "humerus_pt": (0.016, -0.214, 0.000),
        # Multi-point distal path tracks near the radius outer surface with ~1-2 mm clearance.
        "forearm_path_pts": [
            (0.015, -0.020, 0.030),
            (0.016, -0.090, 0.038),
            (0.017, -0.160, 0.046),
            (0.018, -0.225, 0.052),
            (0.0185, -0.268, 0.056),
        ],
        "role": "single elbow-hand flexor path (tight outer radius-side bone-hugging)",
    },
]

ALL_NEW_MUSCLES = NEW_SHOULDER_MUSCLES + NEW_FOREARM_MUSCLES


def main():
    if not SRC_MODEL.exists():
        print(f"ERROR: source model not found at:\n  {SRC_MODEL}")
        return 1

    print(f"Loading: {SRC_MODEL}")
    model = osim.Model(str(SRC_MODEL))
    state = model.initSystem()

    # Cache references
    base    = model.getBodySet().get("base")
    humerus = model.getBodySet().get("r_humerus")
    forearm = model.getBodySet().get("r_ulna_radius_hand")

    # ---------------- 1. Existing muscles report ----------------
    existing_muscles = model.getMuscles()
    existing_names = [existing_muscles.get(i).getName() for i in range(existing_muscles.getSize())]
    print(f"\nExisting muscles ({len(existing_names)}): {existing_names}")

    # ---------------- 2. Add new Hill-type muscles ----------------
    print(f"\nAdding {len(ALL_NEW_MUSCLES)} new Hill-type muscles (Thelen2003Muscle):")
    for spec in ALL_NEW_MUSCLES:
        if spec["name"] in existing_names:
            print(f"  - {spec['name']}: already present, skipping")
            continue

        muscle = osim.Thelen2003Muscle(
            spec["name"],
            spec["max_iso"],
            spec["opt_fiber"],
            spec["tendon_slack"],
            spec["pennation"],
        )

        # Optional but recommended: realistic activation/deactivation time constants
        muscle.set_activation_time_constant(0.015)
        muscle.set_deactivation_time_constant(0.050)

        # Path: 2 points.
        # Shoulder additions use base -> humerus.
        # Forearm additions use humerus -> forearm/hand segment.
        if "base_pt" in spec:
            muscle.addNewPathPoint(
                f"{spec['name']}-P1", base,
                osim.Vec3(*spec["base_pt"])
            )
            muscle.addNewPathPoint(
                f"{spec['name']}-P2", humerus,
                osim.Vec3(*spec["humerus_pt"])
            )
        else:
            muscle.addNewPathPoint(
                f"{spec['name']}-P1", humerus,
                osim.Vec3(*spec["humerus_pt"])
            )
            forearm_pts = spec.get("forearm_path_pts")
            if forearm_pts:
                for idx, point in enumerate(forearm_pts, start=2):
                    muscle.addNewPathPoint(
                        f"{spec['name']}-P{idx}", forearm,
                        osim.Vec3(*point)
                    )
            else:
                muscle.addNewPathPoint(
                    f"{spec['name']}-P2", forearm,
                    osim.Vec3(*spec["forearm_pt"])
                )

        model.addForce(muscle)
        print(f"  + {spec['name']:9s}  ({spec['role']})")
        print(f"      max_iso = {spec['max_iso']:.1f} N, "
              f"opt_fiber = {spec['opt_fiber']*100:.1f} cm, "
              f"pennation = {spec['pennation']*57.3:.1f}°")

    # ---------------- 3. Add CoordinateActuators (assistive torques) ----------------
    print("\nAdding assistive CoordinateActuators:")
    forces = model.getForceSet()
    existing_force_names = {forces.get(i).getName() for i in range(forces.getSize())}

    if "shoulder_assist" not in existing_force_names:
        sh_act = osim.CoordinateActuator("r_shoulder_elev")
        sh_act.setName("shoulder_assist")
        sh_act.setOptimalForce(50.0)
        sh_act.setMinControl(-1.0)
        sh_act.setMaxControl(1.0)
        model.addForce(sh_act)
        print("  + shoulder_assist  (optimalForce = 50 N·m)")

    if "elbow_assist" not in existing_force_names:
        el_act = osim.CoordinateActuator("r_elbow_flex")
        el_act.setName("elbow_assist")
        el_act.setOptimalForce(30.0)
        el_act.setMinControl(-1.0)
        el_act.setMaxControl(1.0)
        model.addForce(el_act)
        print("  + elbow_assist     (optimalForce = 30 N·m)")

    # ---------------- 4. Save ----------------
    model.finalizeFromProperties()
    model.finalizeConnections()   # required so path-point sockets are written
    model.printToXML(str(OUT_MODEL))
    print(f"\nSaved: {OUT_MODEL}")

    # ---------------- 5. Verify ----------------
    print("\nVerification — reloading the saved model:")
    test = osim.Model(str(OUT_MODEL))
    test_state = test.initSystem()

    n_coords    = test.getCoordinateSet().getSize()
    n_bodies    = test.getBodySet().getSize()
    n_muscles   = test.getMuscles().getSize()
    n_forces    = test.getForceSet().getSize()
    print(f"  coordinates: {n_coords}")
    print(f"  bodies:      {n_bodies}")
    expected_muscles = 11
    expected_forces = expected_muscles + 2
    print(f"  muscles:     {n_muscles}     (expected {expected_muscles})")
    print(f"  total forces (muscles + actuators): {n_forces}  (expected {expected_forces})")

    final_muscles = [test.getMuscles().get(i).getName()
                     for i in range(test.getMuscles().getSize())]
    print(f"  muscle names: {final_muscles}")

    print(f"\n  q0_shoulder = {test.getCoordinateSet().get('r_shoulder_elev').getValue(test_state):.3f} rad")
    print(f"  q0_elbow    = {test.getCoordinateSet().get('r_elbow_flex').getValue(test_state):.3f} rad")

    if n_muscles == expected_muscles:
        print(f"\n[OK] SUCCESS - extended {expected_muscles}-muscle arm26 ready.")
        print(f"     Open in OpenSim Creator: {OUT_MODEL}")
    else:
        print(f"\n[!!] WARNING - expected {expected_muscles} muscles, got {n_muscles}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
