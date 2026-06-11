"""
Build the LOADED variant of arm26_paper.osim — matches Figure 2(e) of the paper.

Takes arm26_paper.osim and ADDS:
    - A 2 kg body called 'load_2kg' (dumbbell geometry for visualization)
    - A WeldJoint attaching it at the distal hand/end-effector side
        of r_ulna_radius_hand (not at the elbow origin)

OpenSim's global gravity then acts on the 2 kg body exactly once, producing
the elbow torque described in paper §5.1:
    "External loads ... scaling the gravitational torque applied at the
     distal forearm segment."

NOTE: only the body mass is added - NO extra ConstantForce. Gravity is
applied automatically by OpenSim (g = -9.81 m/s² in world Y direction).

Reads:    Arm26/arm26_paper.osim
Writes:   Arm26/arm26_paper_loaded.osim

Run:
    conda activate biomech
    cd C:\\Users\\21652\\Downloads\\OpenSimOverView\\Arm26
    python build_arm26_paper_loaded.py
"""

from pathlib import Path
import sys
import math

try:
    import opensim as osim
except ImportError:
    print("ERROR: opensim Python bindings not installed.")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
SRC_MODEL = ROOT / "arm26_paper.osim"
OUT_MODEL = ROOT / "arm26_paper_loaded.osim"

LOAD_MASS_KG    = 2.0           # paper §5.1: 0-2 kg, use the maximum
# In arm26, r_ulna_radius_hand frame origin is at the elbow joint.
# Distal hand placement needs a large negative Y offset from that origin.
# Mesh-informed grip region tuned from hand/finger bbox to look hand-held.
# Combined hand bbox center ~= (0.0177, -0.2056, 0.0629)
# Finger bbox center ~= (0.0325, -0.3452, 0.0922)
LOAD_OFFSET_X   = 0.028
LOAD_OFFSET_Y   = -0.305
LOAD_OFFSET_Z   = 0.082

# Keep the dumbbell crosswise and add slight pitch/yaw for a natural grip look.
LOAD_ROT_X_RAD  = -0.30
LOAD_ROT_Y_RAD  = 0.22
LOAD_ROT_Z_RAD  = math.pi / 2.0

# Dumbbell visual dimensions (for display only; mass is set by LOAD_MASS_KG)
DUMBBELL_HANDLE_RADIUS = 0.010  # 1.0 cm
DUMBBELL_HANDLE_HALF   = 0.055  # half-length of cylinder (11 cm total)
DUMBBELL_WEIGHT_RADIUS = 0.035  # 3.5 cm end weights
DUMBBELL_WEIGHT_SPAN   = 0.065  # centers at +/- 6.5 cm along handle axis


def main():
    if not SRC_MODEL.exists():
        print(f"ERROR: source model not found at {SRC_MODEL}")
        print("Run build_arm26_paper.py first.")
        return 1

    print(f"Loading: {SRC_MODEL}")
    model = osim.Model(str(SRC_MODEL))

    # ---------------- 1. Create the load body ----------------
    print(f"\nAdding load body: {LOAD_MASS_KG} kg dumbbell...")
    load_body = osim.Body(
        "load_2kg",
        LOAD_MASS_KG,
        osim.Vec3(0.0, 0.0, 0.0),                # mass center at body origin
        osim.Inertia(0.001, 0.001, 0.001),        # small inertia (point-like)
    )

    # Build a dumbbell shape: one handle cylinder + two spherical end weights.
    # Geometry is visual only; physical mass remains exactly LOAD_MASS_KG.
    handle_frame = osim.PhysicalOffsetFrame()
    handle_frame.setName("dumbbell_handle_frame")
    handle_frame.setParentFrame(load_body)
    handle_frame.set_translation(osim.Vec3(0.0, 0.0, 0.0))
    handle_frame.set_orientation(osim.Vec3(0.0, 0.0, 0.0))

    left_frame = osim.PhysicalOffsetFrame()
    left_frame.setName("dumbbell_left_frame")
    left_frame.setParentFrame(load_body)
    left_frame.set_translation(osim.Vec3(0.0, -DUMBBELL_WEIGHT_SPAN, 0.0))
    left_frame.set_orientation(osim.Vec3(0.0, 0.0, 0.0))

    right_frame = osim.PhysicalOffsetFrame()
    right_frame.setName("dumbbell_right_frame")
    right_frame.setParentFrame(load_body)
    right_frame.set_translation(osim.Vec3(0.0, DUMBBELL_WEIGHT_SPAN, 0.0))
    right_frame.set_orientation(osim.Vec3(0.0, 0.0, 0.0))

    load_body.addComponent(handle_frame)
    load_body.addComponent(left_frame)
    load_body.addComponent(right_frame)

    handle = osim.Cylinder(DUMBBELL_HANDLE_RADIUS, DUMBBELL_HANDLE_HALF)
    handle.setColor(osim.Vec3(0.20, 0.20, 0.20))
    handle_frame.attachGeometry(handle)

    left_weight = osim.Sphere(DUMBBELL_WEIGHT_RADIUS)
    left_weight.setColor(osim.Vec3(0.45, 0.45, 0.45))
    left_frame.attachGeometry(left_weight)

    right_weight = osim.Sphere(DUMBBELL_WEIGHT_RADIUS)
    right_weight.setColor(osim.Vec3(0.45, 0.45, 0.45))
    right_frame.attachGeometry(right_weight)

    model.addBody(load_body)
    print(
        "  + load_2kg  "
        f"(mass={LOAD_MASS_KG} kg, handle={2*DUMBBELL_HANDLE_HALF*100:.1f} cm, "
        f"plate radius={DUMBBELL_WEIGHT_RADIUS*100:.1f} cm)"
    )

    # ---------------- 2. Weld it to the distal hand/end-effector ----------------
    print("\nAttaching load body to distal hand via WeldJoint...")
    forearm = model.getBodySet().get("r_ulna_radius_hand")
    weld = osim.WeldJoint(
        "load_weld",
        forearm,
        osim.Vec3(LOAD_OFFSET_X, LOAD_OFFSET_Y, LOAD_OFFSET_Z),  # offset in parent frame
        osim.Vec3(LOAD_ROT_X_RAD, LOAD_ROT_Y_RAD, LOAD_ROT_Z_RAD),  # parent orientation
        load_body,
        osim.Vec3(0.0, 0.0, 0.0),                                # child offset
        osim.Vec3(0.0, 0.0, 0.0),                                # child orientation
    )
    model.addJoint(weld)
    print(f"  + load_weld  (parent=r_ulna_radius_hand, child=load_2kg)")
    print(f"  + offset    = ({LOAD_OFFSET_X}, {LOAD_OFFSET_Y}, {LOAD_OFFSET_Z}) m")
    print(
        "  + rotation  = "
        f"({LOAD_ROT_X_RAD:.4f}, {LOAD_ROT_Y_RAD:.4f}, {LOAD_ROT_Z_RAD:.4f}) rad"
    )

    # ---------------- 3. Save ----------------
    model.finalizeFromProperties()
    model.finalizeConnections()
    model.printToXML(str(OUT_MODEL))
    print(f"\nSaved: {OUT_MODEL}")

    # ---------------- 4. Verify ----------------
    print("\nVerification - reloading the saved model:")
    test = osim.Model(str(OUT_MODEL))
    state = test.initSystem()

    print(f"  coordinates: {test.getCoordinateSet().getSize()}")
    print(f"  bodies:      {test.getBodySet().getSize()}")
    print(f"  joints:      {test.getJointSet().getSize()}")
    print(f"  muscles:     {test.getMuscles().getSize()}     (inherited from arm26_paper.osim)")

    body_list = [test.getBodySet().get(i).getName()
                 for i in range(test.getBodySet().getSize())]
    print(f"  bodies     : {body_list}")

    if "load_2kg" in body_list:
        print("\n[OK] SUCCESS - 2 kg load body attached to distal hand/end-effector side.")
        print("     Open in OpenSim Creator: " + str(OUT_MODEL))
        print("     Set r_shoulder_elev=60°, r_elbow_flex=80° to reproduce Figure 2(e).")
    else:
        print("\n[!!] WARNING - load body not in final model.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
