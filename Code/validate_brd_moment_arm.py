"""
Validate the brachioradialis (BRD) elbow-flexion moment-arm curve of an arm26
.osim model against the cadaveric data of Murray et al. (2000, 2002).

WHY THIS WORKS WITHOUT OPENSIM
------------------------------
The elbow flexion moment arm of a muscle is, by the principle of virtual work,
    r(theta) = -dL/dtheta
where L is the total musculotendon path length and theta is the elbow angle.
For BRD the origin (P1) is on the humerus and every other path point (P2..Pn)
is rigidly fixed on the forearm, so only the *crossing* segment P1->P2 changes
length with elbow angle. We therefore reconstruct the elbow CustomJoint
kinematics from the .osim (rotation axis + offset) and differentiate L(theta)
numerically. The same method reproduces the known arm26 biceps (~4.9 cm) and
brachialis (~2.4 cm) moment arms, which validates it.

TARGET (Murray 2000 Table 2 / Murray 2002 Table 3):
    peak moment arm 7.7 cm (range 7.0-9.0), at ~108 deg (range 100-118),
    rising monotonically from ~2-3 cm at 20 deg; average ~5.4-5.7 cm.

Usage:  python validate_brd_moment_arm.py [model.osim] [muscle_name]
"""

import math
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "Model", "arm26_paper_loaded_brd_elbow_research.osim")
DEFAULT_MUSCLE = "BRD_hand"


def rodrigues(axis, theta, v):
    ax, ay, az = axis
    vx, vy, vz = v
    c, s = math.cos(theta), math.sin(theta)
    dot = ax * vx + ay * vy + az * vz
    cx, cy, cz = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx
    return (vx * c + cx * s + ax * dot * (1 - c),
            vy * c + cy * s + ay * dot * (1 - c),
            vz * c + cz * s + az * dot * (1 - c))


def vec(text):
    return tuple(float(x) for x in text.split())


def load_geometry(model_path, muscle_name):
    """Return (axis, offset, P1_humerus, P2_forearm) from the .osim."""
    root = ET.parse(model_path).getroot()

    # --- elbow joint: rotation axis (rotation1) and parent offset translation ---
    axis = offset = None
    for joint in root.iter("CustomJoint"):
        if joint.get("name") != "r_elbow":
            continue
        for ta in joint.iter("TransformAxis"):
            if ta.get("name") == "rotation1":
                axis = vec(ta.find("axis").text)
        for pof in joint.iter("PhysicalOffsetFrame"):
            if pof.get("name") == "r_humerus_offset":
                offset = vec(pof.find("translation").text)
    n = math.sqrt(sum(x * x for x in axis))
    axis = tuple(x / n for x in axis)

    # --- BRD crossing segment: P1 (humerus) and P2 (first forearm point) ---
    p1 = p2 = None
    for m in root.iter("Thelen2003Muscle"):
        if m.get("name") != muscle_name:
            continue
        pts = list(m.iter("PathPoint"))
        p1 = vec(pts[0].find("location").text)   # on humerus
        p2 = vec(pts[1].find("location").text)   # first forearm point
    return axis, offset, p1, p2


def moment_arm_curve(axis, offset, p1, p2):
    def length(theta):
        r = rodrigues(axis, theta, p2)
        q = (offset[0] + r[0], offset[1] + r[1], offset[2] + r[2])
        return math.dist(p1, q)

    rows = []
    for deg in range(0, 131, 5):
        th = math.radians(deg)
        d = math.radians(0.1)
        ma = -(length(th + d) - length(th - d)) / (2 * d) * 100.0  # cm
        rows.append((deg, ma))
    return rows


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    muscle = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MUSCLE

    axis, offset, p1, p2 = load_geometry(model, muscle)
    rows = moment_arm_curve(axis, offset, p1, p2)
    peak_deg, peak_ma = max(rows, key=lambda r: r[1])
    avg = sum(ma for _, ma in rows) / len(rows)

    print("Model : %s" % model)
    print("Muscle: %s   (P1 humerus=%s, P2 forearm=%s)" % (muscle, p1, p2))
    print("\n elbow_deg   moment_arm_cm")
    for deg, ma in rows:
        bar = "#" * int(max(ma, 0) * 4)
        print("   %5d      %6.2f  %s" % (deg, ma, bar))

    print("\n PEAK %.2f cm @ %d deg   |   average %.2f cm" % (peak_ma, peak_deg, avg))
    ok_peak = 7.0 <= peak_ma <= 9.0
    ok_angle = 100 <= peak_deg <= 118
    print(" Murray target: peak 7.7 cm (7.0-9.0) @ ~108 deg (100-118), avg ~5.4-5.7 cm")
    print(" peak magnitude in range : %s" % ("PASS" if ok_peak else "FAIL"))
    print(" peak angle in range     : %s" % ("PASS" if ok_angle else "FAIL"))
    print(" VERDICT: %s" % ("VALIDATED" if (ok_peak and ok_angle) else "NEEDS TUNING"))


if __name__ == "__main__":
    main()
