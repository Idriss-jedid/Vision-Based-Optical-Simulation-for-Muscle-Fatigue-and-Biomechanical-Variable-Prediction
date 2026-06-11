"""
Deep analysis of the arm26 muscles when driven by the Stage-1 motion file.

For every muscle it reconstructs the musculotendon path (in the humerus frame,
using the shoulder & elbow CustomJoint kinematics) and, over the elbow trajectory
in the .mot, reports:
    - elbow moment arm  r = -dL/dtheta   (sign: + = flexor, - = extensor)
    - MTU length range and normalised fiber length range (vs optimal_fiber_length)
    - a flexor/extensor classification and validity flags.

Also computes the gravitational elbow torque produced by the welded dumbbell
over the motion (the actual load the flexors must overcome).

Straight-segment moment arms are EXACT for muscles without wrap objects at the
elbow (the flexors BIClong/BICshort/BRA/BRD); for the triceps (which wrap the
'TRI' cylinder) they are approximate and flagged.

No OpenSim dependency: uses the virtual-work identity r = -dL/dtheta.
Usage: python analyze_muscles_over_motion.py [model.osim] [motion.mot]
"""
import math, os, sys, xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DEFAULT = os.path.join(HERE, "..", "Model", "arm26_paper_loaded_brd_elbow_research.osim")
MOTION_DEFAULT = os.path.join(HERE, "..", "Data", "paper_minjerk_fatigue_10cycles.mot")

MODEL  = sys.argv[1] if len(sys.argv) > 1 else MODEL_DEFAULT
MOTION = sys.argv[2] if len(sys.argv) > 2 else MOTION_DEFAULT
SHOULDER_DEG = 20.0          # r_shoulder_elev held constant in the .mot
WRAP_AT_ELBOW = {"TRIlong", "TRIlat", "TRImed"}   # straight-line approximate


def vec(t): return tuple(float(x) for x in t.split())
def norm(a):
    n = math.sqrt(sum(x*x for x in a)); return tuple(x/n for x in a)
def rod(a, th, v):
    ax, ay, az = a; vx, vy, vz = v; c, s = math.cos(th), math.sin(th)
    dot = ax*vx + ay*vy + az*vz
    cx, cy, cz = ay*vz-az*vy, az*vx-ax*vz, ax*vy-ay*vx
    return (vx*c+cx*s+ax*dot*(1-c), vy*c+cy*s+ay*dot*(1-c), vz*c+cz*s+az*dot*(1-c))
def sub(p, q): return (p[0]-q[0], p[1]-q[1], p[2]-q[2])
def add(p, q): return (p[0]+q[0], p[1]+q[1], p[2]+q[2])
def dist(p, q): return math.sqrt(sum((p[i]-q[i])**2 for i in range(3)))


def parse_joint(root, name, parent_off_name):
    axis = off = None
    for j in root.iter("CustomJoint"):
        if j.get("name") != name: continue
        for ta in j.iter("TransformAxis"):
            if ta.get("name") == "rotation1": axis = norm(vec(ta.find("axis").text))
        for pof in j.iter("PhysicalOffsetFrame"):
            if pof.get("name") == parent_off_name: off = vec(pof.find("translation").text)
    return axis, off


def main():
    root = ET.parse(MODEL).getroot()
    a_s, Ts = parse_joint(root, "r_shoulder", "base_offset")
    a_e, Te = parse_joint(root, "r_elbow", "r_humerus_offset")
    ths = math.radians(SHOULDER_DEG)

    def to_humerus(body, P, the):
        if body == "r_humerus": return P
        if body == "r_ulna_radius_hand": return add(Te, rod(a_e, the, P))
        if body == "base": return rod(a_s, -ths, sub(P, Ts))   # inverse shoulder
        return P                                                # ground (unused)

    # collect muscles
    muscles = []
    for m in root.iter("Thelen2003Muscle"):
        applies_force = m.find("appliesForce")
        pts = []
        for pp in m.iter("PathPoint"):
            body = pp.find("socket_parent_frame").text.split("/")[-1]
            pts.append((body, vec(pp.find("location").text)))
        muscles.append({
            "name": m.get("name"),
            "applies_force": applies_force is None or applies_force.text.strip().lower() == "true",
            "pts": pts,
            "Lopt": float(m.find("optimal_fiber_length").text),
            "Lts": float(m.find("tendon_slack_length").text),
            "penn": float(m.find("pennation_angle_at_optimal").text),
            "fmax": float(m.find("max_isometric_force").text),
        })

    def mtu_len(mus, the):
        P = [to_humerus(b, p, the) for b, p in mus["pts"]]
        return sum(dist(P[i], P[i+1]) for i in range(len(P)-1))

    def moment_arm(mus, the):
        d = math.radians(0.1)
        return -(mtu_len(mus, the+d) - mtu_len(mus, the-d)) / (2*d)   # m/rad

    # read elbow trajectory from .mot
    el = []
    with open(MOTION) as f:
        for i, line in enumerate(f):
            if i < 7: continue
            p = line.split()
            if len(p) >= 3: el.append(float(p[2]))
    e_min, e_max = min(el), max(el)
    print("Model : %s" % MODEL)
    print("Motion: %s   (elbow %.1f..%.1f deg, shoulder %.0f deg)\n"
          % (MOTION, e_min, e_max, SHOULDER_DEG))

    grid = [d for d in range(int(e_min), int(e_max)+1, 5)]
    hdr = "%-9s %-9s %8s %8s %8s | %12s %12s %6s"
    print(hdr % ("muscle", "role", "ma@min", "ma@peak", "@deg", "fiber_norm", "MTU_cm", "flag"))
    print("-"*86)
    for mus in sorted(muscles, key=lambda x: x["name"]):
        mas = [(d, moment_arm(mus, math.radians(d))*100) for d in grid]   # cm
        # role by mean sign of elbow moment arm
        mean = sum(v for _, v in mas)/len(mas)
        if abs(mean) < 0.3: role = "shoulder"
        elif mean > 0:      role = "FLEXOR"
        else:               role = "extensor"
        peak_d, peak_v = max(mas, key=lambda x: abs(x[1]))
        ma_min = mas[0][1]
        Ls = [mtu_len(mus, math.radians(d)) for d in grid]
        fib = [(L - mus["Lts"]) / mus["Lopt"] / max(math.cos(mus["penn"]), 1e-6) for L in Ls]
        flag = ""
        if not mus["applies_force"]: flag += "DISABLED"
        elif min(fib) < 0.4 or max(fib) > 1.6: flag += "FIBER!"
        if mus["name"] in WRAP_AT_ELBOW: flag += "~wrap"
        print(hdr % (mus["name"], role, "%.2f"%ma_min, "%.2f"%peak_v, "%d"%peak_d,
                     "%.2f-%.2f"%(min(fib), max(fib)),
                     "%.1f-%.1f"%(min(Ls)*100, max(Ls)*100), flag or "ok"))

    # ---- dumbbell gravitational elbow torque over the motion ----
    print("\n--- 2 kg dumbbell gravitational elbow torque ---")
    # load body weld offset (parent = r_ulna_radius_hand) and mass
    load_off = None; load_mass = 0.0
    for j in root.iter("WeldJoint"):
        if j.get("name") == "load_weld":
            for pof in j.iter("PhysicalOffsetFrame"):
                if pof.get("name") == "r_ulna_radius_hand_offset":
                    load_off = vec(pof.find("translation").text)
    for b in root.iter("Body"):
        if b.get("name") == "load_2kg": load_mass = float(b.find("mass").text)
    # elbow axis point in humerus frame = Te; gravity acts in -Y of ground.
    # With shoulder fixed, ground Y in humerus frame:
    g_hum = rod(a_s, -ths, (0.0, -1.0, 0.0))   # gravity unit vector in humerus frame
    def elbow_torque(the):
        load_hum = add(Te, rod(a_e, the, load_off))      # dumbbell COM in humerus frame
        rvec = sub(load_hum, Te)                          # lever from elbow to load
        F = (load_mass*9.81*g_hum[0], load_mass*9.81*g_hum[1], load_mass*9.81*g_hum[2])
        # torque about elbow axis = a_e . (r x F)
        rx = (rvec[1]*F[2]-rvec[2]*F[1], rvec[2]*F[0]-rvec[0]*F[2], rvec[0]*F[1]-rvec[1]*F[0])
        return a_e[0]*rx[0]+a_e[1]*rx[1]+a_e[2]*rx[2]
    for d in grid:
        t = elbow_torque(math.radians(d))
        print("  elbow %3d deg : flexion torque %6.2f N·m  %s" % (d, t, "#"*int(abs(t)*6)))


if __name__ == "__main__":
    main()
