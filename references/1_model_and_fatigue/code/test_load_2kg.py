"""
Verifies the paper §5.1 2 kg load mechanism on arm26_paper.osim variants.

Compares two short forward simulations side by side:
  1) arm26_paper.osim         — no load
  2) arm26_paper_loaded.osim  — 2 kg body welded to forearm (Figure 2e)

Starts from shoulder=0°, elbow=90° (max gravity moment arm). Runs 0.3s of
passive forward dynamics (all muscles at minimum activation). The 2 kg load
should produce an additional elbow extension torque visible as more drop.

Run:
    conda activate biomech
    cd C:\\Users\\21652\\Downloads\\OpenSimOverView\\Arm26
    python test_load_2kg.py
"""

from pathlib import Path
import sys
import numpy as np

try:
    import opensim as osim
except ImportError:
    print("ERROR: opensim Python bindings not installed.")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
MODEL_NO_LOAD = ROOT / "arm26_paper.osim"
MODEL_LOADED  = ROOT / "arm26_paper_loaded.osim"

SHOULDER_COORD = "r_shoulder_elev"
ELBOW_COORD    = "r_elbow_flex"


def init_plant(model_path):
    """Load model, set pose (shoulder=0°, elbow=90°), zero velocities."""
    model = osim.Model(str(model_path))
    state = model.initSystem()

    sh = model.getCoordinateSet().get(SHOULDER_COORD)
    el = model.getCoordinateSet().get(ELBOW_COORD)
    sh.setValue(state, 0.0)
    el.setValue(state, np.deg2rad(90.0))
    sh.setSpeedValue(state, 0.0)
    el.setSpeedValue(state, 0.0)

    # Set every muscle to minimum activation (passive arm)
    for i in range(model.getMuscles().getSize()):
        try:
            model.getMuscles().get(i).setActivation(state, 0.01)
        except Exception:
            pass

    model.equilibrateMuscles(state)
    return model, state


def run(model_path, duration=0.3, dt=0.005):
    """Return (time, shoulder_deg, elbow_deg) over the simulation."""
    model, state = init_plant(model_path)
    sh = model.getCoordinateSet().get(SHOULDER_COORD)
    el = model.getCoordinateSet().get(ELBOW_COORD)

    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(1e-4)
    manager.initialize(state)

    times, sh_deg, el_deg = [state.getTime()], [np.rad2deg(sh.getValue(state))], [np.rad2deg(el.getValue(state))]
    n = int(duration / dt)
    for i in range(n):
        try:
            state = manager.integrate(state.getTime() + dt)
        except Exception:
            break
        times.append(state.getTime())
        sh_deg.append(np.rad2deg(sh.getValue(state)))
        el_deg.append(np.rad2deg(el.getValue(state)))

    return np.array(times), np.array(sh_deg), np.array(el_deg)


def main():
    if not MODEL_NO_LOAD.exists():
        print(f"ERROR: {MODEL_NO_LOAD} not found. Run build_arm26_paper.py first.")
        return 1
    if not MODEL_LOADED.exists():
        print(f"ERROR: {MODEL_LOADED} not found. Run build_arm26_paper_loaded.py first.")
        return 1

    print("=" * 64)
    print("Paper §5.1 load mechanism — sanity check on arm26_paper.osim")
    print("=" * 64)
    print("Initial pose: shoulder=0°, elbow=90° (max gravity moment arm)")
    print("All muscles at minimum activation (passive arm)")
    print("Duration: 0.3 s of forward dynamics")
    print()

    print("Running NO LOAD simulation (arm26_paper.osim)...")
    t0, sh0, el0 = run(MODEL_NO_LOAD)
    print(f"  Elbow at t=0.00s : {el0[0]:7.2f} deg")
    print(f"  Elbow at t=0.15s : {el0[len(el0)//2]:7.2f} deg")
    print(f"  Elbow at t={t0[-1]:.2f}s : {el0[-1]:7.2f} deg")
    drop0 = el0[0] - el0[-1]
    print(f"  Elbow drop       : {drop0:7.2f} deg")

    print()
    print("Running 2 kg LOAD simulation (arm26_paper_loaded.osim)...")
    t1, sh1, el1 = run(MODEL_LOADED)
    print(f"  Elbow at t=0.00s : {el1[0]:7.2f} deg")
    print(f"  Elbow at t=0.15s : {el1[len(el1)//2]:7.2f} deg")
    print(f"  Elbow at t={t1[-1]:.2f}s : {el1[-1]:7.2f} deg")
    drop1 = el1[0] - el1[-1]
    print(f"  Elbow drop       : {drop1:7.2f} deg")

    print()
    print("=" * 64)
    diff = drop1 - drop0
    print(f"Extra elbow drop attributable to 2 kg load: {diff:+.2f} deg")
    if diff > 0.5:
        print()
        print("[OK] Load body adds extra extension torque at the elbow.")
        print("     The 2 kg dumbbell behaves as a physical mass under gravity.")
        print("     Mechanism matches paper §5.1 (gravitational torque at distal forearm).")
        print()
        print("Open arm26_paper_loaded.osim in OpenSim Creator to see the 2 kg sphere")
        print("attached to the hand (this reproduces Figure 2(e) of the paper).")
    else:
        print("[!!] Load didn't produce expected extra drop. Possible causes:")
        print("     - Weld offset puts the load too close to the elbow")
        print("     - Inertia effects mask the steady-state torque difference")
        print("     Try increasing simulation duration or starting pose")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
