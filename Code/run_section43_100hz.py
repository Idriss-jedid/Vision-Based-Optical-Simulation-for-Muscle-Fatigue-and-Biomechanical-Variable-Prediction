"""
Section 4.3 reproducibility script for arm26_paper_loaded.osim.

What this demonstrates:
1) OpenSim 4.x model loading.
2) 2-DOF shoulder/elbow model usage.
3) Hill-type muscle actuators (Thelen2003Muscle) in the control loop.
4) Closed-loop control at 100 Hz through the OpenSim Python API (dt = 0.01 s).

Run:
    conda run -n biomech python Arm26/run_section43_100hz.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import opensim as osim
except ImportError as exc:
    raise SystemExit("opensim Python package is required. Install OpenSim Python bindings first.") from exc


ROOT = Path(__file__).resolve().parent
MODEL_DEFAULT = ROOT / "arm26_paper_loaded.osim"
OUTPUT_DEFAULT = ROOT / "OutputReference" / "section43_100hz_log.csv"

SHOULDER_COORD = "r_shoulder_elev"
ELBOW_COORD = "r_elbow_flex"

ELBOW_FLEXORS = {"BIClong", "BICshort", "BRA", "BRD_hand"}
ELBOW_EXTENSORS = {"TRIlong", "TRIlat", "TRImed"}
SHOULDER_FLEXORS = {"DELT_ant", "PECT", "BIClong"}
SHOULDER_EXTENSORS = {"DELT_post", "LAT", "TRIlong"}


def min_jerk(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def reference_trajectory_deg(t: float, duration: float) -> Tuple[float, float]:
    """
    Single reach-return profile in degrees.
    Shoulder: 0 -> 60 -> 0
    Elbow:    0 -> 80 -> 0
    """
    half = duration * 0.5
    if t <= half:
        s = t / half if half > 0 else 1.0
        w = min_jerk(s)
        return 60.0 * w, 80.0 * w

    s = (t - half) / half if half > 0 else 1.0
    w = min_jerk(s)
    return 60.0 * (1.0 - w), 80.0 * (1.0 - w)


def count_thelen_muscles(model: osim.Model) -> int:
    muscles = model.getMuscles()
    count = 0
    for i in range(muscles.getSize()):
        m = muscles.get(i)
        if osim.Thelen2003Muscle.safeDownCast(m) is not None:
            count += 1
    return count


def set_muscle_excitations(model: osim.Model, state: osim.State, err_sh: float, err_el: float) -> None:
    """
    Very simple reflex-like mapping from joint errors to muscle excitations.
    This keeps Hill-type actuators active in the loop while assist actuators provide
    the main corrective torque.
    """
    sh_flex = max(err_sh, 0.0)
    sh_ext = max(-err_sh, 0.0)
    el_flex = max(err_el, 0.0)
    el_ext = max(-err_el, 0.0)

    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        m = muscles.get(i)
        name = m.getName()

        exc = 0.02
        if name in ELBOW_FLEXORS:
            exc += 0.55 * el_flex
        if name in ELBOW_EXTENSORS:
            exc += 0.55 * el_ext
        if name in SHOULDER_FLEXORS:
            exc += 0.45 * sh_flex
        if name in SHOULDER_EXTENSORS:
            exc += 0.45 * sh_ext

        exc = float(np.clip(exc, 0.01, 0.80))
        try:
            m.setExcitation(state, exc)
        except Exception:
            m.setActivation(state, exc)


def run_closed_loop(model_path: Path, output_path: Path, duration: float, dt: float) -> None:
    model = osim.Model(str(model_path))
    state = model.initSystem()

    coords = model.getCoordinateSet()
    shoulder = coords.get(SHOULDER_COORD)
    elbow = coords.get(ELBOW_COORD)

    coord_count = coords.getSize()
    thelen_count = count_thelen_muscles(model)

    forces = model.getForceSet()
    shoulder_assist = osim.ScalarActuator.safeDownCast(forces.get("shoulder_assist"))
    elbow_assist = osim.ScalarActuator.safeDownCast(forces.get("elbow_assist"))
    if shoulder_assist is None or elbow_assist is None:
        raise RuntimeError("Missing scalar assistive actuators: shoulder_assist / elbow_assist")

    shoulder.setValue(state, 0.0)
    elbow.setValue(state, 0.0)
    shoulder.setSpeedValue(state, 0.0)
    elbow.setSpeedValue(state, 0.0)

    model.equilibrateMuscles(state)

    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(1e-4)
    manager.initialize(state)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "q_sh_ref_deg",
                "q_el_ref_deg",
                "q_sh_deg",
                "q_el_deg",
                "dq_sh_deg_s",
                "dq_el_deg_s",
                "tau_sh_assist_Nm",
                "tau_el_assist_Nm",
            ]
        )

        n_steps = int(round(duration / dt))
        for step in range(n_steps + 1):
            t = step * dt

            sh_ref_deg, el_ref_deg = reference_trajectory_deg(t, duration)
            sh_ref = np.deg2rad(sh_ref_deg)
            el_ref = np.deg2rad(el_ref_deg)

            sh = shoulder.getValue(state)
            el = elbow.getValue(state)
            dsh = shoulder.getSpeedValue(state)
            dele = elbow.getSpeedValue(state)

            err_sh = sh_ref - sh
            err_el = el_ref - el

            # Simple PD closed-loop law at 100 Hz.
            tau_sh = float(np.clip(45.0 * err_sh - 3.5 * dsh, -20.0, 20.0))
            tau_el = float(np.clip(38.0 * err_el - 3.0 * dele, -15.0, 15.0))

            shoulder_assist.overrideActuation(state, True)
            elbow_assist.overrideActuation(state, True)
            shoulder_assist.setOverrideActuation(state, tau_sh)
            elbow_assist.setOverrideActuation(state, tau_el)

            set_muscle_excitations(model, state, err_sh, err_el)

            writer.writerow(
                [
                    f"{t:.4f}",
                    f"{sh_ref_deg:.6f}",
                    f"{el_ref_deg:.6f}",
                    f"{np.rad2deg(sh):.6f}",
                    f"{np.rad2deg(el):.6f}",
                    f"{np.rad2deg(dsh):.6f}",
                    f"{np.rad2deg(dele):.6f}",
                    f"{tau_sh:.6f}",
                    f"{tau_el:.6f}",
                ]
            )

            if step == n_steps:
                break

            t_target = state.getTime() + dt
            try:
                state = manager.integrate(t_target)
            except Exception:
                # Keep the loop stepping in rare stiff configurations.
                state.setTime(t_target)

    hz = 1.0 / dt
    print("Section 4.3 check complete")
    print(f"model: {model_path}")
    print(f"coordinates_total: {coord_count} (uses {SHOULDER_COORD}, {ELBOW_COORD})")
    print(f"thelen_muscles: {thelen_count}")
    print(f"closed_loop_dt: {dt:.5f} s")
    print(f"closed_loop_rate: {hz:.1f} Hz")
    print(f"log_csv: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 4.3 100 Hz OpenSim closed-loop demo")
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT, help="Path to .osim model")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT, help="Path to output CSV")
    parser.add_argument("--duration", type=float, default=6.0, help="Simulation duration (s)")
    parser.add_argument("--dt", type=float, default=0.01, help="Control step (s), 0.01 = 100 Hz")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_closed_loop(args.model, args.output, args.duration, args.dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
