"""
Generate Arm26 task references that match paper Section 5.2 and the provided figure.

Outputs:
  - Arm26/OutputReference/paper_scenarios.csv
      Static postures (a)-(e), including loaded 2 kg case on the hand.
  - Arm26/OutputReference/paper_minjerk_10cycles.mot
      10 repeated minimum-jerk cycles with 2 s rest between cycles.

Coordinates in this model:
  - r_shoulder_elev (deg)
  - r_elbow_flex (deg)
"""

from pathlib import Path
import numpy as np
import csv

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "OutputReference"

SCENARIO_CSV = OUT_DIR / "paper_scenarios.csv"
TRAJ_MOT = OUT_DIR / "paper_minjerk_10cycles.mot"


def min_jerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def write_scenarios_csv() -> None:
    """Write the 5 static scenarios from the figure and Section 5.2."""
    scenarios = [
        ("a_reset", 0.0, 0.0, 0.0, "Reset posture"),
        ("b_low_elbow", 0.0, 30.0, 0.0, "Low elbow flexion"),
        ("c_moderate_elbow", 0.0, 60.0, 0.0, "Moderate elbow flexion"),
        ("d_high_elbow", 0.0, 80.0, 0.0, "High elbow flexion"),
        ("e_loaded_reaching", 0.0, 80.0, 2.0, "Loaded elbow flexion with 2 kg on hand"),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCENARIO_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "r_shoulder_elev_deg", "r_elbow_flex_deg", "load_kg", "notes"])
        for row in scenarios:
            w.writerow(row)


def write_minjerk_mot(
    dt: float = 0.01,
    n_cycles: int = 10,
    rest_time_s: float = 2.0,
    reach_time_s: float = 1.5,
    hold_time_s: float = 1.0,
    return_time_s: float = 1.5,
    shoulder_start_deg: float = 0.0,
        shoulder_target_deg: float = 0.0,
    elbow_start_deg: float = 0.0,
    elbow_target_deg: float = 80.0,
) -> None:
    """
        Create non-periodic elbow-flexion cycles with fixed upper arm:
      reach -> hold -> return -> rest, repeated n_cycles.
    """
    cycle_time = reach_time_s + hold_time_s + return_time_s + rest_time_s
    total_time = n_cycles * cycle_time
    t = np.arange(0.0, total_time + 0.5 * dt, dt)

    shoulder = np.full_like(t, shoulder_start_deg)
    elbow = np.full_like(t, elbow_start_deg)

    for c in range(n_cycles):
        t0 = c * cycle_time
        t1 = t0 + reach_time_s
        t2 = t1 + hold_time_s
        t3 = t2 + return_time_s

        m_reach = (t >= t0) & (t < t1)
        s = (t[m_reach] - t0) / reach_time_s
        shoulder[m_reach] = shoulder_start_deg + (shoulder_target_deg - shoulder_start_deg) * min_jerk(s)
        elbow[m_reach] = elbow_start_deg + (elbow_target_deg - elbow_start_deg) * min_jerk(s)

        m_hold = (t >= t1) & (t < t2)
        shoulder[m_hold] = shoulder_target_deg
        elbow[m_hold] = elbow_target_deg

        m_ret = (t >= t2) & (t < t3)
        s = (t[m_ret] - t2) / return_time_s
        shoulder[m_ret] = shoulder_target_deg + (shoulder_start_deg - shoulder_target_deg) * min_jerk(s)
        elbow[m_ret] = elbow_target_deg + (elbow_start_deg - elbow_target_deg) * min_jerk(s)

        # rest segment stays at start pose by initialization

    header = (
        f"{TRAJ_MOT.stem}\n"
        "version=1\n"
        f"nRows={len(t)}\n"
        "nColumns=3\n"
        "inDegrees=yes\n"
        "endheader\n"
        "time\tr_shoulder_elev\tr_elbow_flex\n"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAJ_MOT, "w") as f:
        f.write(header)
        for i in range(len(t)):
            f.write(f"{t[i]:.4f}\t{shoulder[i]:.6f}\t{elbow[i]:.6f}\n")


def main() -> int:
    write_scenarios_csv()
    write_minjerk_mot()

    print(f"Wrote scenarios: {SCENARIO_CSV}")
    print(f"Wrote trajectory: {TRAJ_MOT}")
    print("Use e_loaded_reaching with arm26_paper_loaded.osim for the 2 kg hand-load case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
