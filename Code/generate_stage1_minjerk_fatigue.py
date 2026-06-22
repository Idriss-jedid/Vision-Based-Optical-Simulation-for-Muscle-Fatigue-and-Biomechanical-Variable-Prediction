"""
Generate Stage-1 repetitive minimum-jerk elbow-flexion motion with fatigue.

This script creates a new version of Data/paper_minjerk_10cycles.mot using:
    - two minimum-jerk segments per repetition: flexion and extension
    - a fixed stabilizing shoulder angle
    - progressive fatigue: slower motion, reduced ROM, tremor/noise, asymmetry

Outputs by default:
    Data/paper_minjerk_fatigue_30cycles.mot
        OpenSim-compatible coordinate file: time, r_shoulder_elev, r_elbow_flex
    Data/paper_minjerk_fatigue_30cycles_full.csv
        Per-frame analysis table with velocity, acceleration, jerk, fatigue labels
    Data/paper_minjerk_fatigue_30cycles_reps.csv
        Per-repetition fatigue parameters
    Data/paper_minjerk_fatigue_30cycles_meta.json
        Reproducibility metadata and generation settings
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DATA_DIR = WORKSPACE / "Data"

SHOULDER_COORD = "r_shoulder_elev"
ELBOW_COORD = "r_elbow_flex"
DEFAULT_OUT_STEM = "paper_minjerk_fatigue_30cycles"


@dataclass(frozen=True)
class Stage1Config:
    sample_rate_hz: float = 100.0
    repetitions: int = 30
    q0_deg: float = 20.0
    qf_fresh_deg: float = 120.0
    qf_fatigued_deg: float = 100.0
    shoulder_stabilized_deg: float = 20.0
    flexion_duration_fresh_s: float = 2.0
    flexion_duration_fatigued_s: float = 3.0
    extension_duration_fresh_s: float = 2.0
    extension_duration_fatigued_s: float = 2.35
    top_pause_s: float = 0.2
    bottom_pause_s: float = 0.2
    tremor_final_amp_deg: float = 1.2
    noise_final_std_deg: float = 0.25
    tremor_freq_1_hz: float = 6.0
    tremor_freq_2_hz: float = 10.0
    random_seed: int = 20260529


Frame = Dict[str, float | int | str]
RepSummary = Dict[str, float | int]


def min_jerk(s: float) -> float:
    """Classical fifth-order minimum-jerk blend: 10s^3 - 15s^4 + 6s^5."""
    s = min(1.0, max(0.0, s))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def lerp(start: float, end: float, amount: float, power: float = 1.0) -> float:
    shaped = min(1.0, max(0.0, amount)) ** power
    return start + (end - start) * shaped


def fatigue_level(rep_zero_based: int, repetitions: int) -> float:
    if repetitions <= 1:
        return 0.0
    return rep_zero_based / float(repetitions - 1)


def quantized_steps(duration_s: float, dt: float) -> int:
    return max(1, int(round(duration_s / dt)))


def rep_parameters(rep_index: int, cfg: Stage1Config, dt: float) -> RepSummary:
    fatigue = fatigue_level(rep_index - 1, cfg.repetitions)
    flexion_s = lerp(cfg.flexion_duration_fresh_s, cfg.flexion_duration_fatigued_s, fatigue, 1.15)
    extension_s = lerp(cfg.extension_duration_fresh_s, cfg.extension_duration_fatigued_s, fatigue, 1.10)
    flexion_steps = quantized_steps(flexion_s, dt)
    extension_steps = quantized_steps(extension_s, dt)
    target_elbow = lerp(cfg.qf_fresh_deg, cfg.qf_fatigued_deg, fatigue, 1.25)
    tremor_amp = cfg.tremor_final_amp_deg * fatigue**1.40
    noise_std = cfg.noise_final_std_deg * fatigue**1.50

    return {
        "rep": rep_index,
        "fatigue_level": fatigue,
        "target_elbow_deg": target_elbow,
        "flexion_duration_s": flexion_steps * dt,
        "extension_duration_s": extension_steps * dt,
        "top_pause_s": quantized_steps(cfg.top_pause_s, dt) * dt,
        "bottom_pause_s": quantized_steps(cfg.bottom_pause_s, dt) * dt,
        "tremor_amp_deg": tremor_amp,
        "noise_std_deg": noise_std,
        "asymmetry_s": flexion_steps * dt - extension_steps * dt,
    }


def append_frame(
    frames: List[Frame],
    time_s: float,
    shoulder_deg: float,
    elbow_deg: float,
    rep: int,
    phase: str,
    fatigue: float,
    target_elbow_deg: float,
    flexion_duration_s: float,
    extension_duration_s: float,
    perturbation_deg: float,
) -> None:
    frames.append(
        {
            "time": round(time_s, 10),
            SHOULDER_COORD: shoulder_deg,
            ELBOW_COORD: elbow_deg,
            "rep": rep,
            "phase": phase,
            "fatigue_level": fatigue,
            "target_elbow_deg": target_elbow_deg,
            "flexion_duration_s": flexion_duration_s,
            "extension_duration_s": extension_duration_s,
            "perturbation_deg": perturbation_deg,
        }
    )


def add_minjerk_segment(
    frames: List[Frame],
    start_time_s: float,
    start_angle_deg: float,
    end_angle_deg: float,
    duration_s: float,
    shoulder_deg: float,
    rep: int,
    phase: str,
    rep_summary: RepSummary,
    cfg: Stage1Config,
    rng: random.Random,
    dt: float,
) -> float:
    steps = quantized_steps(duration_s, dt)
    actual_duration_s = steps * dt
    fatigue = float(rep_summary["fatigue_level"])
    tremor_amp = float(rep_summary["tremor_amp_deg"])
    noise_std = float(rep_summary["noise_std_deg"])
    phase_1 = rng.uniform(0.0, 2.0 * math.pi)
    phase_2 = rng.uniform(0.0, 2.0 * math.pi)

    time_s = start_time_s
    for step in range(steps):
        s = step / float(steps)
        blend = min_jerk(s)
        base_angle = start_angle_deg + (end_angle_deg - start_angle_deg) * blend

        # The envelope forces perturbations to zero at both movement endpoints.
        envelope = math.sin(math.pi * s) ** 2
        tremor = envelope * tremor_amp * (
            0.75 * math.sin(2.0 * math.pi * cfg.tremor_freq_1_hz * time_s + phase_1)
            + 0.25 * math.sin(2.0 * math.pi * cfg.tremor_freq_2_hz * time_s + phase_2)
        )
        noise = envelope * rng.gauss(0.0, noise_std)
        perturbation = tremor + noise

        append_frame(
            frames=frames,
            time_s=time_s,
            shoulder_deg=shoulder_deg,
            elbow_deg=base_angle + perturbation,
            rep=rep,
            phase=phase,
            fatigue=fatigue,
            target_elbow_deg=float(rep_summary["target_elbow_deg"]),
            flexion_duration_s=float(rep_summary["flexion_duration_s"]),
            extension_duration_s=float(rep_summary["extension_duration_s"]),
            perturbation_deg=perturbation,
        )
        time_s += dt

    return start_time_s + actual_duration_s


def add_pause(
    frames: List[Frame],
    start_time_s: float,
    duration_s: float,
    shoulder_deg: float,
    elbow_deg: float,
    rep: int,
    phase: str,
    rep_summary: RepSummary,
    dt: float,
) -> float:
    steps = quantized_steps(duration_s, dt)
    time_s = start_time_s
    for _ in range(steps):
        append_frame(
            frames=frames,
            time_s=time_s,
            shoulder_deg=shoulder_deg,
            elbow_deg=elbow_deg,
            rep=rep,
            phase=phase,
            fatigue=float(rep_summary["fatigue_level"]),
            target_elbow_deg=float(rep_summary["target_elbow_deg"]),
            flexion_duration_s=float(rep_summary["flexion_duration_s"]),
            extension_duration_s=float(rep_summary["extension_duration_s"]),
            perturbation_deg=0.0,
        )
        time_s += dt
    return start_time_s + steps * dt


def add_numeric_derivatives(frames: List[Frame], dt: float) -> None:
    elbow = [float(row[ELBOW_COORD]) for row in frames]

    def derivative(values: List[float]) -> List[float]:
        result: List[float] = []
        for i, value in enumerate(values):
            if len(values) == 1:
                result.append(0.0)
            elif i == 0:
                result.append((values[i + 1] - value) / dt)
            elif i == len(values) - 1:
                result.append((value - values[i - 1]) / dt)
            else:
                result.append((values[i + 1] - values[i - 1]) / (2.0 * dt))
        return result

    velocity = derivative(elbow)
    acceleration = derivative(velocity)
    jerk = derivative(acceleration)

    for row, vel, acc, jrk in zip(frames, velocity, acceleration, jerk):
        row["elbow_velocity_deg_s"] = vel
        row["elbow_acceleration_deg_s2"] = acc
        row["elbow_jerk_deg_s3"] = jrk


def generate_motion(cfg: Stage1Config) -> Tuple[List[Frame], List[RepSummary]]:
    dt = 1.0 / cfg.sample_rate_hz
    rng = random.Random(cfg.random_seed)
    frames: List[Frame] = []
    rep_rows: List[RepSummary] = []
    time_s = 0.0

    for rep in range(1, cfg.repetitions + 1):
        rep_summary = rep_parameters(rep, cfg, dt)
        rep_rows.append(rep_summary)
        target_elbow = float(rep_summary["target_elbow_deg"])

        time_s = add_minjerk_segment(
            frames=frames,
            start_time_s=time_s,
            start_angle_deg=cfg.q0_deg,
            end_angle_deg=target_elbow,
            duration_s=float(rep_summary["flexion_duration_s"]),
            shoulder_deg=cfg.shoulder_stabilized_deg,
            rep=rep,
            phase="flexion",
            rep_summary=rep_summary,
            cfg=cfg,
            rng=rng,
            dt=dt,
        )
        time_s = add_pause(
            frames=frames,
            start_time_s=time_s,
            duration_s=float(rep_summary["top_pause_s"]),
            shoulder_deg=cfg.shoulder_stabilized_deg,
            elbow_deg=target_elbow,
            rep=rep,
            phase="top_pause",
            rep_summary=rep_summary,
            dt=dt,
        )
        time_s = add_minjerk_segment(
            frames=frames,
            start_time_s=time_s,
            start_angle_deg=target_elbow,
            end_angle_deg=cfg.q0_deg,
            duration_s=float(rep_summary["extension_duration_s"]),
            shoulder_deg=cfg.shoulder_stabilized_deg,
            rep=rep,
            phase="extension",
            rep_summary=rep_summary,
            cfg=cfg,
            rng=rng,
            dt=dt,
        )
        time_s = add_pause(
            frames=frames,
            start_time_s=time_s,
            duration_s=float(rep_summary["bottom_pause_s"]),
            shoulder_deg=cfg.shoulder_stabilized_deg,
            elbow_deg=cfg.q0_deg,
            rep=rep,
            phase="bottom_pause",
            rep_summary=rep_summary,
            dt=dt,
        )

    final_rep = rep_rows[-1] if rep_rows else rep_parameters(1, cfg, dt)
    append_frame(
        frames=frames,
        time_s=time_s,
        shoulder_deg=cfg.shoulder_stabilized_deg,
        elbow_deg=cfg.q0_deg,
        rep=cfg.repetitions,
        phase="complete",
        fatigue=1.0 if cfg.repetitions > 1 else 0.0,
        target_elbow_deg=float(final_rep["target_elbow_deg"]),
        flexion_duration_s=float(final_rep["flexion_duration_s"]),
        extension_duration_s=float(final_rep["extension_duration_s"]),
        perturbation_deg=0.0,
    )

    add_numeric_derivatives(frames, dt)
    return frames, rep_rows


def write_mot(frames: List[Frame], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        out_path.stem,
        "version=1",
        f"nRows={len(frames)}",
        "nColumns=3",
        "inDegrees=yes",
        "endheader",
        f"time\t{SHOULDER_COORD}\t{ELBOW_COORD}",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(header) + "\n")
        for row in frames:
            handle.write(
                f"{float(row['time']):.4f}\t"
                f"{float(row[SHOULDER_COORD]):.6f}\t"
                f"{float(row[ELBOW_COORD]):.6f}\n"
            )


def write_csv(rows: List[Dict[str, float | int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {out_path}")
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(cfg: Stage1Config, frames: List[Frame], rep_rows: List[RepSummary], out_path: Path) -> None:
    duration_s = float(frames[-1]["time"]) if frames else 0.0
    elbow_values = [float(row[ELBOW_COORD]) for row in frames]
    payload = {
        "purpose": "Stage 1 minimum-jerk elbow-curl motion with progressive fatigue perturbation.",
        "model_equation": "q(t) = q0 + (qf - q0) * (10*s^3 - 15*s^4 + 6*s^5), s = t/T",
        "fatigue_rules": {
            "slower_movement": "flexion duration increases more than extension duration",
            "reduced_rom": "target elbow flexion decreases across repetitions",
            "less_smoothness": "enveloped tremor plus small Gaussian perturbation increase with fatigue",
            "asymmetry": "flexion-extension duration difference grows across repetitions",
        },
        "config": asdict(cfg),
        "summary": {
            "rows": len(frames),
            "duration_s": duration_s,
            "elbow_min_deg": min(elbow_values),
            "elbow_max_deg": max(elbow_values),
            "first_rep": rep_rows[0] if rep_rows else {},
            "last_rep": rep_rows[-1] if rep_rows else {},
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage-1 fatigue-aware minimum-jerk arm26 motion.")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR, help="Directory for generated files.")
    parser.add_argument("--out-stem", default=DEFAULT_OUT_STEM, help="Output file stem without extension.")
    parser.add_argument("--reps", type=int, default=30, help="Number of elbow-curl repetitions.")
    parser.add_argument("--sample-rate", type=float, default=100.0, help="Sampling rate in Hz.")
    parser.add_argument("--q0", type=float, default=20.0, help="Initial elbow angle in degrees.")
    parser.add_argument("--qf-fresh", type=float, default=120.0, help="Fresh max elbow angle in degrees.")
    parser.add_argument("--qf-fatigued", type=float, default=100.0, help="Final fatigued max elbow angle in degrees.")
    parser.add_argument("--shoulder", type=float, default=20.0, help="Fixed stabilizing shoulder angle in degrees.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Stage1Config(
        sample_rate_hz=args.sample_rate,
        repetitions=args.reps,
        q0_deg=args.q0,
        qf_fresh_deg=args.qf_fresh,
        qf_fatigued_deg=args.qf_fatigued,
        shoulder_stabilized_deg=args.shoulder,
    )

    frames, rep_rows = generate_motion(cfg)
    out_dir = args.out_dir
    out_stem = args.out_stem

    mot_path = out_dir / f"{out_stem}.mot"
    full_csv_path = out_dir / f"{out_stem}_full.csv"
    reps_csv_path = out_dir / f"{out_stem}_reps.csv"
    meta_path = out_dir / f"{out_stem}_meta.json"

    write_mot(frames, mot_path)
    write_csv(frames, full_csv_path)
    write_csv(rep_rows, reps_csv_path)
    write_metadata(cfg, frames, rep_rows, meta_path)

    print(f"Wrote OpenSim motion: {mot_path}")
    print(f"Wrote full per-frame table: {full_csv_path}")
    print(f"Wrote repetition summary: {reps_csv_path}")
    print(f"Wrote metadata: {meta_path}")
    print(
        "Summary: "
        f"{len(frames)} rows, "
        f"duration={float(frames[-1]['time']):.2f}s, "
        f"elbow range={min(float(r[ELBOW_COORD]) for r in frames):.2f}.."
        f"{max(float(r[ELBOW_COORD]) for r in frames):.2f} deg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())