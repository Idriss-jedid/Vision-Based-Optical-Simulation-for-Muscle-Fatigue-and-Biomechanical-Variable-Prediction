"""
Combine OpenSim GUI result files into project-ready datasets.

Essential sources:
- analyze/arm26_BodyKinematics_pos_global.sto
- static_opt/arm26_StaticOptimization_activation.sto
- static_opt/arm26_StaticOptimization_force.sto

Optional sources:
- inverse_dynamics/inverse_dynamics.sto
- analyze/arm26_Kinematics_q.sto

Outputs:
- gui_project_dataset_full.csv
- gui_project_dataset_minimal.csv

Usage (from workspace root):
python "Vision-Based Optical Simulation/Code/combine_gui_results.py" \
  --run-dir "Vision-Based Optical Simulation/Data/GUI_Run_001"
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BICEPS = ["BIClong", "BICshort", "BRA", "BRD_hand"]
TRICEPS = ["TRIlong", "TRIlat", "TRImed"]
DELTOID = ["DELT_ant", "DELT_post", "PECT"]


def parse_opensim_table(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    metadata: Dict[str, str] = {}
    end_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if "=" in s and end_idx is None:
            k, v = s.split("=", 1)
            metadata[k.strip().lower()] = v.strip()
        if s.lower() == "endheader":
            end_idx = i
            break

    if end_idx is None:
        raise ValueError(f"Could not find endheader in {path}")

    body = "\n".join(lines[end_idx + 1 :]).strip()
    if not body:
        raise ValueError(f"No data found after endheader in {path}")

    df = pd.read_csv(io.StringIO(body), sep=r"\s+", engine="python")
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df, metadata


def align_to_time(df: pd.DataFrame, t_ref: np.ndarray) -> pd.DataFrame:
    t_src = df["time"].to_numpy(dtype=float)
    out = {"time": t_ref}
    for c in df.columns:
        if c == "time":
            continue
        out[c] = np.interp(t_ref, t_src, df[c].to_numpy(dtype=float), left=df[c].iloc[0], right=df[c].iloc[-1])
    return pd.DataFrame(out)


def ensure_has_time(df: pd.DataFrame, path: Path) -> None:
    if "time" not in df.columns:
        raise KeyError(f"'time' column not found in {path}")


def mean_existing(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return np.full(len(df), np.nan, dtype=float)
    return df[existing].to_numpy(dtype=float).mean(axis=1)


def build_datasets(run_dir: Path, out_full: Path, out_min: Path) -> None:
    pos_path = run_dir / "analyze" / "arm26_BodyKinematics_pos_global.sto"
    act_path = run_dir / "static_opt" / "arm26_StaticOptimization_activation.sto"
    force_path = run_dir / "static_opt" / "arm26_StaticOptimization_force.sto"

    id_path = run_dir / "inverse_dynamics" / "inverse_dynamics.sto"
    q_path = run_dir / "analyze" / "arm26_Kinematics_q.sto"

    for p in [pos_path, act_path, force_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    pos_df, _ = parse_opensim_table(pos_path)
    act_df, _ = parse_opensim_table(act_path)
    force_df, _ = parse_opensim_table(force_path)

    ensure_has_time(pos_df, pos_path)
    ensure_has_time(act_df, act_path)
    ensure_has_time(force_df, force_path)

    t_ref = pos_df["time"].to_numpy(dtype=float)
    act_aligned = align_to_time(act_df, t_ref)
    force_aligned = align_to_time(force_df, t_ref)

    # Optional files
    id_aligned = None
    if id_path.exists():
        id_df, _ = parse_opensim_table(id_path)
        ensure_has_time(id_df, id_path)
        id_aligned = align_to_time(id_df, t_ref)

    q_aligned = None
    if q_path.exists():
        q_df, _ = parse_opensim_table(q_path)
        ensure_has_time(q_df, q_path)
        q_aligned = align_to_time(q_df, t_ref)

    full = pd.DataFrame({"time": t_ref})

    # Keep all 3D position channels from BodyKinematics_pos_global
    pos_cols = [c for c in pos_df.columns if c != "time"]
    for c in pos_cols:
        full[c] = pos_df[c].to_numpy(dtype=float)

    # Individual activations and forces
    act_cols = [c for c in act_aligned.columns if c != "time"]
    force_cols = [c for c in force_aligned.columns if c != "time"]
    for c in act_cols:
        full[f"act_{c}"] = act_aligned[c].to_numpy(dtype=float)
    for c in force_cols:
        full[f"force_{c}"] = force_aligned[c].to_numpy(dtype=float)

    # Grouped supervision targets
    full["act_biceps"] = mean_existing(act_aligned, BICEPS)
    full["act_triceps"] = mean_existing(act_aligned, TRICEPS)
    full["act_deltoid"] = mean_existing(act_aligned, DELTOID)

    full["force_biceps"] = mean_existing(force_aligned, BICEPS)
    full["force_triceps"] = mean_existing(force_aligned, TRICEPS)
    full["force_deltoid"] = mean_existing(force_aligned, DELTOID)

    if id_aligned is not None:
        if "r_shoulder_elev_moment" in id_aligned.columns:
            full["id_shoulder_moment"] = id_aligned["r_shoulder_elev_moment"].to_numpy(dtype=float)
        if "r_elbow_flex_moment" in id_aligned.columns:
            full["id_elbow_moment"] = id_aligned["r_elbow_flex_moment"].to_numpy(dtype=float)

    if q_aligned is not None:
        if "r_shoulder_elev" in q_aligned.columns:
            full["q_shoulder"] = q_aligned["r_shoulder_elev"].to_numpy(dtype=float)
        if "r_elbow_flex" in q_aligned.columns:
            full["q_elbow"] = q_aligned["r_elbow_flex"].to_numpy(dtype=float)

    # Minimal v1 dataset for vision -> biomech
    preferred_pos = [
        "r_humerus_X",
        "r_humerus_Y",
        "r_humerus_Z",
        "r_ulna_radius_hand_X",
        "r_ulna_radius_hand_Y",
        "r_ulna_radius_hand_Z",
        "load_2kg_X",
        "load_2kg_Y",
        "load_2kg_Z",
        "center_of_mass_X",
        "center_of_mass_Y",
        "center_of_mass_Z",
    ]
    min_cols = ["time"] + [c for c in preferred_pos if c in full.columns]

    # Add minimal targets
    min_cols += [
        "act_biceps",
        "act_triceps",
        "act_deltoid",
        "force_biceps",
        "force_triceps",
        "force_deltoid",
    ]

    for optional_col in ["id_shoulder_moment", "id_elbow_moment", "q_shoulder", "q_elbow"]:
        if optional_col in full.columns:
            min_cols.append(optional_col)

    minimal = full[min_cols].copy()

    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_min.parent.mkdir(parents=True, exist_ok=True)

    full.to_csv(out_full, index=False)
    minimal.to_csv(out_min, index=False)

    print(f"Saved full dataset: {out_full}")
    print(f"Saved minimal dataset: {out_min}")
    print(f"Rows: {len(full)}")
    print(f"Columns (full): {len(full.columns)}")
    print(f"Columns (minimal): {len(minimal.columns)}")
    print(f"NaN cells (minimal): {int(minimal.isna().sum().sum())}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_run_dir = root / "Data" / "GUI_Run_001"

    parser = argparse.ArgumentParser(description="Combine OpenSim GUI results into project datasets")
    parser.add_argument("--run-dir", type=Path, default=default_run_dir, help="Path to GUI_Run_001 folder")
    parser.add_argument("--out-full", type=Path, default=None, help="Output CSV for full dataset")
    parser.add_argument("--out-min", type=Path, default=None, help="Output CSV for minimal dataset")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir

    out_full = args.out_full or (run_dir / "gui_project_dataset_full.csv")
    out_min = args.out_min or (run_dir / "gui_project_dataset_minimal.csv")

    build_datasets(run_dir=run_dir, out_full=out_full, out_min=out_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
