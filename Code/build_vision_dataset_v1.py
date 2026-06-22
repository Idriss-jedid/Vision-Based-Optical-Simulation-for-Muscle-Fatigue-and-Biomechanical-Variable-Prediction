"""
Build vision dataset v1 for the paper pipeline:
3D biomechanical coordinates -> pinhole projection -> Gaussian noise -> ML-ready CSV.

Expected input (default):
- Data/GUI_Run_001/gui_project_dataset_minimal.csv

Default outputs:
- Data/GUI_Run_001/vision_observations_clean.csv
- Data/GUI_Run_001/vision_dataset_v1.csv
- Data/GUI_Run_001/vision_dataset_v1_meta.json

Usage:
python "Vision-Based Optical Simulation/Code/build_vision_dataset_v1.py" \
  --run-dir "Vision-Based Optical Simulation/Data/GUI_Run_001"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_POINTS = [
    "r_humerus",
    "r_ulna_radius_hand",
    "load_2kg",
    "center_of_mass",
]

TARGET_PREFIXES = ("act_", "force_", "id_", "q_")


def _safe_depth(z: np.ndarray, eps: float) -> np.ndarray:
    sign = np.where(z >= 0.0, 1.0, -1.0)
    return np.where(np.abs(z) < eps, sign * eps, z)


def _available_points(df: pd.DataFrame, requested: List[str]) -> List[str]:
    available: List[str] = []
    for p in requested:
        need = [f"{p}_X", f"{p}_Y", f"{p}_Z"]
        if all(c in df.columns for c in need):
            available.append(p)
    return available


def project_points(
    df: pd.DataFrame,
    points: List[str],
    f: float,
    cx: float,
    cy: float,
    depth_eps: float,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for p in points:
        x = df[f"{p}_X"].to_numpy(dtype=float)
        y = df[f"{p}_Y"].to_numpy(dtype=float)
        z = df[f"{p}_Z"].to_numpy(dtype=float)

        z_safe = _safe_depth(z, depth_eps)
        u = (f * x / z_safe) + cx
        v = (f * y / z_safe) + cy

        out[f"{p}_u"] = u
        out[f"{p}_v"] = v
        out[f"{p}_depth"] = z
    return out


def add_gaussian_noise(df: pd.DataFrame, sigma_px: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noisy = pd.DataFrame(index=df.index)

    for c in df.columns:
        if c.endswith("_u") or c.endswith("_v"):
            noisy[f"{c}_noisy"] = df[c].to_numpy(dtype=float) + rng.normal(0.0, sigma_px, size=len(df))
        else:
            noisy[c] = df[c].to_numpy(dtype=float)

    return noisy


def collect_target_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(TARGET_PREFIXES)]


def build_dataset(
    input_csv: Path,
    out_clean: Path,
    out_v1: Path,
    out_meta: Path,
    points: List[str],
    f: float,
    width: int,
    height: int,
    cx: float | None,
    cy: float | None,
    sigma_px: float,
    seed: int,
    depth_eps: float,
    include_clean_columns: bool,
) -> Dict[str, object]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "time" not in df.columns:
        raise KeyError(f"Missing 'time' column in {input_csv}")

    cx_eff = float(width) / 2.0 if cx is None else float(cx)
    cy_eff = float(height) / 2.0 if cy is None else float(cy)

    points_available = _available_points(df, points)
    if not points_available:
        raise ValueError(
            "No requested 3D points found. Expected columns like <point>_X, <point>_Y, <point>_Z."
        )

    proj_clean = project_points(df, points_available, f=f, cx=cx_eff, cy=cy_eff, depth_eps=depth_eps)
    proj_noisy = add_gaussian_noise(proj_clean, sigma_px=sigma_px, seed=seed)

    targets = collect_target_columns(df)

    clean = pd.concat([df[["time"]], proj_clean], axis=1)

    v1_parts = [df[["time"]], proj_noisy]
    if include_clean_columns:
        v1_parts.append(proj_clean.add_suffix("_clean"))
    if targets:
        v1_parts.append(df[targets])
    v1 = pd.concat(v1_parts, axis=1)

    out_clean.parent.mkdir(parents=True, exist_ok=True)
    out_v1.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    clean.to_csv(out_clean, index=False)
    v1.to_csv(out_v1, index=False)

    meta: Dict[str, object] = {
        "input_csv": str(input_csv),
        "output_clean_csv": str(out_clean),
        "output_v1_csv": str(out_v1),
        "rows": int(len(v1)),
        "points_requested": points,
        "points_used": points_available,
        "camera": {
            "f": float(f),
            "width": int(width),
            "height": int(height),
            "cx": float(cx_eff),
            "cy": float(cy_eff),
            "depth_eps": float(depth_eps),
        },
        "noise": {
            "sigma_px": float(sigma_px),
            "seed": int(seed),
        },
        "targets_included": targets,
        "columns_clean": list(clean.columns),
        "columns_v1": list(v1.columns),
    }

    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_run_dir = root / "Data" / "GUI_Run_001"

    parser = argparse.ArgumentParser(description="Build vision dataset v1 from OpenSim-derived 3D data")

    parser.add_argument("--run-dir", type=Path, default=default_run_dir, help="Path to GUI run folder")
    parser.add_argument("--input-csv", type=Path, default=None, help="Input merged CSV (defaults to gui_project_dataset_minimal.csv)")

    parser.add_argument("--points", nargs="+", default=DEFAULT_POINTS, help="3D points to project")

    parser.add_argument("--f", type=float, default=1200.0, help="Focal length in pixels")
    parser.add_argument("--image-width", type=int, default=1280, help="Image width in pixels")
    parser.add_argument("--image-height", type=int, default=720, help="Image height in pixels")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x (pixels), default width/2")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y (pixels), default height/2")
    parser.add_argument("--depth-eps", type=float, default=1e-6, help="Minimum absolute depth to avoid divide by zero")

    parser.add_argument("--sigma-px", type=float, default=2.0, help="Gaussian noise std in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--include-clean-columns", action="store_true", help="Include clean 2D columns in final v1 CSV")

    parser.add_argument("--out-clean", type=Path, default=None, help="Output clean observations CSV")
    parser.add_argument("--out-v1", type=Path, default=None, help="Output final v1 dataset CSV")
    parser.add_argument("--out-meta", type=Path, default=None, help="Output metadata JSON")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_dir: Path = args.run_dir
    input_csv = args.input_csv or (run_dir / "gui_project_dataset_minimal.csv")

    out_clean = args.out_clean or (run_dir / "vision_observations_clean.csv")
    out_v1 = args.out_v1 or (run_dir / "vision_dataset_v1.csv")
    out_meta = args.out_meta or (run_dir / "vision_dataset_v1_meta.json")

    meta = build_dataset(
        input_csv=input_csv,
        out_clean=out_clean,
        out_v1=out_v1,
        out_meta=out_meta,
        points=args.points,
        f=args.f,
        width=args.image_width,
        height=args.image_height,
        cx=args.cx,
        cy=args.cy,
        sigma_px=args.sigma_px,
        seed=args.seed,
        depth_eps=args.depth_eps,
        include_clean_columns=bool(args.include_clean_columns),
    )

    print(f"Saved clean observations: {out_clean}")
    print(f"Saved v1 dataset: {out_v1}")
    print(f"Saved metadata: {out_meta}")
    print(f"Rows: {meta['rows']}")
    print(f"Points used: {', '.join(meta['points_used'])}")
    print(f"Targets included: {len(meta['targets_included'])}")
    print(f"Columns (v1): {len(meta['columns_v1'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
