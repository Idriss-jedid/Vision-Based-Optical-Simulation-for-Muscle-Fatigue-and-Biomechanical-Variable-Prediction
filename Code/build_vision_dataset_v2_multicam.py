"""
Build vision dataset v2 with multiple virtual cameras.

Pipeline:
3D OpenSim coordinates -> camera extrinsics/intrinsics -> visibility/dropout/noise
-> multi-camera CSV for AI training.

Expected input (default):
- Data/GUI_Run_001/gui_project_dataset_minimal.csv

Expected camera config (default):
- Data/camera_configs_v2.json

Default outputs:
- Data/GUI_Run_001/vision_dataset_v2_multicam.csv
- Data/GUI_Run_001/vision_dataset_v2_meta.json
- optional Data/GUI_Run_001/vision_dataset_v2_long.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_POINTS = [
    "r_humerus",
    "r_ulna_radius_hand",
    "load_2kg",
    "center_of_mass",
]

DEFAULT_TARGET_PREFIXES = ["act_", "force_", "id_", "q_"]


def _available_points(df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    available: List[str] = []
    for p in requested:
        required = [f"{p}_X", f"{p}_Y", f"{p}_Z"]
        if all(c in df.columns for c in required):
            available.append(p)
    return available


def _collect_targets(df: pd.DataFrame, prefixes: Sequence[str]) -> List[str]:
    return [c for c in df.columns if any(c.startswith(pref) for pref in prefixes)]


def _safe_depth(z: np.ndarray, eps: float) -> np.ndarray:
    sign = np.where(z >= 0.0, 1.0, -1.0)
    return np.where(np.abs(z) < eps, sign * eps, z)


def _as_mat3(x: Any, fallback: np.ndarray) -> np.ndarray:
    if x is None:
        return fallback
    arr = np.asarray(x, dtype=float)
    if arr.size != 9:
        raise ValueError(f"Camera rotation must have 9 values, got {arr.size}")
    return arr.reshape(3, 3)


def _as_vec3(x: Any, fallback: np.ndarray) -> np.ndarray:
    if x is None:
        return fallback
    arr = np.asarray(x, dtype=float)
    if arr.size != 3:
        raise ValueError(f"Camera translation must have 3 values, got {arr.size}")
    return arr.reshape(3)


def _load_camera_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Camera config not found: {path}. Create it from Data/camera_configs_v2.json template."
        )

    cfg = json.loads(path.read_text(encoding="utf-8"))
    image = cfg.get("image", {})

    width = int(image.get("width", 1280))
    height = int(image.get("height", 720))

    cameras_raw = cfg.get("cameras", [])
    if not isinstance(cameras_raw, list) or not cameras_raw:
        raise ValueError("Camera config must contain a non-empty cameras list")

    cams: List[Dict[str, Any]] = []
    for i, cam in enumerate(cameras_raw):
        if not isinstance(cam, dict):
            raise ValueError(f"Camera at index {i} is not an object")

        cam_id = str(cam.get("id", f"cam_{i:02d}"))
        fx = float(cam.get("fx", 1200.0))
        fy = float(cam.get("fy", fx))
        cx = float(cam.get("cx", width / 2.0))
        cy = float(cam.get("cy", height / 2.0))

        R = _as_mat3(cam.get("R"), np.eye(3, dtype=float))
        t = _as_vec3(cam.get("t"), np.zeros(3, dtype=float))

        sigma_px = float(cam.get("sigma_px", 2.0))
        dropout_prob = float(cam.get("dropout_prob", 0.0))
        depth_eps = float(cam.get("depth_eps", 1e-6))
        clip_to_image = bool(cam.get("clip_to_image", True))

        if dropout_prob < 0.0 or dropout_prob > 1.0:
            raise ValueError(f"Camera {cam_id}: dropout_prob must be in [0,1]")

        cams.append(
            {
                "id": cam_id,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "R": R,
                "t": t,
                "sigma_px": sigma_px,
                "dropout_prob": dropout_prob,
                "depth_eps": depth_eps,
                "clip_to_image": clip_to_image,
            }
        )

    return {
        "path": str(path),
        "width": width,
        "height": height,
        "cameras": cams,
    }


def _project_one_camera(
    world_xyz: np.ndarray,
    cam: Dict[str, Any],
    width: int,
    height: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    R = cam["R"]
    t = cam["t"]
    fx = cam["fx"]
    fy = cam["fy"]
    cx = cam["cx"]
    cy = cam["cy"]
    sigma_px = cam["sigma_px"]
    dropout_prob = cam["dropout_prob"]
    depth_eps = cam["depth_eps"]
    clip_to_image = cam["clip_to_image"]

    cam_xyz = (R @ world_xyz.T).T + t
    x = cam_xyz[:, 0]
    y = cam_xyz[:, 1]
    z = cam_xyz[:, 2]

    z_safe = _safe_depth(z, depth_eps)

    u = (fx * x / z_safe) + cx
    v = (fy * y / z_safe) + cy

    visible = z > depth_eps
    if clip_to_image:
        visible &= (u >= 0.0) & (u <= (width - 1)) & (v >= 0.0) & (v <= (height - 1))

    if dropout_prob > 0.0:
        dropped = rng.random(len(u)) < dropout_prob
        visible &= ~dropped

    nu = rng.normal(0.0, sigma_px, size=len(u)) if sigma_px > 0 else np.zeros(len(u), dtype=float)
    nv = rng.normal(0.0, sigma_px, size=len(v)) if sigma_px > 0 else np.zeros(len(v), dtype=float)

    u_noisy = np.where(visible, u + nu, np.nan)
    v_noisy = np.where(visible, v + nv, np.nan)

    x_norm = np.where(visible, (u_noisy - cx) / fx, np.nan)
    y_norm = np.where(visible, (v_noisy - cy) / fy, np.nan)

    return {
        "u": u,
        "v": v,
        "u_noisy": u_noisy,
        "v_noisy": v_noisy,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "depth": z,
        "visible": visible.astype(int),
    }


def build_v2_multicam(
    input_csv: Path,
    camera_config: Path,
    out_wide: Path,
    out_meta: Path,
    points: Sequence[str],
    target_prefixes: Sequence[str],
    seed: int,
    write_long: bool,
    out_long: Path | None,
) -> Dict[str, Any]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "time" not in df.columns:
        raise KeyError(f"Missing time column in {input_csv}")

    points_used = _available_points(df, points)
    if not points_used:
        raise ValueError("None of the requested points are available in the input CSV")

    targets = _collect_targets(df, target_prefixes)

    cfg = _load_camera_config(camera_config)
    width = cfg["width"]
    height = cfg["height"]
    cameras = cfg["cameras"]

    rng = np.random.default_rng(seed)

    wide = pd.DataFrame({"time": df["time"].to_numpy(dtype=float)})
    long_parts: List[pd.DataFrame] = []

    visible_stats: Dict[str, float] = {}

    for cam in cameras:
        cam_id = cam["id"]

        for p in points_used:
            xyz = df[[f"{p}_X", f"{p}_Y", f"{p}_Z"]].to_numpy(dtype=float)
            proj = _project_one_camera(xyz, cam=cam, width=width, height=height, rng=rng)

            prefix = f"{cam_id}_{p}"
            wide[f"{prefix}_u"] = proj["u"]
            wide[f"{prefix}_v"] = proj["v"]
            wide[f"{prefix}_u_noisy"] = proj["u_noisy"]
            wide[f"{prefix}_v_noisy"] = proj["v_noisy"]
            wide[f"{prefix}_x_norm"] = proj["x_norm"]
            wide[f"{prefix}_y_norm"] = proj["y_norm"]
            wide[f"{prefix}_depth"] = proj["depth"]
            wide[f"{prefix}_visible"] = proj["visible"]

            key = f"{cam_id}:{p}"
            visible_stats[key] = float(np.mean(proj["visible"]))

            if write_long:
                long_parts.append(
                    pd.DataFrame(
                        {
                            "time": df["time"].to_numpy(dtype=float),
                            "camera_id": cam_id,
                            "point_id": p,
                            "u": proj["u"],
                            "v": proj["v"],
                            "u_noisy": proj["u_noisy"],
                            "v_noisy": proj["v_noisy"],
                            "x_norm": proj["x_norm"],
                            "y_norm": proj["y_norm"],
                            "depth": proj["depth"],
                            "visible": proj["visible"],
                        }
                    )
                )

    if targets:
        wide = pd.concat([wide, df[targets]], axis=1)

    out_wide.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    wide.to_csv(out_wide, index=False)

    long_csv_path = None
    if write_long:
        if out_long is None:
            out_long = out_wide.with_name("vision_dataset_v2_long.csv")

        long_df = pd.concat(long_parts, axis=0, ignore_index=True)
        if targets:
            long_df = long_df.merge(df[["time"] + targets], on="time", how="left")

        out_long.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(out_long, index=False)
        long_csv_path = str(out_long)

    meta = {
        "input_csv": str(input_csv),
        "camera_config": str(camera_config),
        "output_wide_csv": str(out_wide),
        "output_long_csv": long_csv_path,
        "rows": int(len(wide)),
        "points_requested": list(points),
        "points_used": points_used,
        "targets_included": targets,
        "image": {"width": width, "height": height},
        "seed": int(seed),
        "camera_count": len(cameras),
        "cameras": [
            {
                "id": c["id"],
                "fx": float(c["fx"]),
                "fy": float(c["fy"]),
                "cx": float(c["cx"]),
                "cy": float(c["cy"]),
                "sigma_px": float(c["sigma_px"]),
                "dropout_prob": float(c["dropout_prob"]),
                "depth_eps": float(c["depth_eps"]),
                "clip_to_image": bool(c["clip_to_image"]),
            }
            for c in cameras
        ],
        "visibility_rate": visible_stats,
        "columns_wide": list(wide.columns),
    }

    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_run_dir = root / "Data" / "GUI_Run_001"

    parser = argparse.ArgumentParser(description="Build vision dataset v2 (multi-camera) from OpenSim-derived 3D data")

    parser.add_argument("--run-dir", type=Path, default=default_run_dir, help="Path to GUI run folder")
    parser.add_argument("--input-csv", type=Path, default=None, help="Input merged CSV (default: gui_project_dataset_minimal.csv)")
    parser.add_argument("--camera-config", type=Path, default=root / "Data" / "camera_configs_v2.json", help="Camera config JSON")

    parser.add_argument("--points", nargs="+", default=DEFAULT_POINTS, help="3D points to project")
    parser.add_argument("--target-prefixes", nargs="+", default=DEFAULT_TARGET_PREFIXES, help="Prefixes for target columns")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--write-long", action="store_true", help="Also write long-format CSV")

    parser.add_argument("--out-wide", type=Path, default=None, help="Output wide CSV (default: vision_dataset_v2_multicam.csv)")
    parser.add_argument("--out-long", type=Path, default=None, help="Output long CSV (default: vision_dataset_v2_long.csv)")
    parser.add_argument("--out-meta", type=Path, default=None, help="Output metadata JSON (default: vision_dataset_v2_meta.json)")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_dir = args.run_dir
    input_csv = args.input_csv or (run_dir / "gui_project_dataset_minimal.csv")
    out_wide = args.out_wide or (run_dir / "vision_dataset_v2_multicam.csv")
    out_meta = args.out_meta or (run_dir / "vision_dataset_v2_meta.json")

    meta = build_v2_multicam(
        input_csv=input_csv,
        camera_config=args.camera_config,
        out_wide=out_wide,
        out_meta=out_meta,
        points=args.points,
        target_prefixes=args.target_prefixes,
        seed=args.seed,
        write_long=bool(args.write_long),
        out_long=args.out_long,
    )

    print(f"Saved v2 multi-camera dataset: {out_wide}")
    if meta.get("output_long_csv"):
        print(f"Saved v2 long dataset: {meta['output_long_csv']}")
    print(f"Saved metadata: {out_meta}")
    print(f"Rows: {meta['rows']}")
    print(f"Points used: {', '.join(meta['points_used'])}")
    print(f"Cameras: {meta['camera_count']}")
    print(f"Targets included: {len(meta['targets_included'])}")
    print(f"Columns (wide): {len(meta['columns_wide'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
