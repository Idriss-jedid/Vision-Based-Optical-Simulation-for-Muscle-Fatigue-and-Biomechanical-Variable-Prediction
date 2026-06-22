"""
Build a synthetic single-camera MP4 from the Stage-2 GUI dataset.

Pipeline:
3D biomechanical coordinates -> pinhole projection -> optional Gaussian noise -> MP4.

Expected input (default):
- Data/GUI_Run_001/gui_project_dataset_minimal.csv

Default outputs:
- Data/GUI_Run_001/vision_video_v1.mp4
- Data/GUI_Run_001/vision_video_v1_meta.json

Usage:
    conda run -n biomech python Code/build_vision_video_v1.py \
      --run-dir Data/GUI_Run_001
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import pandas as pd

from build_vision_dataset_v1 import DEFAULT_POINTS, add_gaussian_noise, project_points


POINT_LABELS = {
    "r_humerus": "shoulder",
    "r_ulna_radius_hand": "hand",
    "load_2kg": "load",
    "center_of_mass": "COM",
}

POINT_COLORS = {
    "r_humerus": "#3b82f6",
    "r_ulna_radius_hand": "#f59e0b",
    "load_2kg": "#ef4444",
    "center_of_mass": "#22c55e",
}


def _available_points(df: pd.DataFrame, requested: List[str]) -> List[str]:
    available: List[str] = []
    for point in requested:
        required = [f"{point}_X", f"{point}_Y", f"{point}_Z"]
        if all(col in df.columns for col in required):
            available.append(point)
    return available


def _sample_indices(times: np.ndarray, fps: int) -> np.ndarray:
    if len(times) < 2:
        return np.array([0], dtype=int)

    duration = max(float(times[-1] - times[0]), 0.0)
    frame_count = max(2, int(round(duration * fps)) + 1)
    sample = np.linspace(0, len(times) - 1, frame_count)
    return np.unique(np.round(sample).astype(int))


def _connection_pairs(points: List[str]) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    if "r_humerus" in points and "r_ulna_radius_hand" in points:
        pairs.append(("r_humerus", "r_ulna_radius_hand"))
    if "r_ulna_radius_hand" in points and "load_2kg" in points:
        pairs.append(("r_ulna_radius_hand", "load_2kg"))
    return pairs


def _make_2d_coordinates(
    df: pd.DataFrame,
    points: List[str],
    f: float,
    width: int,
    height: int,
    cx: float | None,
    cy: float | None,
    depth_eps: float,
    noisy: bool,
    sigma_px: float,
    seed: int,
) -> tuple[pd.DataFrame, List[str]]:
    cx_eff = float(width) / 2.0 if cx is None else float(cx)
    cy_eff = float(height) / 2.0 if cy is None else float(cy)

    points_used = _available_points(df, points)
    if not points_used:
        raise ValueError("No requested 3D points were found in the input CSV.")

    projected = project_points(df, points_used, f=f, cx=cx_eff, cy=cy_eff, depth_eps=depth_eps)

    coords = pd.DataFrame(index=df.index)
    if noisy:
        noisy_df = add_gaussian_noise(projected, sigma_px=sigma_px, seed=seed)
        for point in points_used:
            coords[f"{point}_u"] = noisy_df[f"{point}_u_noisy"]
            coords[f"{point}_v"] = noisy_df[f"{point}_v_noisy"]
    else:
        for point in points_used:
            coords[f"{point}_u"] = projected[f"{point}_u"]
            coords[f"{point}_v"] = projected[f"{point}_v"]

    return coords, points_used


def build_video(
    input_csv: Path,
    output_mp4: Path,
    output_meta: Path,
    points: List[str],
    f: float,
    width: int,
    height: int,
    cx: float | None,
    cy: float | None,
    depth_eps: float,
    noisy: bool,
    sigma_px: float,
    seed: int,
    fps: int,
    dpi: int,
    trail_seconds: float,
) -> Dict[str, object]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH. It is required to encode MP4 output.")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "time" not in df.columns:
        raise KeyError(f"Missing 'time' column in {input_csv}")

    coords, points_used = _make_2d_coordinates(
        df=df,
        points=points,
        f=f,
        width=width,
        height=height,
        cx=cx,
        cy=cy,
        depth_eps=depth_eps,
        noisy=noisy,
        sigma_px=sigma_px,
        seed=seed,
    )

    times = df["time"].to_numpy(dtype=float)
    frame_indices = _sample_indices(times, fps=fps)
    trail_frames = max(1, int(round(trail_seconds * fps)))
    connections = _connection_pairs(points_used)
    trail_points = [p for p in ("r_ulna_radius_hand", "load_2kg") if p in points_used]

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#ffffff")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#ffffff")

    point_artists: Dict[str, object] = {}
    label_artists: Dict[str, object] = {}
    for point in points_used:
        (artist,) = ax.plot([], [], "o", color=POINT_COLORS.get(point, "#111827"), markersize=8)
        label = ax.text(
            0,
            0,
            POINT_LABELS.get(point, point),
            fontsize=10,
            color=POINT_COLORS.get(point, "#111827"),
            ha="left",
            va="bottom",
            fontweight="bold",
        )
        point_artists[point] = artist
        label_artists[point] = label

    connection_artists: Dict[tuple[str, str], object] = {}
    for start, end in connections:
        (artist,) = ax.plot([], [], color="#9ca3af", linewidth=2.0, alpha=0.9)
        connection_artists[(start, end)] = artist

    trail_artists: Dict[str, object] = {}
    for point in trail_points:
        (artist,) = ax.plot([], [], color=POINT_COLORS.get(point, "#111827"), linewidth=2.0, alpha=0.28)
        trail_artists[point] = artist

    title_text = ax.text(
        0.02,
        0.98,
        "Synthetic single-camera vision video",
        transform=ax.transAxes,
        fontsize=14,
        color="#111827",
        ha="left",
        va="top",
        fontweight="bold",
    )
    time_text = ax.text(
        0.02,
        0.93,
        "",
        transform=ax.transAxes,
        fontsize=11,
        color="#374151",
        ha="left",
        va="top",
    )
    angle_text = ax.text(
        0.02,
        0.89,
        "",
        transform=ax.transAxes,
        fontsize=11,
        color="#374151",
        ha="left",
        va="top",
    )
    mode_text = ax.text(
        0.98,
        0.98,
        f"mode: {'noisy' if noisy else 'clean'}",
        transform=ax.transAxes,
        fontsize=11,
        color="#374151",
        ha="right",
        va="top",
    )

    def update(frame_number: int) -> None:
        row_idx = int(frame_indices[frame_number])

        for point in points_used:
            x = float(coords.at[row_idx, f"{point}_u"])
            y = float(coords.at[row_idx, f"{point}_v"])
            if np.isfinite(x) and np.isfinite(y):
                point_artists[point].set_data([x], [y])
                label_artists[point].set_position((x + 10.0, y - 10.0))
                label_artists[point].set_alpha(1.0)
            else:
                point_artists[point].set_data([], [])
                label_artists[point].set_alpha(0.0)

        for (start, end), artist in connection_artists.items():
            x1 = float(coords.at[row_idx, f"{start}_u"])
            y1 = float(coords.at[row_idx, f"{start}_v"])
            x2 = float(coords.at[row_idx, f"{end}_u"])
            y2 = float(coords.at[row_idx, f"{end}_v"])
            if all(np.isfinite(v) for v in (x1, y1, x2, y2)):
                artist.set_data([x1, x2], [y1, y2])
            else:
                artist.set_data([], [])

        window_start = max(0, frame_number - trail_frames)
        trail_idx = frame_indices[window_start : frame_number + 1]
        for point, artist in trail_artists.items():
            x_hist = coords.loc[trail_idx, f"{point}_u"].to_numpy(dtype=float)
            y_hist = coords.loc[trail_idx, f"{point}_v"].to_numpy(dtype=float)
            artist.set_data(x_hist, y_hist)

        rel_time = float(times[row_idx] - times[0])
        time_text.set_text(f"t = {rel_time:0.2f} s")

        if "q_elbow" in df.columns:
            elbow_deg = float(np.rad2deg(df.at[row_idx, "q_elbow"]))
            angle_text.set_text(f"elbow = {elbow_deg:0.1f} deg")
        else:
            angle_text.set_text("")

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000.0 / fps, blit=False)

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=4000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    animation.save(str(output_mp4), writer=writer, dpi=dpi)
    plt.close(fig)

    meta: Dict[str, object] = {
        "input_csv": str(input_csv),
        "output_mp4": str(output_mp4),
        "output_meta": str(output_meta),
        "rows_input": int(len(df)),
        "frames_rendered": int(len(frame_indices)),
        "duration_seconds": float(max(times[-1] - times[0], 0.0)) if len(times) > 1 else 0.0,
        "fps": int(fps),
        "image_width": int(width),
        "image_height": int(height),
        "dpi": int(dpi),
        "points_requested": points,
        "points_used": points_used,
        "mode": "noisy" if noisy else "clean",
        "noise_sigma_px": float(sigma_px),
        "noise_seed": int(seed),
        "trail_seconds": float(trail_seconds),
        "camera": {
            "f": float(f),
            "cx": float(width) / 2.0 if cx is None else float(cx),
            "cy": float(height) / 2.0 if cy is None else float(cy),
            "depth_eps": float(depth_eps),
        },
        "connections": [[start, end] for start, end in connections],
    }
    output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_run_dir = root / "Data" / "GUI_Run_001"

    parser = argparse.ArgumentParser(description="Build a synthetic single-camera MP4 from OpenSim-derived 3D data")
    parser.add_argument("--run-dir", type=Path, default=default_run_dir, help="Path to GUI run folder")
    parser.add_argument("--input-csv", type=Path, default=None, help="Input merged CSV (defaults to gui_project_dataset_minimal.csv)")
    parser.add_argument("--out-mp4", type=Path, default=None, help="Output MP4 path")
    parser.add_argument("--out-meta", type=Path, default=None, help="Output metadata JSON path")

    parser.add_argument("--points", nargs="+", default=DEFAULT_POINTS, help="3D points to project and render")
    parser.add_argument("--f", type=float, default=1200.0, help="Focal length in pixels")
    parser.add_argument("--image-width", type=int, default=1280, help="Image width in pixels")
    parser.add_argument("--image-height", type=int, default=720, help="Image height in pixels")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x (pixels), default width/2")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y (pixels), default height/2")
    parser.add_argument("--depth-eps", type=float, default=1e-6, help="Minimum absolute depth to avoid divide by zero")

    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate")
    parser.add_argument("--dpi", type=int, default=100, help="Render DPI")
    parser.add_argument("--trail-seconds", type=float, default=0.4, help="History trail length in seconds")

    parser.add_argument("--clean", action="store_true", help="Render clean projected points instead of noisy points")
    parser.add_argument("--sigma-px", type=float, default=2.0, help="Gaussian noise std in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir: Path = args.run_dir

    input_csv = args.input_csv or (run_dir / "gui_project_dataset_minimal.csv")
    output_mp4 = args.out_mp4 or (run_dir / "vision_video_v1.mp4")
    output_meta = args.out_meta or (run_dir / "vision_video_v1_meta.json")

    meta = build_video(
        input_csv=input_csv,
        output_mp4=output_mp4,
        output_meta=output_meta,
        points=args.points,
        f=args.f,
        width=args.image_width,
        height=args.image_height,
        cx=args.cx,
        cy=args.cy,
        depth_eps=args.depth_eps,
        noisy=not args.clean,
        sigma_px=args.sigma_px,
        seed=args.seed,
        fps=args.fps,
        dpi=args.dpi,
        trail_seconds=args.trail_seconds,
    )

    print(f"Saved MP4: {output_mp4}")
    print(f"Saved metadata: {output_meta}")
    print(f"Frames rendered: {meta['frames_rendered']}")
    print(f"Duration (s): {meta['duration_seconds']:.2f}")
    print(f"Points used: {', '.join(meta['points_used'])}")
    print(f"Mode: {meta['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())