"""
Build a NeuroRehab-compatible CSV from OpenSim GUI outputs for arm26.

Inputs (typical):
- Motion file (.mot): paper_minjerk_10cycles.mot
- Static Optimization activation (.sto)

Outputs:
- gui_run_dataset.csv with both strict and functional EMG channel mappings.

Example:
    conda run -n biomech python Arm26/build_csv_from_gui.py \
      --run-dir Arm26/OutputReference/GUI_Run_001 \
      --motion Arm26/OutputReference/paper_minjerk_10cycles.mot \
    --activation Arm26/OutputReference/GUI_Run_001/static_opt/arm26_StaticOptimization_activation.sto \
      --out Arm26/OutputReference/GUI_Run_001/gui_run_dataset.csv

Paper-aligned extended output (adds RMS/VAR/CCI/NMI columns and uses measured
kinematics for theta_s/theta_e):
    conda run -n biomech python Arm26/build_csv_from_gui.py \
    --run-dir Arm26/OutputReference/GUI_Run_001 \
    --motion Arm26/OutputReference/paper_minjerk_10cycles.mot \
    --kinematics Arm26/OutputReference/GUI_Run_001/analyze/arm26_Kinematics_q.sto \
    --paper-features \
    --out Arm26/OutputReference/GUI_Run_001/gui_run_dataset.csv
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

SHOULDER_COORD = "r_shoulder_elev"
ELBOW_COORD = "r_elbow_flex"

STRICT_MAP = {
    "biceps": ["BIClong", "BICshort"],
    "triceps": ["TRIlong", "TRIlat", "TRImed"],
    "deltoid": ["DELT_ant"],
}

DEFAULT_ALPHA_S = 0.4
DEFAULT_ALPHA_E = 0.4
DEFAULT_BETA_S = 0.1
DEFAULT_BETA_E = 0.1
DEFAULT_EMAX_S = 30.0
DEFAULT_EMAX_E = 30.0

FUNCTIONAL_MAP = {
    "biceps": ["BIClong", "BICshort", "BRA", "BRD_hand"],
    "triceps": ["TRIlong", "TRIlat", "TRImed"],
    "deltoid": ["DELT_ant", "DELT_post", "PECT"],
}


def parse_opensim_table(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    end_idx = None
    metadata: Dict[str, str] = {}
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
        raise ValueError(f"No data after endheader in {path}")

    df = pd.read_csv(io.StringIO(body), sep=r"\s+", engine="python")
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df, metadata


def find_column(df: pd.DataFrame, target: str) -> str:
    cols = list(df.columns)

    # Exact
    if target in cols:
        return target

    # Case-insensitive exact
    lower = {c.lower(): c for c in cols}
    if target.lower() in lower:
        return lower[target.lower()]

    # Heuristic matches for OpenSim-style names
    candidates = []
    t = target.lower()
    for c in cols:
        lc = c.lower()
        if lc == t:
            candidates.append(c)
        elif lc.endswith("/" + t):
            candidates.append(c)
        elif lc.endswith("/" + t + "/activation"):
            candidates.append(c)
        elif lc.endswith("_" + t):
            candidates.append(c)
        elif t in lc and "activation" in lc:
            candidates.append(c)

    if candidates:
        return candidates[0]

    raise KeyError(f"Column for '{target}' not found. Available columns sample: {cols[:20]}")


def interp_series(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return np.interp(t_dst, t_src, y_src, left=y_src[0], right=y_src[-1])


def mean_muscles(act: pd.DataFrame, t_act: np.ndarray, muscles: List[str], t_ref: np.ndarray) -> np.ndarray:
    vals = []
    for m in muscles:
        col = find_column(act, m)
        vals.append(interp_series(t_act, act[col].to_numpy(dtype=float), t_ref))
    return np.mean(np.vstack(vals), axis=0)


def rolling_rms_var(signal: np.ndarray, window_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    s = pd.Series(signal)
    rms = np.sqrt(
        s.pow(2)
        .rolling(window=window_samples, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=float)
    )
    var = (
        s.rolling(window=window_samples, min_periods=1, center=True)
        .var(ddof=0)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return rms, var


def compute_cci(ag: np.ndarray, ant: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return 2.0 * np.minimum(ag, ant) / (ag + ant + eps)


def compute_nmi(
    cci_s: np.ndarray,
    cci_e: np.ndarray,
    theta_s: np.ndarray,
    theta_s_ref: np.ndarray,
    theta_e: np.ndarray,
    theta_e_ref: np.ndarray,
    alpha_s: float,
    alpha_e: float,
    beta_s: float,
    beta_e: float,
    emax_s: float,
    emax_e: float,
) -> np.ndarray:
    e_s = np.abs(theta_s - theta_s_ref) / (emax_s + 1e-6)
    e_e = np.abs(theta_e - theta_e_ref) / (emax_e + 1e-6)
    nmi = alpha_s * cci_s + alpha_e * cci_e + beta_s * e_s + beta_e * e_e
    return np.clip(nmi, 0.0, 1.0)


def build_dataset(
    motion_path: Path,
    activation_path: Path,
    kinematics_path: Optional[Path],
    out_path: Path,
    load_kg: float,
    condition: str,
    subject_id: int,
    trial_id: int,
    paper_features: bool,
    window_ms: int,
    alpha_s: float,
    alpha_e: float,
    beta_s: float,
    beta_e: float,
    emax_s: float,
    emax_e: float,
) -> None:
    mot, mot_meta = parse_opensim_table(motion_path)
    act, _ = parse_opensim_table(activation_path)

    t_ref = mot[find_column(mot, "time")].to_numpy(dtype=float)
    t_act = act[find_column(act, "time")].to_numpy(dtype=float)

    sh_col = find_column(mot, SHOULDER_COORD)
    el_col = find_column(mot, ELBOW_COORD)

    theta_s_ref = mot[sh_col].to_numpy(dtype=float)
    theta_e_ref = mot[el_col].to_numpy(dtype=float)

    in_degrees = mot_meta.get("indegrees", "yes").lower() == "yes"
    if not in_degrees:
        theta_s_ref = np.rad2deg(theta_s_ref)
        theta_e_ref = np.rad2deg(theta_e_ref)

    theta_s = theta_s_ref.copy()
    theta_e = theta_e_ref.copy()

    if kinematics_path is not None:
        kin, kin_meta = parse_opensim_table(kinematics_path)
        t_kin = kin[find_column(kin, "time")].to_numpy(dtype=float)
        kin_sh = kin[find_column(kin, SHOULDER_COORD)].to_numpy(dtype=float)
        kin_el = kin[find_column(kin, ELBOW_COORD)].to_numpy(dtype=float)

        kin_in_degrees = kin_meta.get("indegrees", "yes").lower() == "yes"
        if not kin_in_degrees:
            kin_sh = np.rad2deg(kin_sh)
            kin_el = np.rad2deg(kin_el)

        theta_s = interp_series(t_kin, kin_sh, t_ref)
        theta_e = interp_series(t_kin, kin_el, t_ref)

    emg_biceps_strict = mean_muscles(act, t_act, STRICT_MAP["biceps"], t_ref)
    emg_triceps_strict = mean_muscles(act, t_act, STRICT_MAP["triceps"], t_ref)
    emg_deltoid_strict = mean_muscles(act, t_act, STRICT_MAP["deltoid"], t_ref)

    emg_biceps_fn = mean_muscles(act, t_act, FUNCTIONAL_MAP["biceps"], t_ref)
    emg_triceps_fn = mean_muscles(act, t_act, FUNCTIONAL_MAP["triceps"], t_ref)
    emg_deltoid_fn = mean_muscles(act, t_act, FUNCTIONAL_MAP["deltoid"], t_ref)

    for arr in (
        emg_biceps_strict,
        emg_triceps_strict,
        emg_deltoid_strict,
        emg_biceps_fn,
        emg_triceps_fn,
        emg_deltoid_fn,
    ):
        np.clip(arr, 0.0, 1.0, out=arr)

    out = pd.DataFrame(
        {
            "time": t_ref,
            "theta_s": theta_s,
            "theta_e": theta_e,
            "theta_s_ref": theta_s_ref,
            "theta_e_ref": theta_e_ref,
            # Default channels are strict paper-faithful mapping.
            "emg_biceps": emg_biceps_strict,
            "emg_triceps": emg_triceps_strict,
            "emg_deltoid": emg_deltoid_strict,
            # Functional channels are kept for richer model training.
            "emg_biceps_fn": emg_biceps_fn,
            "emg_triceps_fn": emg_triceps_fn,
            "emg_deltoid_fn": emg_deltoid_fn,
            "load_kg": float(load_kg),
            "condition": str(condition),
            "trial_id": int(trial_id),
            "subject_id": int(subject_id),
        }
    )

    if paper_features:
        dt = float(np.median(np.diff(t_ref))) if t_ref.shape[0] > 1 else 0.01
        dt = max(dt, 1e-6)
        window_samples = max(2, int(round((window_ms / 1000.0) / dt)))

        rms_b, var_b = rolling_rms_var(emg_biceps_strict, window_samples)
        rms_t, var_t = rolling_rms_var(emg_triceps_strict, window_samples)
        rms_d, var_d = rolling_rms_var(emg_deltoid_strict, window_samples)

        cci_s = compute_cci(rms_d, rms_t)
        cci_e = compute_cci(rms_b, rms_t)
        nmi = compute_nmi(
            cci_s=cci_s,
            cci_e=cci_e,
            theta_s=theta_s,
            theta_s_ref=theta_s_ref,
            theta_e=theta_e,
            theta_e_ref=theta_e_ref,
            alpha_s=alpha_s,
            alpha_e=alpha_e,
            beta_s=beta_s,
            beta_e=beta_e,
            emax_s=emax_s,
            emax_e=emax_e,
        )

        out["emg_biceps_rms"] = rms_b
        out["emg_triceps_rms"] = rms_t
        out["emg_deltoid_rms"] = rms_d
        out["emg_biceps_var"] = var_b
        out["emg_triceps_var"] = var_t
        out["emg_deltoid_var"] = var_d
        out["cci_s"] = cci_s
        out["cci_e"] = cci_e
        out["nmi"] = nmi

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows -> {out_path}")


def autodetect_activation(run_dir: Path) -> Path:
    static_opt = run_dir / "static_opt"
    if static_opt.exists():
        hits = sorted(static_opt.glob("*StaticOptimization_activation.sto"))
        if hits:
            return hits[0]

    hits = sorted(run_dir.glob("**/*StaticOptimization_activation.sto"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        "Could not auto-detect StaticOptimization activation file under run_dir. "
        "Pass --activation explicitly."
    )


def autodetect_kinematics(run_dir: Path) -> Path:
    analyze_dir = run_dir / "analyze"
    if analyze_dir.exists():
        hits = sorted(analyze_dir.glob("*Kinematics_q.sto"))
        if hits:
            return hits[0]

    hits = sorted(run_dir.glob("**/*Kinematics_q.sto"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        "Could not auto-detect a kinematics coordinates file under run_dir/analyze. "
        "Pass --kinematics explicitly."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build merged CSV from OpenSim GUI outputs")
    p.add_argument("--run-dir", type=Path, default=ROOT / "OutputReference" / "GUI_Run_001")
    p.add_argument("--motion", type=Path, default=ROOT / "OutputReference" / "paper_minjerk_10cycles.mot")
    p.add_argument("--activation", type=Path, default=None)
    p.add_argument("--kinematics", type=Path, default=None)
    p.add_argument(
        "--auto-kinematics",
        action="store_true",
        help="Auto-detect analyze/*Kinematics_q.sto under --run-dir and use it for theta_s/theta_e.",
    )
    p.add_argument(
        "--paper-features",
        action="store_true",
        help="Add paper-aligned columns: RMS/VAR EMG, CCI, and NMI.",
    )
    p.add_argument("--window-ms", type=int, default=200)
    p.add_argument("--alpha-s", type=float, default=DEFAULT_ALPHA_S)
    p.add_argument("--alpha-e", type=float, default=DEFAULT_ALPHA_E)
    p.add_argument("--beta-s", type=float, default=DEFAULT_BETA_S)
    p.add_argument("--beta-e", type=float, default=DEFAULT_BETA_E)
    p.add_argument("--emax-s", type=float, default=DEFAULT_EMAX_S)
    p.add_argument("--emax-e", type=float, default=DEFAULT_EMAX_E)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--condition", type=str, default="loaded_reaching")
    p.add_argument("--load-kg", type=float, default=2.0)
    p.add_argument("--subject-id", type=int, default=1)
    p.add_argument("--trial-id", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    run_dir = args.run_dir
    motion = args.motion
    if args.activation is not None and args.activation.exists():
        activation = args.activation
    elif args.activation is not None and not args.activation.exists():
        activation = autodetect_activation(run_dir)
        print(f"Activation file not found: {args.activation}")
        print(f"Using auto-detected activation file: {activation}")
    else:
        activation = autodetect_activation(run_dir)

    if args.kinematics is not None:
        if not args.kinematics.exists():
            raise FileNotFoundError(f"Kinematics file not found: {args.kinematics}")
        kinematics = args.kinematics
    elif args.auto_kinematics:
        kinematics = autodetect_kinematics(run_dir)
        print(f"Using auto-detected kinematics file: {kinematics}")
    else:
        kinematics = None

    out = args.out or (run_dir / "gui_run_dataset.csv")

    build_dataset(
        motion_path=motion,
        activation_path=activation,
        kinematics_path=kinematics,
        out_path=out,
        load_kg=args.load_kg,
        condition=args.condition,
        subject_id=args.subject_id,
        trial_id=args.trial_id,
        paper_features=args.paper_features,
        window_ms=args.window_ms,
        alpha_s=args.alpha_s,
        alpha_e=args.alpha_e,
        beta_s=args.beta_s,
        beta_e=args.beta_e,
        emax_s=args.emax_s,
        emax_e=args.emax_e,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
