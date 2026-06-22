"""
Arm26-only OpenSim pipeline aligned to:
- Opensim Basics Tutorial workflow concepts (Scale/IK/ID/RRA/CMC/FD/Analyze)
- Neuro-adaptive predictive control paper structure (NMI + fuzzy load + SVR-LSTM + QSVM + adaptive admittance)

This implementation intentionally uses only synthetic data generated from the current model:
    Arm26/arm26_paper_loaded.osim

Main stages:
1) Generate OpenSim-only synthetic dataset from arm26_paper_loaded.osim.
2) Preprocess EMG-like channels (RMS/variance windows).
3) Compute CCI + NMI.
4) Train Hybrid SVR-LSTM trajectory predictor.
5) Train QSVM intention decoder (falls back to classical SVM if needed).
6) Run closed-loop benchmarks (C1 no assist, C2 fixed admittance, C3 proposed).
7) Save metrics, stats, and plots.

Example:
    conda run -n biomech python Arm26/pipeline_arm26_full.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import opensim as osim
except ImportError as exc:
    raise SystemExit("OpenSim Python bindings are required. Install opensim in the selected environment.") from exc


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
NEUROREHAB_ROOT = WORKSPACE / "NeuroRehab"
if str(NEUROREHAB_ROOT) not in sys.path:
    sys.path.insert(0, str(NEUROREHAB_ROOT))

# Reuse the paper-like modules already available in this repository.
from src.controller import (  # type: ignore
    AdaptiveAdmittanceController,
    FixedAdmittanceController,
    NoAssistController,
)
from src.fuzzy_load import FuzzyLoadEstimator  # type: ignore
from src.predictor import HybridPredictor  # type: ignore
from src.qsvm_intention import (  # type: ignore
    INTENT_EXTENSION,
    INTENT_FLEXION,
    INTENT_HOLD,
    QSVMIntentionDecoder,
)


MODEL_DEFAULT = ROOT / "arm26_paper_loaded.osim"
DATA_DIR_DEFAULT = ROOT / "PipelineData"
RESULTS_DIR_DEFAULT = ROOT / "PipelineResults"

SHOULDER_COORD = "r_shoulder_elev"
ELBOW_COORD = "r_elbow_flex"

BICEPS_MUSCLES = ("BIClong", "BICshort", "BRA", "BRD_hand")
TRICEPS_MUSCLES = ("TRIlong", "TRIlat", "TRImed")
DELTOID_MUSCLES = ("DELT_ant", "DELT_post", "PECT")

ALL_REQUIRED_MUSCLES = tuple(sorted(set(BICEPS_MUSCLES + TRICEPS_MUSCLES + DELTOID_MUSCLES + ("LAT",))))

FEATURE_COLS = [
    "emg_biceps_rms",
    "emg_triceps_rms",
    "emg_deltoid_rms",
    "emg_biceps_var",
    "emg_triceps_var",
    "emg_deltoid_var",
    "theta_s",
    "theta_e",
]

INTENT_FEATURE_COLS = [
    "emg_biceps_rms",
    "emg_triceps_rms",
    "emg_deltoid_rms",
    "emg_biceps_var",
    "emg_triceps_var",
    "emg_deltoid_var",
]

TARGET_COLS = ["theta_s_ref", "theta_e_ref"]


@dataclass
class PipelineConfig:
    model_path: Path
    data_dir: Path
    results_dir: Path
    fs_control: int = 100
    dt: float = 0.01
    # Task profile
    reach_s: float = 1.5
    hold_s: float = 1.0
    return_s: float = 1.5
    rest_s: float = 2.0
    n_cycles: int = 5
    # Dataset
    n_subjects: int = 8
    trials_per_condition: int = 2
    conditions: Tuple[str, ...] = ("nominal", "fatigue", "spasticity", "fatigue_spasticity", "load")
    # Evaluation
    benchmark_runs: int = 8
    random_seed: int = 42
    # Features
    window_ms: int = 200
    overlap: float = 0.5
    # NMI
    alpha_s: float = 0.4
    alpha_e: float = 0.4
    beta_s: float = 0.1
    beta_e: float = 0.1
    emax_s: float = 30.0
    emax_e: float = 30.0
    # Runtime perturbation
    runtime_noise_std: float = 0.01


class Arm26Plant:
    """OpenSim-backed plant for arm26_paper_loaded.osim."""

    def __init__(self, model_path: Path, integrator_accuracy: float = 1e-4):
        self.model = osim.Model(str(model_path))
        self.integrator_accuracy = integrator_accuracy

        coords = self.model.getCoordinateSet()
        self.coord_sh = coords.get(SHOULDER_COORD)
        self.coord_el = coords.get(ELBOW_COORD)

        forces = self.model.getForceSet()
        self.assist_sh = osim.ScalarActuator.safeDownCast(forces.get("shoulder_assist"))
        self.assist_el = osim.ScalarActuator.safeDownCast(forces.get("elbow_assist"))
        if self.assist_sh is None or self.assist_el is None:
            raise RuntimeError("Expected scalar actuators shoulder_assist and elbow_assist in arm26_paper_loaded.osim")

        muscles = self.model.getMuscles()
        self.muscle_by_name: Dict[str, osim.Muscle] = {}
        for i in range(muscles.getSize()):
            m = muscles.get(i)
            self.muscle_by_name[m.getName()] = m

        missing = [m for m in ALL_REQUIRED_MUSCLES if m not in self.muscle_by_name]
        if missing:
            raise RuntimeError(f"Missing required muscles in model: {missing}")

        self.state = None
        self.manager = None
        self.reset(q0_rad=np.array([0.0, 0.0]))

    def reset(self, q0_rad: np.ndarray) -> None:
        self.state = self.model.initSystem()
        self.coord_sh.setValue(self.state, float(q0_rad[0]))
        self.coord_el.setValue(self.state, float(q0_rad[1]))
        self.coord_sh.setSpeedValue(self.state, 0.0)
        self.coord_el.setSpeedValue(self.state, 0.0)

        self.model.equilibrateMuscles(self.state)

        self.manager = osim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def read_state_deg(self) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([
            self.coord_sh.getValue(self.state),
            self.coord_el.getValue(self.state),
        ])
        dq = np.array([
            self.coord_sh.getSpeedValue(self.state),
            self.coord_el.getSpeedValue(self.state),
        ])
        return np.rad2deg(q), np.rad2deg(dq)

    def set_assist_torque(self, tau_sh: float, tau_el: float) -> None:
        self.assist_sh.overrideActuation(self.state, True)
        self.assist_el.overrideActuation(self.state, True)
        self.assist_sh.setOverrideActuation(self.state, float(tau_sh))
        self.assist_el.setOverrideActuation(self.state, float(tau_el))

    def set_channel_excitations(
        self,
        emg_biceps: float,
        emg_triceps: float,
        emg_deltoid: float,
        channel_gain: Dict[str, float],
    ) -> None:
        b = float(np.clip(emg_biceps, 0.01, 0.95)) * channel_gain["biceps"]
        t = float(np.clip(emg_triceps, 0.01, 0.95)) * channel_gain["triceps"]
        d = float(np.clip(emg_deltoid, 0.01, 0.95)) * channel_gain["deltoid"]

        excitations = {
            "BIClong": b,
            "BICshort": 0.95 * b,
            "BRA": 0.85 * b,
            "BRD_hand": 0.80 * b,
            "TRIlong": t,
            "TRIlat": 0.92 * t,
            "TRImed": 0.90 * t,
            "DELT_ant": d,
            "DELT_post": 0.60 * d + 0.20 * t,
            "PECT": 0.75 * d,
            "LAT": 0.65 * t + 0.15 * d,
        }

        for name, value in excitations.items():
            muscle = self.muscle_by_name[name]
            exc = float(np.clip(value, 0.01, 0.95))
            try:
                muscle.setExcitation(self.state, exc)
            except Exception:
                muscle.setActivation(self.state, exc)

    def get_emg_channels(self) -> Tuple[float, float, float]:
        def avg(names: Tuple[str, ...]) -> float:
            vals = []
            for n in names:
                m = self.muscle_by_name[n]
                # Activation readout can fail if OpenSim state is not realized to dynamics;
                # fall back to excitation so the pipeline remains numerically stable.
                try:
                    self.model.realizeDynamics(self.state)
                    val = float(m.getActivation(self.state))
                except Exception:
                    try:
                        val = float(m.getExcitation(self.state))
                    except Exception:
                        val = 0.01
                vals.append(float(np.clip(val, 0.0, 1.0)))
            return float(np.mean(vals))

        return avg(BICEPS_MUSCLES), avg(TRICEPS_MUSCLES), avg(DELTOID_MUSCLES)

    def step(self, dt: float) -> None:
        t_target = self.state.getTime() + float(dt)
        try:
            self.state = self.manager.integrate(t_target)
        except Exception:
            # Keep simulation alive on occasional stiff integration steps.
            self.state.setTime(t_target)


def min_jerk_scalar(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def _write_mot_from_log(df: pd.DataFrame, out_path: Path, label: str = "controller_predicted") -> Path:
    """
    Export a controller log DataFrame to an OpenSim .mot motion file so the
    user can replay the resulting joint trajectory in the GUI.

    Expects columns: 'time', 'theta_s', 'theta_e' (degrees).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t  = df["time"].to_numpy()
    qs = df["theta_s"].to_numpy()
    qe = df["theta_e"].to_numpy()
    n_rows = len(t)

    header_lines = [
        f"{label}",
        "version=1",
        f"nRows={n_rows}",
        "nColumns=3",
        "inDegrees=yes",
        "endheader",
        "time\tr_shoulder_elev\tr_elbow_flex",
    ]
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header_lines) + "\n")
        for i in range(n_rows):
            fh.write(f"{t[i]:.6f}\t{qs[i]:.6f}\t{qe[i]:.6f}\n")

    return out_path


def build_task_reference(
    cfg: PipelineConfig,
    shoulder_target_deg: float,
    elbow_target_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cycle_time = cfg.reach_s + cfg.hold_s + cfg.return_s + cfg.rest_s
    total_time = cfg.n_cycles * cycle_time
    t = np.arange(0.0, total_time + 0.5 * cfg.dt, cfg.dt)

    sh = np.zeros_like(t)
    el = np.zeros_like(t)

    for c in range(cfg.n_cycles):
        t0 = c * cycle_time
        t1 = t0 + cfg.reach_s
        t2 = t1 + cfg.hold_s
        t3 = t2 + cfg.return_s

        m_reach = (t >= t0) & (t < t1)
        if np.any(m_reach):
            s = (t[m_reach] - t0) / cfg.reach_s
            w = min_jerk_scalar(s)
            sh[m_reach] = shoulder_target_deg * w
            el[m_reach] = elbow_target_deg * w

        m_hold = (t >= t1) & (t < t2)
        sh[m_hold] = shoulder_target_deg
        el[m_hold] = elbow_target_deg

        m_ret = (t >= t2) & (t < t3)
        if np.any(m_ret):
            s = (t[m_ret] - t2) / cfg.return_s
            w = min_jerk_scalar(s)
            sh[m_ret] = shoulder_target_deg * (1.0 - w)
            el[m_ret] = elbow_target_deg * (1.0 - w)

    return t, sh, el


def fatigue_efficiency(t: float, total_time: float, onset_frac: float = 0.4, tau_s: float = 8.0, min_eff: float = 0.45) -> float:
    onset = onset_frac * total_time
    if t <= onset:
        return 1.0
    decay = math.exp(-(t - onset) / tau_s)
    return min_eff + (1.0 - min_eff) * decay


def make_spastic_series(
    n_steps: int,
    rng: np.random.Generator,
    bursts: int,
    amp_low: float,
    amp_high: float,
    dur_min_steps: int,
    dur_max_steps: int,
) -> np.ndarray:
    out = np.zeros(n_steps, dtype=float)
    for _ in range(bursts):
        center = int(rng.integers(0, n_steps))
        dur = int(rng.integers(dur_min_steps, dur_max_steps + 1))
        amp = float(rng.uniform(amp_low, amp_high))

        i0 = max(0, center - dur // 2)
        i1 = min(n_steps, center + dur // 2 + 1)
        idx = np.arange(i0, i1)
        sigma = max(dur / 6.0, 1.0)
        bump = amp * np.exp(-0.5 * ((idx - center) / sigma) ** 2)
        out[idx] += bump
    return out


def compute_cci(ag: float, ant: float, eps: float = 1e-6) -> float:
    return float(2.0 * min(ag, ant) / (ag + ant + eps))


def compute_nmi(
    cci_s: float,
    cci_e: float,
    theta_s: float,
    theta_s_ref: float,
    theta_e: float,
    theta_e_ref: float,
    cfg: PipelineConfig,
) -> float:
    e_s = abs(theta_s - theta_s_ref) / (cfg.emax_s + 1e-6)
    e_e = abs(theta_e - theta_e_ref) / (cfg.emax_e + 1e-6)
    nmi = cfg.alpha_s * cci_s + cfg.alpha_e * cci_e + cfg.beta_s * e_s + cfg.beta_e * e_e
    return float(np.clip(nmi, 0.0, 1.0))


def derive_intention_from_ref(theta_e_ref: np.ndarray, dt: float, threshold_deg_s: float = 5.0) -> np.ndarray:
    vel = np.gradient(theta_e_ref, dt)
    y = np.full(theta_e_ref.shape[0], INTENT_HOLD, dtype=int)
    y[vel > threshold_deg_s] = INTENT_FLEXION
    y[vel < -threshold_deg_s] = INTENT_EXTENSION
    return y


def generate_single_trial(
    cfg: PipelineConfig,
    subject_id: int,
    trial_id: int,
    condition: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    plant = Arm26Plant(cfg.model_path)

    shoulder_target = float(np.clip(60.0 * (1.0 + 0.12 * rng.normal()), 40.0, 85.0))
    elbow_target = float(np.clip(80.0 * (1.0 + 0.12 * rng.normal()), 50.0, 110.0))

    t, sh_ref_deg, el_ref_deg = build_task_reference(cfg, shoulder_target, elbow_target)
    n_steps = len(t)

    total_time = float(t[-1]) if len(t) > 0 else 1.0
    load_kg = float(rng.uniform(0.0, 2.0)) if condition == "load" else 0.0

    # Subject-dependent channel sensitivity.
    channel_gain = {
        "biceps": float(np.clip(1.0 + 0.10 * rng.normal(), 0.75, 1.25)),
        "triceps": float(np.clip(1.0 + 0.10 * rng.normal(), 0.75, 1.25)),
        "deltoid": float(np.clip(1.0 + 0.10 * rng.normal(), 0.75, 1.25)),
    }

    fatigue_on = condition in ("fatigue", "fatigue_spasticity")
    spastic_on = condition in ("spasticity", "fatigue_spasticity")

    spastic_b = np.zeros(n_steps)
    spastic_d = np.zeros(n_steps)
    if spastic_on:
        bursts = max(2, int(total_time // 3.0))
        spastic_b = make_spastic_series(
            n_steps,
            rng,
            bursts=bursts,
            amp_low=0.05,
            amp_high=0.25,
            dur_min_steps=8,
            dur_max_steps=30,
        )
        spastic_d = make_spastic_series(
            n_steps,
            rng,
            bursts=bursts,
            amp_low=0.04,
            amp_high=0.20,
            dur_min_steps=8,
            dur_max_steps=30,
        )

    q0_rad = np.deg2rad(np.array([sh_ref_deg[0], el_ref_deg[0]]))
    plant.reset(q0_rad=q0_rad)

    rows: List[Dict[str, float]] = []

    for i in range(n_steps):
        ti = float(t[i])
        q_deg, dq_deg = plant.read_state_deg()

        sh_ref = float(sh_ref_deg[i])
        el_ref = float(el_ref_deg[i])

        err_sh_rad = math.radians(sh_ref - q_deg[0])
        err_el_rad = math.radians(el_ref - q_deg[1])
        dq_sh_rad = math.radians(dq_deg[0])
        dq_el_rad = math.radians(dq_deg[1])

        # Generation-time tracking torque to create rich synthetic trajectories.
        tau_sh = float(np.clip(45.0 * err_sh_rad - 3.2 * dq_sh_rad, -22.0, 22.0))
        tau_el = float(np.clip(38.0 * err_el_rad - 2.8 * dq_el_rad, -16.0, 16.0))

        # Additional virtual load demand in the load condition (paper-like 0-2 kg context).
        tau_el += 0.35 * load_kg * 9.81 * 0.30

        b_cmd = 0.03 + 0.95 * max(err_el_rad, 0.0) + 0.20 * max(err_sh_rad, 0.0) + 0.04 * abs(dq_el_rad)
        t_cmd = 0.03 + 0.95 * max(-err_el_rad, 0.0) + 0.15 * max(-err_sh_rad, 0.0) + 0.04 * abs(dq_el_rad)
        d_cmd = 0.03 + 0.85 * max(err_sh_rad, 0.0) + 0.20 * abs(err_el_rad) + 0.04 * abs(dq_sh_rad)

        if fatigue_on:
            eff = fatigue_efficiency(ti, total_time)
            floor = (1.0 - eff) * 0.03
            b_cmd = b_cmd * eff + floor
            t_cmd = t_cmd * eff + floor
            d_cmd = d_cmd * eff + floor

        b_cmd += spastic_b[i]
        d_cmd += spastic_d[i]

        noise_sigma = 0.015 if condition == "nominal" else 0.025
        b_cmd += float(rng.normal(0.0, noise_sigma))
        t_cmd += float(rng.normal(0.0, noise_sigma))
        d_cmd += float(rng.normal(0.0, noise_sigma))

        b_cmd = float(np.clip(b_cmd, 0.01, 0.95))
        t_cmd = float(np.clip(t_cmd, 0.01, 0.95))
        d_cmd = float(np.clip(d_cmd, 0.01, 0.95))

        plant.set_channel_excitations(b_cmd, t_cmd, d_cmd, channel_gain)
        plant.set_assist_torque(tau_sh, tau_el)

        emg_b, emg_t, emg_d = plant.get_emg_channels()

        rows.append(
            {
                "time": ti,
                "theta_s": float(q_deg[0]),
                "theta_e": float(q_deg[1]),
                "dtheta_s": float(dq_deg[0]),
                "dtheta_e": float(dq_deg[1]),
                "theta_s_ref": sh_ref,
                "theta_e_ref": el_ref,
                "emg_biceps": float(np.clip(emg_b, 0.0, 1.0)),
                "emg_triceps": float(np.clip(emg_t, 0.0, 1.0)),
                "emg_deltoid": float(np.clip(emg_d, 0.0, 1.0)),
                "exc_biceps_cmd": b_cmd,
                "exc_triceps_cmd": t_cmd,
                "exc_deltoid_cmd": d_cmd,
                "tau_sh_cmd": tau_sh,
                "tau_el_cmd": tau_el,
                "load_kg": load_kg,
                "subject_id": subject_id,
                "trial_id": trial_id,
                "condition": condition,
            }
        )

        if i < n_steps - 1:
            plant.step(cfg.dt)

    return pd.DataFrame(rows)


def generate_dataset(cfg: PipelineConfig, force_regenerate: bool) -> pd.DataFrame:
    data_trials = cfg.data_dir / "trials"
    data_trials.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.data_dir / "manifest.csv"

    if manifest_path.exists() and not force_regenerate:
        return pd.read_csv(manifest_path)

    records: List[Dict[str, object]] = []

    for sid in range(1, cfg.n_subjects + 1):
        for cond in cfg.conditions:
            for tr in range(cfg.trials_per_condition):
                seed = cfg.random_seed + sid * 10000 + tr * 100 + abs(hash(cond)) % 97
                df = generate_single_trial(cfg, sid, tr, cond, seed)

                rel = Path(f"trials/subj_{sid:03d}/trial_{tr:02d}_{cond}.csv")
                out = cfg.data_dir / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, index=False)

                records.append(
                    {
                        "subject_id": sid,
                        "trial_id": tr,
                        "condition": cond,
                        "seed": seed,
                        "file": str(rel).replace("\\", "/"),
                    }
                )

    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def window_trial_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    fs = cfg.fs_control
    win = max(2, int(round(cfg.window_ms * fs / 1000.0)))
    step = max(1, int(round(win * (1.0 - cfg.overlap))))

    y_int = derive_intention_from_ref(df["theta_e_ref"].to_numpy(), cfg.dt)

    rows = []
    eb = df["emg_biceps"].to_numpy()
    et = df["emg_triceps"].to_numpy()
    ed = df["emg_deltoid"].to_numpy()

    for start in range(0, len(df) - win + 1, step):
        end = start + win
        mid = start + win // 2

        seg_b = eb[start:end]
        seg_t = et[start:end]
        seg_d = ed[start:end]

        rms_b = float(np.sqrt(np.mean(seg_b**2) + 1e-12))
        rms_t = float(np.sqrt(np.mean(seg_t**2) + 1e-12))
        rms_d = float(np.sqrt(np.mean(seg_d**2) + 1e-12))

        var_b = float(np.var(seg_b))
        var_t = float(np.var(seg_t))
        var_d = float(np.var(seg_d))

        theta_s = float(df["theta_s"].iloc[mid])
        theta_e = float(df["theta_e"].iloc[mid])
        theta_s_ref = float(df["theta_s_ref"].iloc[mid])
        theta_e_ref = float(df["theta_e_ref"].iloc[mid])

        cci_s = compute_cci(rms_d, rms_t)
        cci_e = compute_cci(rms_b, rms_t)
        nmi = compute_nmi(cci_s, cci_e, theta_s, theta_s_ref, theta_e, theta_e_ref, cfg)

        rows.append(
            {
                "time": float(df["time"].iloc[mid]),
                "subject_id": int(df["subject_id"].iloc[mid]),
                "trial_id": int(df["trial_id"].iloc[mid]),
                "condition": str(df["condition"].iloc[mid]),
                "load_kg": float(df["load_kg"].iloc[mid]),
                "theta_s": theta_s,
                "theta_e": theta_e,
                "theta_s_ref": theta_s_ref,
                "theta_e_ref": theta_e_ref,
                "emg_biceps_rms": rms_b,
                "emg_triceps_rms": rms_t,
                "emg_deltoid_rms": rms_d,
                "emg_biceps_var": var_b,
                "emg_triceps_var": var_t,
                "emg_deltoid_var": var_d,
                "cci_s": cci_s,
                "cci_e": cci_e,
                "nmi": nmi,
                "intent": int(y_int[mid]),
            }
        )

    return pd.DataFrame(rows)


def build_feature_table(cfg: PipelineConfig, manifest: pd.DataFrame, force_recompute: bool) -> pd.DataFrame:
    out_path = cfg.data_dir / "features.csv"
    if out_path.exists() and not force_recompute:
        return pd.read_csv(out_path)

    all_rows = []
    for _, row in manifest.iterrows():
        trial_path = cfg.data_dir / str(row["file"])
        trial_df = pd.read_csv(trial_path)
        feat_df = window_trial_features(trial_df, cfg)
        all_rows.append(feat_df)

    features = pd.concat(all_rows, ignore_index=True)
    features.to_csv(out_path, index=False)
    return features


def split_subjects(features: pd.DataFrame, seed: int) -> Dict[str, np.ndarray]:
    ids = np.array(sorted(features["subject_id"].unique().tolist()), dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = max(1, int(round(0.70 * n)))
    n_val = max(1, int(round(0.15 * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = max(1, n - n_train - n_val)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val : n_train + n_val + n_test]

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def save_split(split: Dict[str, np.ndarray], path: Path) -> None:
    payload = {
        "train": [int(x) for x in split["train"]],
        "val": [int(x) for x in split["val"]],
        "test": [int(x) for x in split["test"]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_split(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {
        "train": np.asarray(raw["train"], dtype=int),
        "val": np.asarray(raw["val"], dtype=int),
        "test": np.asarray(raw["test"], dtype=int),
    }


def train_predictor_model(
    cfg: PipelineConfig,
    features: pd.DataFrame,
    split: Dict[str, np.ndarray],
    model_root: Path,
    force_retrain: bool,
    seq_len: int,
    horizon: int,
    val_frac: float,
    fast_svr: bool,
    lstm_epochs: int,
    verbose: int,
) -> HybridPredictor:
    predictor_meta = model_root / "predictor_meta.joblib"
    if predictor_meta.exists() and not force_retrain:
        return HybridPredictor().load(root=model_root)

    is_train = features["subject_id"].isin(split["train"])
    train_df = features.loc[is_train].reset_index(drop=True)
    X = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y = train_df[TARGET_COLS].to_numpy(dtype=float)

    predictor = HybridPredictor(seq_len=seq_len, horizon=horizon)
    predictor.train(
        X,
        y,
        val_frac=val_frac,
        fast_svr=fast_svr,
        lstm_epochs=lstm_epochs,
        verbose=verbose,
    )
    predictor.save(root=model_root)
    return predictor


def train_decoder_model(
    cfg: PipelineConfig,
    features: pd.DataFrame,
    split: Dict[str, np.ndarray],
    qsvm_path: Path,
    force_retrain: bool,
    n_qubits: int,
    vote_window: int,
    max_samples: int,
) -> QSVMIntentionDecoder:
    if qsvm_path.exists() and not force_retrain:
        return QSVMIntentionDecoder().load(path=qsvm_path)

    is_train = features["subject_id"].isin(split["train"])
    train_df = features.loc[is_train].reset_index(drop=True)

    Xi = train_df[INTENT_FEATURE_COLS].to_numpy(dtype=float)
    yi = train_df["intent"].to_numpy(dtype=int)

    qsvm = QSVMIntentionDecoder(n_qubits=n_qubits, vote_window=vote_window)
    qsvm.train(Xi, yi, max_samples=max_samples, random_state=cfg.random_seed)
    qsvm.save(path=qsvm_path)
    return qsvm


def tune_fuzzy_estimator(
    features: pd.DataFrame,
    split: Dict[str, np.ndarray],
    w1_values: np.ndarray,
    w2_values: np.ndarray,
) -> Tuple[FuzzyLoadEstimator, Dict[str, float]]:
    train_df = features.loc[features["subject_id"].isin(split["train"])].reset_index(drop=True)
    val_df = features.loc[features["subject_id"].isin(split["val"])].reset_index(drop=True)

    if len(val_df) == 0:
        val_df = train_df.copy()

    best = {
        "w1": 0.7,
        "w2": 0.7,
        "mae_load": float("inf"),
    }

    for w1 in w1_values:
        for w2 in w2_values:
            est = FuzzyLoadEstimator(w1=float(w1), w2=float(w2))
            preds = []
            for _, row in val_df.iterrows():
                pred = est(
                    float(row["emg_biceps_rms"]),
                    float(row["emg_biceps_var"]),
                    emg_fds_rms=0.8 * float(row["emg_biceps_rms"]),
                    emg_fds_var=0.8 * float(row["emg_biceps_var"]),
                )
                preds.append(pred)

            mae = float(np.mean(np.abs(np.asarray(preds) - val_df["load_kg"].to_numpy(dtype=float))))
            if mae < best["mae_load"]:
                best = {
                    "w1": float(w1),
                    "w2": float(w2),
                    "mae_load": mae,
                }

    return FuzzyLoadEstimator(w1=best["w1"], w2=best["w2"]), best


def save_fuzzy_config(path: Path, cfg_dict: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cfg_dict, fh, indent=2)


def load_fuzzy_estimator(path: Path, default_w1: float, default_w2: float) -> FuzzyLoadEstimator:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return FuzzyLoadEstimator(w1=float(payload.get("w1", default_w1)), w2=float(payload.get("w2", default_w2)))
    return FuzzyLoadEstimator(w1=default_w1, w2=default_w2)


def train_models(
    cfg: PipelineConfig,
    features: pd.DataFrame,
    split: Dict[str, np.ndarray],
    force_retrain: bool,
    pred_seq_len: int,
    pred_horizon: int,
    pred_val_frac: float,
    pred_fast_svr: bool,
    pred_lstm_epochs: int,
    pred_verbose: int,
    dec_n_qubits: int,
    dec_vote_window: int,
    dec_max_samples: int,
    fuzzy_w1: float,
    fuzzy_w2: float,
    tune_fuzzy: bool,
    fuzzy_grid_values: np.ndarray,
) -> Tuple[HybridPredictor, QSVMIntentionDecoder, FuzzyLoadEstimator]:
    model_root = cfg.results_dir / "models"
    model_root.mkdir(parents=True, exist_ok=True)

    split_path = model_root / "subject_split.json"
    qsvm_path = model_root / "qsvm.joblib"
    fuzzy_path = model_root / "fuzzy_config.json"

    save_split(split, split_path)

    predictor = train_predictor_model(
        cfg,
        features,
        split,
        model_root=model_root,
        force_retrain=force_retrain,
        seq_len=pred_seq_len,
        horizon=pred_horizon,
        val_frac=pred_val_frac,
        fast_svr=pred_fast_svr,
        lstm_epochs=pred_lstm_epochs,
        verbose=pred_verbose,
    )

    qsvm = train_decoder_model(
        cfg,
        features,
        split,
        qsvm_path=qsvm_path,
        force_retrain=force_retrain,
        n_qubits=dec_n_qubits,
        vote_window=dec_vote_window,
        max_samples=dec_max_samples,
    )

    if tune_fuzzy:
        fuzzy, best = tune_fuzzy_estimator(features, split, fuzzy_grid_values, fuzzy_grid_values)
        save_fuzzy_config(fuzzy_path, best)
    else:
        fuzzy = load_fuzzy_estimator(fuzzy_path, fuzzy_w1, fuzzy_w2)
        save_fuzzy_config(
            fuzzy_path,
            {
                "w1": float(fuzzy.w1),
                "w2": float(fuzzy.w2),
                "mae_load": float("nan"),
            },
        )

    return predictor, qsvm, fuzzy


def select_eval_trial(manifest: pd.DataFrame, test_subjects: np.ndarray) -> Path:
    sub = manifest[manifest["subject_id"].isin(test_subjects)]
    if len(sub) == 0:
        sub = manifest

    for cond in ("fatigue_spasticity", "load", "spasticity", "fatigue", "nominal"):
        hit = sub[sub["condition"] == cond]
        if len(hit) > 0:
            return Path(hit.iloc[0]["file"])

    return Path(sub.iloc[0]["file"])


def compute_rms_var_window(x: np.ndarray) -> Tuple[float, float]:
    rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    var = float(np.var(x))
    return rms, var


def map_runtime_excitations(b: float, t: float, d: float) -> Tuple[float, float, float]:
    return (
        float(np.clip(b, 0.01, 0.95)),
        float(np.clip(t, 0.01, 0.95)),
        float(np.clip(d, 0.01, 0.95)),
    )


def reset_predictor_buffer(predictor: HybridPredictor) -> None:
    if predictor.buffer is not None:
        predictor.buffer = np.zeros_like(predictor.buffer)


def reset_decoder_buffer(decoder: QSVMIntentionDecoder) -> None:
    decoder._buffer = deque(maxlen=decoder.vote_window)


def run_controller_on_trial(
    cfg: PipelineConfig,
    trial_df: pd.DataFrame,
    controller_name: str,
    predictor: HybridPredictor,
    decoder: QSVMIntentionDecoder,
    fuzzy: FuzzyLoadEstimator,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    plant = Arm26Plant(cfg.model_path)
    q0 = np.deg2rad(trial_df[["theta_s_ref", "theta_e_ref"]].iloc[0].to_numpy())
    plant.reset(q0_rad=q0)

    n = len(trial_df)
    win = max(2, int(round(cfg.window_ms * cfg.fs_control / 1000.0)))
    decode_step = max(1, int(round(cfg.fs_control / 10)))

    if controller_name == "C1":
        controller = NoAssistController()
    elif controller_name == "C2":
        controller = FixedAdmittanceController()
    elif controller_name == "C3":
        controller = AdaptiveAdmittanceController()
    else:
        raise ValueError(f"Unknown controller: {controller_name}")

    reset_predictor_buffer(predictor)
    reset_decoder_buffer(decoder)

    current_intent = INTENT_HOLD
    current_load = 0.0

    logs: List[Dict[str, float]] = []

    emg_b_all = trial_df["emg_biceps"].to_numpy(dtype=float)
    emg_t_all = trial_df["emg_triceps"].to_numpy(dtype=float)
    emg_d_all = trial_df["emg_deltoid"].to_numpy(dtype=float)

    for i in range(n):
        q_deg, dq_deg = plant.read_state_deg()

        t_now = float(trial_df["time"].iloc[i])
        q_ref = trial_df[["theta_s_ref", "theta_e_ref"]].iloc[i].to_numpy(dtype=float)

        i0 = max(0, i - win + 1)
        seg_b = emg_b_all[i0 : i + 1]
        seg_t = emg_t_all[i0 : i + 1]
        seg_d = emg_d_all[i0 : i + 1]

        rms_b, var_b = compute_rms_var_window(seg_b)
        rms_t, var_t = compute_rms_var_window(seg_t)
        rms_d, var_d = compute_rms_var_window(seg_d)

        cci_s = compute_cci(rms_d, rms_t)
        cci_e = compute_cci(rms_b, rms_t)
        nmi = compute_nmi(cci_s, cci_e, q_deg[0], q_ref[0], q_deg[1], q_ref[1], cfg)

        if i % decode_step == 0 and controller_name == "C3":
            f_intent = np.array([rms_b, rms_t, rms_d, var_b, var_t, var_d], dtype=float)
            current_intent = decoder.predict_safe(f_intent)

            # Finger-flexor proxy from biceps features in this arm26-only setup.
            current_load = fuzzy(rms_b, var_b, emg_fds_rms=0.8 * rms_b, emg_fds_var=0.8 * var_b)

        if controller_name == "C3":
            x_pred = np.array([rms_b, rms_t, rms_d, var_b, var_t, var_d, q_deg[0], q_deg[1]], dtype=float)
            q_pred = predictor.predict(x_pred)
        else:
            q_pred = q_ref.copy()

        if controller_name == "C1":
            tau = np.zeros(2, dtype=float)
        else:
            tau = controller.compute_torque(
                np.deg2rad(q_deg),
                np.deg2rad(dq_deg),
                np.deg2rad(q_pred),
                nmi,
                load_kg=float(current_load),
            )
            tau = np.asarray(tau, dtype=float)

            if controller_name == "C3" and current_intent == INTENT_HOLD:
                tau *= 0.3

        b_in = emg_b_all[i] + float(rng.normal(0.0, cfg.runtime_noise_std))
        t_in = emg_t_all[i] + float(rng.normal(0.0, cfg.runtime_noise_std))
        d_in = emg_d_all[i] + float(rng.normal(0.0, cfg.runtime_noise_std))
        b_in, t_in, d_in = map_runtime_excitations(b_in, t_in, d_in)

        plant.set_channel_excitations(
            b_in,
            t_in,
            d_in,
            channel_gain={"biceps": 1.0, "triceps": 1.0, "deltoid": 1.0},
        )
        plant.set_assist_torque(float(tau[0]), float(tau[1]))

        logs.append(
            {
                "time": t_now,
                "theta_s": float(q_deg[0]),
                "theta_e": float(q_deg[1]),
                "theta_s_ref": float(q_ref[0]),
                "theta_e_ref": float(q_ref[1]),
                "theta_s_pred": float(q_pred[0]),
                "theta_e_pred": float(q_pred[1]),
                "tau_sh": float(tau[0]),
                "tau_el": float(tau[1]),
                "nmi": nmi,
                "cci_s": cci_s,
                "cci_e": cci_e,
                "emg_biceps_rms": rms_b,
                "emg_triceps_rms": rms_t,
                "emg_deltoid_rms": rms_d,
                "load_hat": float(current_load),
                "intent": int(current_intent),
                "controller": controller_name,
            }
        )

        if i < n - 1:
            plant.step(cfg.dt)

    return pd.DataFrame(logs)


def evaluate_log(log_df: pd.DataFrame, dt: float) -> Dict[str, float]:
    q = log_df[["theta_s", "theta_e"]].to_numpy(dtype=float)
    q_ref = log_df[["theta_s_ref", "theta_e_ref"]].to_numpy(dtype=float)
    err = q - q_ref

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    rms_emg = float(log_df[["emg_biceps_rms", "emg_triceps_rms", "emg_deltoid_rms"]].to_numpy(dtype=float).mean())
    mean_cci = float(0.5 * (log_df["cci_s"].mean() + log_df["cci_e"].mean()))

    q_rad = np.deg2rad(q)
    jerk = np.diff(q_rad, n=3, axis=0) / (dt**3)
    jerk_int = float(np.sum(jerk**2)) if len(jerk) > 0 else 0.0

    return {
        "MAE_deg": mae,
        "RMSE_deg": rmse,
        "mean_RMS_EMG": rms_emg,
        "mean_CCI": mean_cci,
        "jerk_integrated": jerk_int,
    }


def aggregate_metrics(rows: List[Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def pairwise_stats(metric_name: str, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray) -> pd.DataFrame:
    pairs = [("C1 vs C2", c1, c2), ("C1 vs C3", c1, c3), ("C2 vs C3", c2, c3)]
    out = []
    for label, a, b in pairs:
        diff = a - b
        if len(diff) >= 3:
            _, p_norm = stats.shapiro(diff)
        else:
            p_norm = 0.0

        if p_norm > 0.05:
            test = "paired_t"
            stat, p = stats.ttest_rel(a, b)
        else:
            test = "wilcoxon"
            try:
                stat, p = stats.wilcoxon(a, b)
            except ValueError:
                stat, p = np.nan, 1.0

        p_bonf = min(float(p) * 3.0, 1.0)
        out.append(
            {
                "metric": metric_name,
                "pair": label,
                "test": test,
                "statistic": float(stat),
                "p_raw": float(p),
                "p_bonferroni": p_bonf,
                "significant_alpha05": bool(p_bonf < 0.05),
                "highly_significant_p001": bool(p_bonf < 0.001),
            }
        )
    return pd.DataFrame(out)


def plot_representative_runs(logs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, idx, title in zip(axes, [0, 1], ["Shoulder", "Elbow"]):
        ref_name = ["theta_s_ref", "theta_e_ref"][idx]
        q_name = ["theta_s", "theta_e"][idx]

        t = logs["C1"]["time"].to_numpy()
        ax.plot(t, logs["C1"][ref_name], "k--", linewidth=2, label="Reference")
        ax.plot(t, logs["C1"][q_name], label="C1")
        ax.plot(t, logs["C2"][q_name], label="C2")
        ax.plot(t, logs["C3"][q_name], label="C3")
        ax.set_ylabel(f"{title} angle (deg)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "tracking_comparison.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, key in zip(axes, ["C1", "C2", "C3"]):
        t = logs[key]["time"].to_numpy()
        rms_mean = logs[key][["emg_biceps_rms", "emg_triceps_rms", "emg_deltoid_rms"]].mean(axis=1)
        ax.plot(t, rms_mean, color="steelblue", label="RMS EMG")
        ax2 = ax.twinx()
        ax2.plot(t, logs[key]["nmi"], color="firebrick", label="NMI")
        ax.set_title(key)
        ax.set_ylabel("RMS EMG")
        ax2.set_ylabel("NMI")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "emg_nmi_dynamics.png", dpi=140)
    plt.close(fig)


def run_benchmarks(
    cfg: PipelineConfig,
    trial_df: pd.DataFrame,
    predictor: HybridPredictor,
    decoder: QSVMIntentionDecoder,
    fuzzy: FuzzyLoadEstimator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows_c1: List[Dict[str, float]] = []
    rows_c2: List[Dict[str, float]] = []
    rows_c3: List[Dict[str, float]] = []

    representative_logs: Dict[str, pd.DataFrame] = {}

    for run in range(cfg.benchmark_runs):
        seed = cfg.random_seed + 5000 + run

        log1 = run_controller_on_trial(cfg, trial_df, "C1", predictor, decoder, fuzzy, seed=seed)
        m1 = evaluate_log(log1, cfg.dt)

        log2 = run_controller_on_trial(cfg, trial_df, "C2", predictor, decoder, fuzzy, seed=seed)
        m2 = evaluate_log(log2, cfg.dt)

        log3 = run_controller_on_trial(cfg, trial_df, "C3", predictor, decoder, fuzzy, seed=seed)
        m3 = evaluate_log(log3, cfg.dt)

        # Relative reductions vs C1 for interpretability.
        m1["EMG_reduction_%"] = 0.0
        m1["CCI_reduction_%"] = 0.0

        m2["EMG_reduction_%"] = 100.0 * (1.0 - m2["mean_RMS_EMG"] / max(m1["mean_RMS_EMG"], 1e-9))
        m2["CCI_reduction_%"] = 100.0 * (1.0 - m2["mean_CCI"] / max(m1["mean_CCI"], 1e-9))

        m3["EMG_reduction_%"] = 100.0 * (1.0 - m3["mean_RMS_EMG"] / max(m1["mean_RMS_EMG"], 1e-9))
        m3["CCI_reduction_%"] = 100.0 * (1.0 - m3["mean_CCI"] / max(m1["mean_CCI"], 1e-9))

        m1["run"] = run
        m2["run"] = run
        m3["run"] = run

        rows_c1.append(m1)
        rows_c2.append(m2)
        rows_c3.append(m3)

        if run == 0:
            representative_logs = {"C1": log1, "C2": log2, "C3": log3}

        print(
            f"run {run + 1:02d}/{cfg.benchmark_runs}: "
            f"MAE C1={m1['MAE_deg']:.2f}, C2={m2['MAE_deg']:.2f}, C3={m3['MAE_deg']:.2f}"
        )

    return aggregate_metrics(rows_c1), aggregate_metrics(rows_c2), aggregate_metrics(rows_c3), representative_logs


def summarize_controller(df: pd.DataFrame) -> pd.DataFrame:
    stats_df = df.agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    return stats_df


def load_eval_trial(
    cfg: PipelineConfig,
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    split: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, str]:
    if args.gui_csv is not None:
        # User supplied a CSV built from OpenSim GUI outputs via build_csv_from_gui.py.
        eval_path = Path(args.gui_csv).resolve()
        if not eval_path.exists():
            raise FileNotFoundError(f"--gui-csv file not found: {eval_path}")
        trial_df = pd.read_csv(eval_path)
        # Normalize column names so run_controller_on_trial() can consume the data.
        # Runtime expects emg_biceps/emg_triceps/emg_deltoid columns.
        if "emg_biceps" not in trial_df.columns and "biceps" in trial_df.columns:
            trial_df = trial_df.rename(
                columns={
                    "biceps": "emg_biceps",
                    "triceps": "emg_triceps",
                    "deltoid": "emg_deltoid",
                }
            )

        required_cols = [
            "time",
            "theta_s_ref",
            "theta_e_ref",
            "emg_biceps",
            "emg_triceps",
            "emg_deltoid",
        ]
        missing = [c for c in required_cols if c not in trial_df.columns]
        if missing:
            raise ValueError(
                "--gui-csv is missing required columns: "
                f"{missing}. Expected at least: {required_cols}"
            )
        eval_rel = eval_path.name
        print(f"Evaluation trial (from --gui-csv): {eval_path}")
        return trial_df, str(eval_rel)

    eval_rel = select_eval_trial(manifest, split["test"])
    eval_path = cfg.data_dir / eval_rel
    trial_df = pd.read_csv(eval_path)
    print(f"Evaluation trial: {eval_rel}")
    return trial_df, str(eval_rel)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arm26 full OpenSim-only paper pipeline")
    p.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=(
            "all",
            "data",
            "features",
            "split",
            "train_predictor",
            "train_decoder",
            "train_fuzzy",
            "train_all",
            "benchmark",
        ),
        help="Run one stage only, or run the full pipeline with 'all'.",
    )
    p.add_argument("--model", type=Path, default=MODEL_DEFAULT, help="Path to arm26_paper_loaded.osim")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT, help="Folder for generated dataset")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR_DEFAULT, help="Folder for outputs")
    p.add_argument("--quick", action="store_true", help="Smaller, faster run for validation")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cycles", type=int, default=None, help="Override number of task cycles")
    p.add_argument("--subjects", type=int, default=None, help="Override number of synthetic subjects")
    p.add_argument("--trials", type=int, default=None, help="Override trials per condition")
    p.add_argument("--runs", type=int, default=None, help="Override benchmark repeats")
    p.add_argument("--regen-data", action="store_true", help="Force regeneration of trial dataset")
    p.add_argument("--recompute-features", action="store_true", help="Force recomputation of windowed features")
    p.add_argument("--retrain-models", action="store_true", help="Force retraining predictor and decoder")
    p.add_argument("--resplit", action="store_true", help="Force recomputation of subject train/val/test split")

    p.add_argument("--pred-seq-len", type=int, default=10, help="Hybrid predictor sequence length")
    p.add_argument("--pred-horizon", type=int, default=3, help="Hybrid predictor horizon")
    p.add_argument("--pred-val-frac", type=float, default=0.2, help="Validation fraction for predictor training")
    p.add_argument("--pred-svr-mode", choices=("fast", "grid"), default="fast", help="SVR training mode")
    p.add_argument("--pred-lstm-epochs", type=int, default=15, help="LSTM epochs")
    p.add_argument("--pred-verbose", type=int, default=0, help="Keras training verbosity for predictor")
    p.add_argument("--retrain-predictor", action="store_true", help="Force retraining predictor only")

    p.add_argument("--dec-n-qubits", type=int, default=4, help="QSVM PCA output dimension / qubits")
    p.add_argument("--dec-vote-window", type=int, default=5, help="QSVM majority vote window")
    p.add_argument("--dec-max-samples", type=int, default=400, help="Maximum decoder train samples")
    p.add_argument("--retrain-decoder", action="store_true", help="Force retraining decoder only")

    p.add_argument("--fuzzy-w1", type=float, default=0.7, help="Fuzzy feature blend weight for f1")
    p.add_argument("--fuzzy-w2", type=float, default=0.7, help="Fuzzy feature blend weight for f2")
    p.add_argument("--fuzzy-tune", action="store_true", help="Grid-search w1,w2 on split validation set")
    p.add_argument("--fuzzy-grid-min", type=float, default=0.4, help="Lower bound for fuzzy tuning grid")
    p.add_argument("--fuzzy-grid-max", type=float, default=0.9, help="Upper bound for fuzzy tuning grid")
    p.add_argument("--fuzzy-grid-steps", type=int, default=6, help="Number of grid points per fuzzy axis")
    p.add_argument("--retune-fuzzy", action="store_true", help="Force fuzzy retuning stage")

    p.add_argument("--split-file", type=Path, default=None, help="Optional custom split JSON path")
    p.add_argument("--gui-csv", type=Path, default=None,
                   help="Use this CSV (from build_csv_from_gui.py) as the evaluation trial "
                        "instead of selecting from the auto-generated synthetic test set.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.quick:
        cfg = PipelineConfig(
            model_path=args.model,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            n_subjects=3,
            trials_per_condition=1,
            conditions=("nominal", "fatigue_spasticity", "load"),
            n_cycles=2,
            benchmark_runs=3,
            random_seed=args.seed,
        )
    else:
        cfg = PipelineConfig(
            model_path=args.model,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            random_seed=args.seed,
        )

    if args.cycles is not None:
        cfg.n_cycles = int(args.cycles)
    if args.subjects is not None:
        cfg.n_subjects = int(args.subjects)
    if args.trials is not None:
        cfg.trials_per_condition = int(args.trials)
    if args.runs is not None:
        cfg.benchmark_runs = int(args.runs)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    model_root = cfg.results_dir / "models"
    model_root.mkdir(parents=True, exist_ok=True)

    split_path = args.split_file if args.split_file is not None else (model_root / "subject_split.json")
    qsvm_path = model_root / "qsvm.joblib"
    fuzzy_path = model_root / "fuzzy_config.json"

    stage = args.stage
    force_pred = bool(args.retrain_models or args.retrain_predictor)
    force_dec = bool(args.retrain_models or args.retrain_decoder)
    force_fuzzy = bool(args.retrain_models or args.retune_fuzzy)
    tune_fuzzy = bool(args.fuzzy_tune or force_fuzzy)

    fuzzy_grid_values = np.linspace(
        float(args.fuzzy_grid_min),
        float(args.fuzzy_grid_max),
        int(max(2, args.fuzzy_grid_steps)),
    )

    print("=" * 70)
    print(f"ARM26 PIPELINE MODE: {stage}")
    print("=" * 70)

    print("=" * 70)
    print("STAGE 1: Generate OpenSim-only dataset from arm26_paper_loaded.osim")
    print("=" * 70)
    manifest = generate_dataset(cfg, force_regenerate=args.regen_data)
    print(f"Trials: {len(manifest)}")
    if stage == "data":
        print("Done: dataset stage complete.")
        return 0

    print("\n" + "=" * 70)
    print("STAGE 2: Build windowed features, CCI, NMI")
    print("=" * 70)
    features = build_feature_table(cfg, manifest, force_recompute=args.recompute_features)
    print(f"Feature rows: {len(features)}")
    if stage == "features":
        print("Done: features stage complete.")
        return 0

    print("\n" + "=" * 70)
    print("STAGE 3: Split by subject (cross-subject protocol)")
    print("=" * 70)
    if split_path.exists() and not args.resplit:
        split = load_split(split_path)
    else:
        split = split_subjects(features, seed=cfg.random_seed)
        save_split(split, split_path)

    print(f"Train subjects: {split['train'].tolist()}")
    print(f"Val subjects:   {split['val'].tolist()}")
    print(f"Test subjects:  {split['test'].tolist()}")
    print(f"Split file:     {split_path}")
    if stage == "split":
        print("Done: split stage complete.")
        return 0

    print("\n" + "=" * 70)
    print("STAGE 4A: Train SVR-LSTM predictor")
    print("=" * 70)
    predictor = train_predictor_model(
        cfg,
        features,
        split,
        model_root=model_root,
        force_retrain=force_pred,
        seq_len=int(args.pred_seq_len),
        horizon=int(args.pred_horizon),
        val_frac=float(args.pred_val_frac),
        fast_svr=(args.pred_svr_mode == "fast"),
        lstm_epochs=int(args.pred_lstm_epochs),
        verbose=int(args.pred_verbose),
    )
    with (model_root / "predictor_config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "seq_len": int(args.pred_seq_len),
                "horizon": int(args.pred_horizon),
                "val_frac": float(args.pred_val_frac),
                "svr_mode": args.pred_svr_mode,
                "lstm_epochs": int(args.pred_lstm_epochs),
            },
            fh,
            indent=2,
        )

    if stage == "train_predictor":
        print("Done: predictor stage complete.")
        return 0

    print("\n" + "=" * 70)
    print("STAGE 4B: Train QSVM intention decoder")
    print("=" * 70)
    decoder = train_decoder_model(
        cfg,
        features,
        split,
        qsvm_path=qsvm_path,
        force_retrain=force_dec,
        n_qubits=int(args.dec_n_qubits),
        vote_window=int(args.dec_vote_window),
        max_samples=int(args.dec_max_samples),
    )
    with (model_root / "decoder_config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "n_qubits": int(args.dec_n_qubits),
                "vote_window": int(args.dec_vote_window),
                "max_samples": int(args.dec_max_samples),
            },
            fh,
            indent=2,
        )

    if stage == "train_decoder":
        print("Done: decoder stage complete.")
        return 0

    print("\n" + "=" * 70)
    print("STAGE 4C: Configure/Tune fuzzy load estimator")
    print("=" * 70)
    if tune_fuzzy:
        fuzzy, best = tune_fuzzy_estimator(features, split, fuzzy_grid_values, fuzzy_grid_values)
        save_fuzzy_config(fuzzy_path, best)
        print(f"Best fuzzy params: w1={best['w1']:.3f}, w2={best['w2']:.3f}, val_MAE={best['mae_load']:.4f}")
    else:
        fuzzy = load_fuzzy_estimator(fuzzy_path, default_w1=float(args.fuzzy_w1), default_w2=float(args.fuzzy_w2))
        save_fuzzy_config(
            fuzzy_path,
            {
                "w1": float(fuzzy.w1),
                "w2": float(fuzzy.w2),
                "mae_load": float("nan"),
            },
        )
        print(f"Fuzzy params: w1={float(fuzzy.w1):.3f}, w2={float(fuzzy.w2):.3f}")

    if stage == "train_fuzzy":
        print("Done: fuzzy stage complete.")
        return 0

    if stage == "train_all":
        print("Done: all training stages complete. Run with --stage benchmark to evaluate controllers.")
        return 0

    if stage != "all" and stage != "benchmark":
        raise ValueError(f"Unsupported stage: {stage}")

    trial_df, eval_rel = load_eval_trial(cfg, args, manifest, split)

    print("\n" + "=" * 70)
    print("STAGE 5: Closed-loop benchmark (C1, C2, C3)")
    print("=" * 70)
    m1, m2, m3, logs = run_benchmarks(cfg, trial_df, predictor, decoder, fuzzy)

    out_metrics = cfg.results_dir / "metrics"
    out_metrics.mkdir(parents=True, exist_ok=True)

    m1.to_csv(out_metrics / "metrics_C1.csv", index=False)
    m2.to_csv(out_metrics / "metrics_C2.csv", index=False)
    m3.to_csv(out_metrics / "metrics_C3.csv", index=False)

    s1 = summarize_controller(m1)
    s2 = summarize_controller(m2)
    s3 = summarize_controller(m3)

    s1.insert(0, "controller", "C1")
    s2.insert(0, "controller", "C2")
    s3.insert(0, "controller", "C3")

    summary = pd.concat([s1, s2, s3], ignore_index=True)
    summary.to_csv(out_metrics / "summary_metrics.csv", index=False)

    stats_rows = []
    for metric in [
        "MAE_deg",
        "RMSE_deg",
        "mean_RMS_EMG",
        "mean_CCI",
        "jerk_integrated",
        "EMG_reduction_%",
        "CCI_reduction_%",
    ]:
        stats_rows.append(
            pairwise_stats(
                metric,
                m1[metric].to_numpy(dtype=float),
                m2[metric].to_numpy(dtype=float),
                m3[metric].to_numpy(dtype=float),
            )
        )

    stats_df = pd.concat(stats_rows, ignore_index=True)
    stats_df.to_csv(out_metrics / "pairwise_stats.csv", index=False)

    out_plots = cfg.results_dir / "plots"
    plot_representative_runs(logs, out_plots)

    for key, df in logs.items():
        df.to_csv(cfg.results_dir / f"log_{key}_run01.csv", index=False)

    # Export each controller's joint trajectory as an OpenSim .mot file so the user
    # can replay the controlled motion directly in the GUI (paper Fig 6 / 7 reproduction).
    for key, df in logs.items():
        try:
            _write_mot_from_log(df, cfg.results_dir / f"{key.lower()}_predicted.mot",
                                 label=f"{key}_predicted")
        except Exception as exc:
            print(f"[warn] could not export {key}_predicted.mot: {exc}")

    metadata = {
        "model": str(cfg.model_path),
        "eval_trial": str(eval_rel),
        "subjects": cfg.n_subjects,
        "trials_per_condition": cfg.trials_per_condition,
        "conditions": list(cfg.conditions),
        "cycles": cfg.n_cycles,
        "benchmark_runs": cfg.benchmark_runs,
        "fs_control": cfg.fs_control,
        "dt": cfg.dt,
    }
    with (cfg.results_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Data folder:    {cfg.data_dir}")
    print(f"Results folder: {cfg.results_dir}")
    print(f"Summary file:   {out_metrics / 'summary_metrics.csv'}")
    print(f"Stats file:     {out_metrics / 'pairwise_stats.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
