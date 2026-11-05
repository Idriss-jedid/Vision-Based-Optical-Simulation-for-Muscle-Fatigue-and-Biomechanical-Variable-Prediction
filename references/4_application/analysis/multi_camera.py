"""Multi-camera route: several views to 3D joints, then to angles and features.

The stages, in order: triangulate the right arm from the synchronised views,
low-pass filter the trajectories, then derive the joint angles and the
anthropometric features the Level-1 model consumes.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from analysis.config import MIN_KEYPOINT_CONFIDENCE, RIGHT_ARM_JOINTS

# Joint centres move slowly; anything above this in the trajectory is detector
# jitter rather than movement.
FILTER_CUTOFF_HZ = 2.0
FILTER_ORDER = 4

# Segment mass and length of the reference cadaveric model, used to scale the
# subject's own segments proportionally.
REFERENCE_HUMERUS_MASS_KG, REFERENCE_HUMERUS_LENGTH_M = 1.864, 0.291
REFERENCE_FOREARM_MASS_KG, REFERENCE_FOREARM_LENGTH_M = 1.534, 0.254

# Mass held in the hand during the recorded exercise, in kilograms.
HELD_LOAD_KG = 2


def triangulate_right_arm(per_camera_keypoints, projection_matrices):
    """Reconstruct shoulder, elbow and wrist in 3D from the synchronised views.

    Each joint is solved independently per frame by direct linear
    transformation: every camera that saw it confidently contributes two rows to
    a homogeneous system, and the solution is its smallest singular vector.

    A joint needs at least two contributing cameras (four rows) to be solvable;
    below that it is left as NaN and the filter downstream skips it.
    """
    frame_count = min(len(frames) for frames in per_camera_keypoints)
    reconstructed = np.full((frame_count, len(RIGHT_ARM_JOINTS), 3), np.nan)

    for frame in range(frame_count):
        for slot, joint in enumerate(RIGHT_ARM_JOINTS):
            rows = []
            for camera_index, frames in enumerate(per_camera_keypoints):
                keypoints = frames[frame]
                if keypoints is None or keypoints[joint, 2] < MIN_KEYPOINT_CONFIDENCE:
                    continue
                x, y = keypoints[joint, 0], keypoints[joint, 1]
                projection = projection_matrices[camera_index]
                rows.append(x * projection[2] - projection[0])
                rows.append(y * projection[2] - projection[1])

            if len(rows) >= 4:
                solution = np.linalg.svd(np.array(rows))[2][-1]
                reconstructed[frame, slot] = solution[:3] / solution[3]

    return reconstructed


def lowpass_filter_trajectories(trajectories, fps, cutoff_hz=FILTER_CUTOFF_HZ):
    """Zero-phase Butterworth filter over each joint and each axis.

    filtfilt runs the filter forwards and backwards, so the smoothing introduces
    no lag — which matters because the angles are differentiated afterwards and
    a phase shift would misplace the velocity peaks.

    Frames the triangulation could not solve are left out of the filter and
    stay NaN. Short valid stretches are skipped: filtfilt needs more samples
    than the filter order to run at all.
    """
    numerator, denominator = butter(FILTER_ORDER, cutoff_hz / (fps / 2))
    filtered = trajectories.copy()

    for joint in range(trajectories.shape[1]):
        for axis in range(trajectories.shape[2]):
            series = trajectories[:, joint, axis]
            valid = ~np.isnan(series)
            if valid.sum() > 13:
                filtered[valid, joint, axis] = filtfilt(numerator, denominator, series[valid])

    return filtered


def build_feature_table(trajectories, fps):
    """Derive joint angles, anthropometry and the model's input features.

    Returns one row per frame, holding the twenty-two columns the Level-1 model
    was trained on plus the time base.

    Known defect, inherited from the validated pipeline: the elbow angle is
    computed as 180 - the angle between the segments, so it reads 180° with the
    arm extended, whereas the model was trained on the OpenSim convention where
    an extended arm is 0°. The multi-camera route therefore feeds an inverted
    angle and its outputs should not be trusted until this is corrected. It is
    kept as-is so the endpoint behaves exactly as it was measured. The
    single-camera route is unaffected: it consumes raw 2D keypoints.
    """
    shoulder, elbow, wrist = trajectories[:, 0], trajectories[:, 1], trajectories[:, 2]
    upper_arm, forearm = elbow - shoulder, wrist - elbow

    cos_elbow = (upper_arm * forearm).sum(1) / (
        np.linalg.norm(upper_arm, axis=1) * np.linalg.norm(forearm, axis=1) + 1e-9
    )
    elbow_angle = 180 - np.degrees(np.arccos(np.clip(cos_elbow, -1, 1)))   # see docstring
    shoulder_angle = np.degrees(
        np.arctan2(np.abs(upper_arm[:, 0]) + np.abs(upper_arm[:, 2]), -upper_arm[:, 1] + 1e-9)
    )

    table = pd.DataFrame(
        {
            "subj": "u",
            "time": np.arange(len(elbow_angle)) / fps,
            "q_sh": shoulder_angle,
            "q_el": elbow_angle,
        }
    )
    # Frames with no reconstruction leave gaps; the derivatives below need a
    # continuous series, so they are bridged before anything is differentiated.
    for column in ("q_sh", "q_el"):
        table[column] = table[column].interpolate().bfill().ffill()

    timestep = 1 / fps
    table["qd_sh"] = np.gradient(table.q_sh, timestep)
    table["qd_el"] = np.gradient(table.q_el, timestep)
    table["qdd_sh"] = np.gradient(table.qd_sh, timestep)
    table["qdd_el"] = np.gradient(table.qd_el, timestep)

    # Segment lengths are measured from the reconstruction itself, by median so
    # a few bad frames cannot stretch them, and the masses follow by scaling the
    # reference model proportionally.
    humerus_length = float(np.nanmedian(np.linalg.norm(upper_arm, axis=1)))
    forearm_length = float(np.nanmedian(np.linalg.norm(forearm, axis=1)))
    table["humerus_len"] = humerus_length
    table["forearm_len"] = forearm_length
    table["humerus_mass"] = REFERENCE_HUMERUS_MASS_KG * humerus_length / REFERENCE_HUMERUS_LENGTH_M
    table["forearm_mass"] = REFERENCE_FOREARM_MASS_KG * forearm_length / REFERENCE_FOREARM_LENGTH_M

    # Trigonometric forms of the angles: the moment arm varies with the sine and
    # cosine of the joint angle, and a tree model cannot derive them on its own.
    to_radians = np.pi / 180
    table["sin_qel"] = np.sin(table.q_el * to_radians)
    table["cos_qel"] = np.cos(table.q_el * to_radians)
    table["sin_qsh"] = np.sin(table.q_sh * to_radians)
    table["cos_qsh"] = np.cos(table.q_sh * to_radians)

    table["abs_qd_el"] = table.qd_el.abs()
    table["abs_qdd_el"] = table.qdd_el.abs()
    table["qd_el2"] = table.qd_el ** 2

    table["grav_load"] = (
        (table.forearm_mass + HELD_LOAD_KG)
        * table.forearm_len
        * np.sin((table.q_sh + table.q_el) * to_radians)
    )
    table["qel_x_fmass"] = table.q_el * table.forearm_mass

    # Running totals: fatigue depends on what has already been done, so the
    # model is given the accumulated movement and the accumulated gravity load.
    table["cum_path_el"] = np.cumsum(table.abs_qd_el.values) * timestep
    table["cum_grav_imp"] = np.cumsum(np.abs(table.grav_load.values)) * timestep

    return table
