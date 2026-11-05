"""Describing and packaging the thirteen output signals.

Both endpoints return the same envelope, so the interfaces render a
single-camera and a multi-camera result with the same code.
"""
import numpy as np

from analysis.config import MAX_SERIES_POINTS

# The prefix a model uses for each family of target, and how to present it.
_FAMILY_BY_PREFIX = {"act": "activation", "frc": "force", "MF": "fatigue"}
_UNIT_BY_FAMILY = {"activation": "0–1", "force": "N", "fatigue": "%"}


def describe_signal(target_name):
    """Turn a model's target name into (label, unit, family).

    Targets are named `<prefix>_<muscle>`, except the joint moment which stands
    alone. `act_BIClong` becomes ("BIClong activation", "0–1", "activation").
    """
    if target_name == "elbow_moment":
        return "Joint torque", "N·m", "torque"

    prefix, muscle = target_name.split("_", 1)
    family = _FAMILY_BY_PREFIX[prefix]
    return f"{muscle.replace('_hand', '')} {family}", _UNIT_BY_FAMILY[family], family


def sample_indices(frame_count):
    """Evenly spaced frame indices, at most MAX_SERIES_POINTS of them."""
    return np.linspace(0, frame_count - 1, min(MAX_SERIES_POINTS, frame_count)).astype(int)


def build_signal_series(predictions_by_target, indices):
    """Package each predicted curve with its label, unit and family.

    `predictions_by_target` maps a target name to its full-length series; the
    result carries only the sampled points.
    """
    series = {}
    for target_name, values in predictions_by_target.items():
        label, unit, family = describe_signal(target_name)
        series[target_name] = {
            "label": label,
            "unit": unit,
            "group": family,
            "pred": [round(float(values[i]), 4) for i in indices],
        }
    return series


def flatten_keypoints(frames_keypoints):
    """Flatten per-frame COCO-17 keypoints for the overlay the client draws.

    Each frame becomes a flat [x, y, confidence, x, y, confidence, ...] list,
    with coordinates rounded to the pixel and confidence to a percentage —
    the overlay cannot use more precision, and the response is a third smaller.
    A frame where nobody was detected becomes an empty list.
    """
    flattened = []
    for keypoints in frames_keypoints:
        if keypoints is None:
            flattened.append([])
            continue
        flattened.append(
            [
                value
                for joint in keypoints
                for value in (round(float(joint[0])), round(float(joint[1])), round(float(joint[2]) * 100))
            ]
        )
    return flattened
