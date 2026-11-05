"""Reading a camera-array calibration file.

Triangulation needs each camera's projection matrix, which is the intrinsic
matrix combined with the camera's pose in the shared world frame. Both come from
the Calib.toml produced when the array was calibrated.
"""
import numpy as np

try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib


def parse_calibration_file(path):
    """Read a Calib.toml into one projection matrix per camera.

    Each returned entry holds the intrinsic matrix K, the rotation R, the
    translation t, and the projection matrix P = K [R | t] that triangulation
    actually uses.

    Rotations are stored either as a Rodrigues vector or as a full matrix
    depending on the tool that wrote the file, so both are accepted. Any table
    without a `matrix` key is metadata, not a camera, and is skipped.
    """
    import cv2

    document = tomllib.loads(open(path, "rb").read().decode())
    cameras = []

    for _, entry in document.items():
        if not isinstance(entry, dict) or "matrix" not in entry:
            continue

        intrinsics = np.array(entry["matrix"], dtype=float)
        rotation_raw = np.array(entry["rotation"], dtype=float)
        rotation = (
            cv2.Rodrigues(rotation_raw)[0] if rotation_raw.size == 3 else rotation_raw.reshape(3, 3)
        )
        translation = np.array(entry["translation"], dtype=float)

        cameras.append(
            {
                "K": intrinsics,
                "R": rotation,
                "t": translation,
                "P": intrinsics @ np.hstack([rotation, translation[:, None]]),
            }
        )
    return cameras
