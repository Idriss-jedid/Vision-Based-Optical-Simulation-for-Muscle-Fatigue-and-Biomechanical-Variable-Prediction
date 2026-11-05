"""Paths and acquisition constants for the analysis service."""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(PACKAGE_DIR, "weights")

# The single-camera predictor: 2D arm keypoints -> thirteen biomechanical signals.
SINGLE_CAMERA_WEIGHTS = os.path.join(WEIGHTS_DIR, "single_camera_bigru.pt")

# The multi-camera predictor: joint angles + anthropometry -> the same thirteen.
MULTI_CAMERA_WEIGHTS = os.path.join(WEIGHTS_DIR, "multicamera_lgbm.joblib")

# The patient progressive web app, served from this same origin so its calls to
# /analyze_single are same-origin and need no CORS preflight.
PATIENT_APP_DIR = os.path.join(os.path.dirname(PACKAGE_DIR), "patient_app")

# The corpus this pipeline was validated on was recorded at 50 fps. Frames are
# processed one in every `stride`, so the effective rate is CAPTURE_FPS / stride.
CAPTURE_FPS = 50.0

# Multi-camera runs are slower per frame, so they skip more of them by default.
DEFAULT_MULTI_CAMERA_STRIDE = 6
DEFAULT_SINGLE_CAMERA_STRIDE = 3

# The curves returned to the client are decimated to this many points: enough to
# draw a faithful chart on a phone, small enough to keep the response light.
MAX_SERIES_POINTS = 300

# COCO-17 indices of the right arm, the chain the models were trained on.
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 6, 8, 10
RIGHT_ARM_JOINTS = (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)

# A triangulated joint is only used when at least this many cameras saw it
# confidently; below that the reconstruction is noise.
MIN_KEYPOINT_CONFIDENCE = 0.3

# Offline leave-one-subject-out accuracy of each model. Reported to the client
# for display, never recomputed at inference: there is no ground truth here.
SINGLE_CAMERA_VALIDATED_R2 = 0.955
MULTI_CAMERA_VALIDATED_R2 = 0.95
