"""The analysis service: video in, thirteen biomechanical signals out.

    uvicorn analysis.app:app --host 127.0.0.1 --port 8000

It also serves the patient app, so the browser calls /analyze_single on its own
origin. Two routes are exposed:

    POST /analyze_single   one camera, no calibration, no 3D  (what the app uses)
    POST /analyze          four synchronised cameras + Calib.toml

Nothing is stored: every request runs the pipeline and returns the result.
"""
import os
import tempfile

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from analysis.calibration import parse_calibration_file
from analysis.config import (
    CAPTURE_FPS,
    DEFAULT_MULTI_CAMERA_STRIDE,
    DEFAULT_SINGLE_CAMERA_STRIDE,
    MULTI_CAMERA_VALIDATED_R2,
    MULTI_CAMERA_WEIGHTS,
    PATIENT_APP_DIR,
    SINGLE_CAMERA_VALIDATED_R2,
    SINGLE_CAMERA_WEIGHTS,
)
from analysis.multi_camera import (
    build_feature_table,
    lowpass_filter_trajectories,
    triangulate_right_arm,
)
from analysis.pose import estimate_keypoints_in_video
from analysis.signals import build_signal_series, flatten_keypoints, sample_indices
from analysis.single_camera import extract_right_arm, get_predictor

app = FastAPI(title="VisionBiomech Analysis")

# The app is normally same-origin, but the endpoints are also called directly
# during development from a page served elsewhere.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _save_upload(upload: UploadFile, directory: str, fallback_name: str) -> str:
    """Write an uploaded file to `directory` and return its path.

    The name is taken from the upload but stripped to its last component, so a
    crafted name cannot write outside the temporary directory.
    """
    name = os.path.basename(upload.filename or fallback_name)
    path = os.path.join(directory, name)
    with open(path, "wb") as destination:
        destination.write(upload.file.read())
    return path


# Both endpoints are declared `def` rather than `async def` on purpose: FastAPI
# then runs them in a worker thread, so a minute of CPU-bound pipeline does not
# block the event loop and the app stays responsive.


@app.post("/analyze_single")
def analyze_single_camera(
    cam1: UploadFile = File(...),
    stride: int = Form(DEFAULT_SINGLE_CAMERA_STRIDE),
):
    """One ordinary camera: RTMPose 2D keypoints, then the recurrent predictor."""
    upload_dir = tempfile.mkdtemp()
    video_path = _save_upload(cam1, upload_dir, "cam1.mp4")

    frames = estimate_keypoints_in_video(video_path, stride)
    fps = CAPTURE_FPS / stride

    arm_keypoints = extract_right_arm(frames)
    predictions, elbow_angle = get_predictor(SINGLE_CAMERA_WEIGHTS).predict(arm_keypoints, fps)

    indices = sample_indices(len(elbow_angle))
    return JSONResponse(
        {
            "time": [round(float(index / fps), 3) for index in indices],
            "vars": build_signal_series(predictions, indices),
            "keypoints": {"cam1": flatten_keypoints(frames)},
            "stride": stride,
            "res": 900,
            "fps": int(CAPTURE_FPS),
            "frames": int(len(elbow_angle)),
            "elbow_deg": [round(float(elbow_angle[i]), 1) for i in indices],
            "pose": "RTMPose-s (real-time)",
            "model_R2_validated": SINGLE_CAMERA_VALIDATED_R2,
            "note": (
                "Single-camera: RTMPose-s 2D -> BiGRU. One view, no calibration, no 3D. "
                "R² is the model's offline leave-one-subject-out accuracy, not a "
                "measurement on this recording."
            ),
        }
    )


@app.post("/analyze")
def analyze_multi_camera(
    cam1: UploadFile = File(...),
    cam2: UploadFile = File(...),
    cam3: UploadFile = File(...),
    cam4: UploadFile = File(...),
    calib: UploadFile = File(...),
    stride: int = Form(DEFAULT_MULTI_CAMERA_STRIDE),
):
    """Four synchronised cameras: pose, triangulation, filtering, angles, model.

    See `build_feature_table` for a known defect in the elbow-angle convention
    on this route; its outputs should not be trusted until it is corrected.
    """
    upload_dir = tempfile.mkdtemp()
    video_paths = [
        _save_upload(upload, upload_dir, f"cam{index}.mp4")
        for index, upload in enumerate((cam1, cam2, cam3, cam4), start=1)
    ]
    calibration_path = _save_upload(calib, upload_dir, "Calib.toml")
    fps = CAPTURE_FPS / stride

    per_camera_keypoints = [estimate_keypoints_in_video(path, stride) for path in video_paths]
    cameras = parse_calibration_file(calibration_path)

    trajectories = triangulate_right_arm(per_camera_keypoints, [c["P"] for c in cameras])
    trajectories = lowpass_filter_trajectories(trajectories, fps)
    features = build_feature_table(trajectories, fps)

    bundle = joblib.load(MULTI_CAMERA_WEIGHTS)
    predictions = bundle["y_scaler"].inverse_transform(
        bundle["model"].predict(bundle["x_scaler"].transform(features[bundle["features"]]))
    )

    indices = sample_indices(len(features))
    by_target = {
        name: predictions[:, column] for column, name in enumerate(bundle["targets"])
    }

    return JSONResponse(
        {
            "time": [round(float(features.time.values[i]), 3) for i in indices],
            "vars": build_signal_series(by_target, indices),
            "keypoints": {
                f"cam{index}": flatten_keypoints(frames)
                for index, frames in enumerate(per_camera_keypoints, start=1)
            },
            "stride": stride,
            "res": 900,
            "fps": int(CAPTURE_FPS),
            "frames": int(len(features)),
            "model_R2_validated": MULTI_CAMERA_VALIDATED_R2,
            "note": (
                "Multi-camera: RTMPose -> triangulation -> filter -> angles -> Level-1 model. "
                "R² is the model's offline leave-one-subject-out accuracy, not a "
                "measurement on this recording."
            ),
        }
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "single_camera_model": os.path.basename(SINGLE_CAMERA_WEIGHTS),
        "multi_camera_model": os.path.basename(MULTI_CAMERA_WEIGHTS),
        "single_camera_R2": SINGLE_CAMERA_VALIDATED_R2,
        "multi_camera_R2": MULTI_CAMERA_VALIDATED_R2,
    }


class PatientAppFiles(StaticFiles):
    """The patient app, served with caching disabled for the app itself.

    A stale index.html or script in the browser cache presents as a broken
    button rather than as an out-of-date file. The ONNX models under /models are
    large and immutable, so they keep normal caching.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if not path.startswith("models/"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


app.mount("/", PatientAppFiles(directory=PATIENT_APP_DIR, html=True), name="patient-app")
