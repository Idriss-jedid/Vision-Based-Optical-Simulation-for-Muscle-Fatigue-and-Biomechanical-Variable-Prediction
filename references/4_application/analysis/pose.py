"""Two-dimensional pose estimation from video.

The detector is RTMPose in its lightweight configuration, running on CPU through
onnxruntime. It is the stage every other stage depends on, and the only one that
touches the video file.
"""
import cv2
import numpy as np
from rtmlib import Body

_body_estimator = None


def get_pose_estimator():
    """The shared RTMPose estimator.

    Loaded once and reused across videos and requests: construction reads the
    model from disk and costs far more than an inference does.
    """
    global _body_estimator
    if _body_estimator is None:
        _body_estimator = Body(mode="lightweight", backend="onnxruntime", device="cpu")
    return _body_estimator


def estimate_keypoints_in_video(video_path, stride):
    """Run pose estimation over a video, keeping one frame in every `stride`.

    Returns one entry per kept frame: a (17, 3) array of [x, y, confidence] in
    COCO-17 order, or None where no person was detected. The None is preserved
    rather than dropped so callers keep a regular time base.
    """
    estimator = get_pose_estimator()
    capture = cv2.VideoCapture(video_path)
    frames = []
    frame_index = 0

    while True:
        was_read, frame = capture.read()
        if not was_read:
            break
        if frame_index % stride == 0:
            keypoints, scores = estimator(frame)
            if not len(keypoints):
                frames.append(None)
            else:
                # Only the first person: these are single-subject recordings.
                frames.append(np.hstack([keypoints[0], np.array(scores[0])[:, None]]))
        frame_index += 1

    capture.release()
    return frames
