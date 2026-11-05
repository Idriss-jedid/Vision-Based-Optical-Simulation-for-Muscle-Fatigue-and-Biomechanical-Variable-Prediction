"""Single-camera prediction: 2D arm keypoints straight to thirteen signals.

No calibration, no triangulation, no three-dimensional reconstruction. A
recurrent network reads the pixel trajectory of shoulder, elbow and wrist and
produces the same thirteen signals the multi-camera route computes, which is
what lets an ordinary phone stand in for a camera array.
"""
import numpy as np
import torch
import torch.nn as nn

from analysis.config import RIGHT_ARM_JOINTS


class BiGRU(nn.Module):
    """Bidirectional GRU with a small feed-forward head.

    Bidirectional because fatigue at a given instant depends on what the muscle
    has already done and on how the movement finishes; the whole clip is
    available at inference, so there is no reason to read it forwards only.
    """

    def __init__(self, cin, cout=13, d=128, layers=2, p=0.1, cell="gru", **_):
        """Build the network described by a checkpoint's `model_kw`.

        These parameter names are terse because they are not ours to choose: the
        training code stored them verbatim in the checkpoint and they are passed
        back in as keywords, so renaming them here would silently fall back to
        the defaults and build the wrong-sized network.

            cin    number of input features per frame
            cout   number of predicted signals
            d      hidden size of one GRU direction
            layers stacked GRU layers
            p      dropout probability
            cell   recurrent unit named by the checkpoint; only GRU was trained

        `**_` absorbs any other key a future checkpoint may carry.
        """
        super().__init__()
        input_size, output_size, hidden_size, dropout = cin, cout, d, p

        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, sequence):
        return self.head(self.rnn(sequence)[0])


def arm_angles_2d(shoulder, elbow, wrist):
    """Elbow flexion and upper-arm tilt, in degrees, from image coordinates.

    These are a two-dimensional proxy for the joint angles the musculoskeletal
    model uses. Image axes run +x right and +y *down*, so "up" is -y.

    Elbow: 0° fully extended, larger means more flexed — the OpenSim convention
    the models were trained on.
    """
    upper_arm = elbow - shoulder
    forearm = wrist - elbow

    upper_arm_length = np.linalg.norm(upper_arm, axis=1) + 1e-9
    forearm_length = np.linalg.norm(forearm, axis=1) + 1e-9

    cos_elbow = np.clip(
        np.sum(upper_arm * forearm, axis=1) / (upper_arm_length * forearm_length), -1, 1
    )
    elbow_angle = np.degrees(np.arccos(cos_elbow))
    shoulder_angle = np.degrees(
        np.arccos(np.clip(-upper_arm[:, 1] / upper_arm_length, -1, 1))
    )
    return shoulder_angle.astype(np.float32), elbow_angle.astype(np.float32)


def build_model_input(arm_keypoints, fps):
    """Assemble the network's input from the raw arm trajectory.

    `arm_keypoints` is (frames, 3, 2): shoulder, elbow and wrist in pixels.

    The twenty-two features, in the order the model was trained on:
      - the six coordinates, and their first and second time derivatives, so the
        network sees velocity and acceleration without having to infer them;
      - a gravity-load proxy, its running integral, elapsed time, and normalised
        time. Fatigue depends on accumulated effort, not on the current pose, so
        the history has to be supplied explicitly.
    """
    shoulder, elbow, wrist = arm_keypoints[:, 0], arm_keypoints[:, 1], arm_keypoints[:, 2]
    coordinates = arm_keypoints.reshape(len(arm_keypoints), 6).astype(np.float32)

    frame_count = len(coordinates)
    timestep = 1.0 / float(fps)
    velocity = np.gradient(coordinates, axis=0)
    acceleration = np.gradient(velocity, axis=0)

    shoulder_angle, elbow_angle = arm_angles_2d(shoulder, elbow, wrist)
    gravity_load = np.abs(np.sin(np.radians(shoulder_angle + elbow_angle))).astype(np.float32)
    cumulative_load = (np.cumsum(gravity_load) * timestep).astype(np.float32)
    elapsed = (np.arange(frame_count) * timestep).astype(np.float32)
    normalised_time = (np.arange(frame_count) / max(frame_count - 1, 1)).astype(np.float32)

    features = np.concatenate(
        [
            coordinates,
            velocity,
            acceleration,
            gravity_load[:, None],
            cumulative_load[:, None],
            elapsed[:, None],
            normalised_time[:, None],
        ],
        axis=1,
    )
    return features.astype(np.float32), shoulder_angle, elbow_angle


def extract_right_arm(frames_keypoints):
    """Pull the right-arm trajectory out of per-frame COCO-17 keypoints.

    Frames where nobody was detected leave a gap, and the gap is filled by
    interpolating between the surrounding frames: the network reads a sequence
    and a hole in it would propagate through the derivatives.
    """
    arm = np.array(
        [
            [np.nan] * 6
            if keypoints is None
            else [keypoints[joint, axis] for joint in RIGHT_ARM_JOINTS for axis in (0, 1)]
            for keypoints in frames_keypoints
        ],
        dtype=float,
    )

    for column in range(arm.shape[1]):
        values = arm[:, column]
        known = ~np.isnan(values)
        if known.sum() > 1:
            arm[:, column] = np.interp(np.arange(len(values)), np.where(known)[0], values[known])

    # Any column still empty (the joint was never seen) collapses to zero.
    return np.nan_to_num(arm).reshape(-1, 3, 2)


class SingleCameraPredictor:
    """The trained network together with everything needed to use it.

    Weights, the input and output normalisation statistics and the target names
    all come from one checkpoint and are meaningless apart, so they are loaded
    and held together. Callers see one method and never touch a tensor, which is
    what allows the predictor to be replaced without changing either interface.
    """

    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.model = BiGRU(checkpoint["cin"], checkpoint["cout"], **checkpoint["model_kw"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.input_mean = checkpoint["xm"]
        self.input_std = checkpoint["xs"]
        self.output_mean = checkpoint["ym"]
        self.output_std = checkpoint["ys"]
        self.target_names = list(checkpoint["targets"])

    def predict(self, arm_keypoints, fps):
        """Predict the thirteen signals for one recording.

        Returns the signals keyed by target name, and the elbow angle, which the
        interfaces show alongside them as the movement actually performed.
        """
        features, _, elbow_angle = build_model_input(arm_keypoints, fps)
        normalised = ((features - self.input_mean) / self.input_std).astype(np.float32)

        with torch.no_grad():
            raw = self.model(torch.tensor(normalised[None])).numpy()[0]

        predictions = raw * self.output_std + self.output_mean

        # The network is not constrained to produce physically possible values,
        # so the ones with a known range are clipped into it: an activation is a
        # fraction, and neither a force nor a fatigue level can be negative.
        predictions[:, 1:5] = np.clip(predictions[:, 1:5], 0, 1)
        predictions[:, 5:] = np.clip(predictions[:, 5:], 0, None)

        by_target = {
            name: predictions[:, index] for index, name in enumerate(self.target_names)
        }
        return by_target, elbow_angle


_predictor = None


def get_predictor(checkpoint_path):
    """The shared predictor, loaded on first use rather than at import.

    Importing this module must stay cheap: the service also serves static files
    and answers /health, and neither should wait for a checkpoint to load.
    """
    global _predictor
    if _predictor is None:
        _predictor = SingleCameraPredictor(checkpoint_path)
    return _predictor
