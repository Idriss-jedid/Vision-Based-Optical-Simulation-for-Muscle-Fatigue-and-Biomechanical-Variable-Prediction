"""
Render a static Arm26 loaded pose and save a PNG screenshot.

Usage:
    conda run -n biomech python Arm26/render_arm26_pose_png.py
"""

from pathlib import Path
import math
import sys
import time

try:
    import opensim as osim
    from PIL import ImageGrab
    import win32gui
except Exception as exc:
    print(f"ERROR: missing dependency: {exc}")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
MODEL_PATH = ROOT / "arm26_paper_loaded.osim"
OUT_PNG = ROOT / "arm26_pose_preview.png"

# Figure-like reference pose used in the paper context.
SHOULDER_DEG = 60.0
ELBOW_DEG = 80.0


def set_coord_if_exists(model: osim.Model, state: osim.State, name: str, value_rad: float) -> None:
    coords = model.getCoordinateSet()
    if coords.contains(name):
        coords.get(name).setValue(state, value_rad)


def find_simtk_window(max_wait_s: float = 5.0):
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        matches = []

        def callback(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return
            t = title.lower()
            if "simtk" in t or "simbody" in t or "opensim" in t:
                rect = win32gui.GetWindowRect(hwnd)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w > 200 and h > 200:
                    matches.append((hwnd, rect, title, w * h))

        win32gui.EnumWindows(callback, None)
        if matches:
            matches.sort(key=lambda x: x[3], reverse=True)
            return matches[0][1], matches[0][2]
        time.sleep(0.1)

    return None, None


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found: {MODEL_PATH}")
        return 1

    # Ensure geometry meshes are discoverable by OpenSim visualizer.
    osim.ModelVisualizer.addDirToGeometrySearchPaths(str(WORKSPACE / "Geometry"))
    osim.ModelVisualizer.addDirToGeometrySearchPaths(str(ROOT / "Geometry"))

    model = osim.Model(str(MODEL_PATH))
    model.setUseVisualizer(True)
    state = model.initSystem()

    set_coord_if_exists(model, state, "r_shoulder_elev", math.radians(SHOULDER_DEG))
    set_coord_if_exists(model, state, "r_elbow_flex", math.radians(ELBOW_DEG))

    model.realizePosition(state)

    viz = model.updVisualizer()
    sviz = viz.getSimbodyVisualizer()

    # Keep a stable, clean frame for screenshot.
    sviz.setShowFrameRate(False)
    sviz.setShowFrameNumber(False)
    sviz.setShowShadows(True)

    hand = model.getBodySet().get("r_ulna_radius_hand")
    hand_pos = hand.getPositionInGround(state)
    sviz.pointCameraAt(hand_pos, osim.Vec3(0, 1, 0))
    sviz.zoomCameraToShowAllGeometry()

    viz.show(state)
    time.sleep(0.7)

    # Capture only the OpenSim/SimTK viewer window when possible.
    rect, title = find_simtk_window()
    if rect is not None:
        img = ImageGrab.grab(bbox=rect, all_screens=True)
        print(f"Captured window: {title}")
    else:
        img = ImageGrab.grab(all_screens=True)
        print("WARNING: SimTK window not found; captured full screen.")

    img.save(OUT_PNG)

    print(f"Saved PNG: {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
