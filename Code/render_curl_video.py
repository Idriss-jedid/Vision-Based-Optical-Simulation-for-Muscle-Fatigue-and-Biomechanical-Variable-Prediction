# -*- coding: utf-8 -*-
"""
render_curl_video.py
====================
Generate a standardized, high-quality SIDE-VIEW (sagittal) dumbbell-curl video
(.mp4) driven by a real OpenSim .mot kinematics file.

This is a *synthetic biomechanical render*, not a photoreal human:
- single virtual camera, pure right-side sagittal view, fixed (tripod-like)
- 1080p, 60 fps, landscape, clean studio background, floor + soft shadow
- head, torso, shoulder, upper-arm, forearm, hand, dumbbell all visible
- 2 s "standing still" before and after the motion (for trimming/sync)
- the elbow trajectory is EXACTLY the .mot r_elbow_flex (fatigue slowing,
  reduced ROM and tremor are already baked into the kinematics)

Anti-aliasing is done by drawing at SS x resolution and down-sampling (LANCZOS).
Frames are piped raw to ffmpeg (libx264, yuv420p) -- no extra Python deps.

Usage:
    python Code/render_curl_video.py                      # default 10-rep, 2 kg
    python Code/render_curl_video.py --mot <file.mot> --load 4 --reps 30 \
        --out Data/synthetic_videos/S01_right_4kg_side_30rep_take01.mp4
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------- .mot parsing
def read_mot(path):
    with open(path) as f:
        lines = f.read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().lower() == "endheader")
    cols = lines[i + 1].split()
    data = np.array([[float(x) for x in l.split()] for l in lines[i + 2:] if l.strip()])
    return cols, data


# --------------------------------------------------------- 2D sagittal skeleton
# All lengths in metres; person faces +x (camera looks at the right side).
SEG = dict(
    torso=0.52, neck=0.07, head_r=0.105,
    upper_arm=0.30, forearm=0.235, hand=0.085,
    pelvis_w=0.17, shoulder_w=0.20,
    thigh=0.45, shank=0.45,
    db_r=0.075,           # dumbbell plate radius
)
SHOULDER_W = SEG["shoulder_w"]


def forward_kinematics(shoulder_elev_deg, elbow_flex_deg):
    """Return dict of 2D joint positions (metres, y up) for the right arm."""
    S = np.array([0.0, 0.0])                       # shoulder = origin
    elev = math.radians(shoulder_elev_deg)
    # upper arm hangs down, tilted forward (+x) by the shoulder elevation
    d_ua = np.array([math.sin(elev), -math.cos(elev)])
    E = S + SEG["upper_arm"] * d_ua
    # elbow flexion rotates the forearm CCW (lifts hand forward & up)
    th = math.radians(elbow_flex_deg)
    c, s = math.cos(th), math.sin(th)
    d_fa = np.array([c * d_ua[0] - s * d_ua[1], s * d_ua[0] + c * d_ua[1]])
    W = E + SEG["forearm"] * d_fa
    Hd = W + SEG["hand"] * d_fa                     # hand / dumbbell centre
    hip = S + np.array([0.0, -SEG["torso"]])
    neck = S + np.array([0.0, SEG["neck"]])
    head = neck + np.array([0.0, SEG["head_r"] * 1.1])
    knee = hip + np.array([0.04, -SEG["thigh"]])
    ankle = knee + np.array([0.02, -SEG["shank"]])
    return dict(S=S, E=E, W=W, Hd=Hd, hip=hip, neck=neck, head=head,
                knee=knee, ankle=ankle)


# ------------------------------------------------------------------- rendering
W_OUT, H_OUT = 1920, 1080
SS = 2                                             # supersample factor
W, H = W_OUT * SS, H_OUT * SS

# studio palette
C_BG_TOP = (236, 238, 241)
C_BG_BOT = (208, 212, 218)
C_FLOOR = (188, 192, 198)
C_SHADOW = (170, 174, 180)
C_SKIN = (224, 178, 148)
C_SKIN_DK = (196, 150, 120)
C_SHIRT = (54, 96, 150)
C_SHIRT_DK = (38, 70, 112)
C_FARARM = (170, 188, 210)        # far (left) arm, behind torso
C_DB = (40, 42, 46)
C_DB_HI = (90, 92, 98)
C_OUTLINE = (28, 32, 40)


def _bg():
    """Vertical-gradient studio background with a floor band."""
    bg = Image.new("RGB", (W, H), C_BG_TOP)
    top = np.array(C_BG_TOP, float)
    bot = np.array(C_BG_BOT, float)
    grad = np.zeros((H, 1, 3), np.uint8)
    for y in range(H):
        t = y / (H - 1)
        grad[y, 0] = (top * (1 - t) + bot * t).astype(np.uint8)
    bg = Image.fromarray(np.repeat(grad, W, axis=1))
    return bg


def capsule(d, p0, p1, width, fill, outline=None, ow=0):
    """Draw a rounded segment (line + end caps)."""
    if outline and ow:
        d.line([tuple(p0), tuple(p1)], fill=outline, width=width + 2 * ow)
        for p in (p0, p1):
            r = width / 2 + ow
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=outline)
    d.line([tuple(p0), tuple(p1)], fill=fill, width=width)
    for p in (p0, p1):
        r = width / 2
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill)


def render_frame(elbow_deg, shoulder_deg, world_to_px):
    img = _bg()
    d = ImageDraw.Draw(img, "RGBA")
    # floor band
    floor_y = world_to_px(np.array([0, -(SEG["torso"] + SEG["thigh"] + SEG["shank"])]))[1]
    d.rectangle([0, floor_y, W, H], fill=C_FLOOR)
    d.line([0, floor_y, W, floor_y], fill=(150, 154, 160), width=2 * SS)

    j = forward_kinematics(shoulder_deg, elbow_deg)
    P = {k: world_to_px(v) for k, v in j.items()}

    # soft contact shadow under the feet
    sh_w, sh_h = 0.46, 0.05
    cx = P["ankle"][0]
    sw = sh_w * SCALE
    d.ellipse([cx - sw, floor_y - sh_h * SCALE, cx + sw, floor_y + sh_h * SCALE],
              fill=C_SHADOW + (120,))

    lw = lambda m: max(2, int(m * SCALE))

    # far (left) leg + arm first (behind torso) -- static, faded
    capsule(d, P["hip"] + np.array([8 * SS, 0]), P["knee"] + np.array([8 * SS, 0]), lw(0.085), C_SHIRT_DK)
    capsule(d, P["knee"] + np.array([8 * SS, 0]), P["ankle"] + np.array([8 * SS, 0]), lw(0.07), C_SKIN_DK)
    far_E = P["S"] + np.array([-6 * SS, 0]) + (P["E"] - P["S"]) * 0.9
    capsule(d, P["S"] + np.array([-6 * SS, 0]), far_E, lw(0.075), C_FARARM)

    # near leg
    capsule(d, P["hip"], P["knee"], lw(0.10), C_SHIRT, C_OUTLINE, SS)
    capsule(d, P["knee"], P["ankle"], lw(0.08), C_SKIN, C_OUTLINE, SS)

    # torso (tapered hip->shoulder)
    pw = SEG["pelvis_w"] / 2 * SCALE
    sw2 = SHOULDER_W / 2 * SCALE
    hipP, shP = P["hip"], P["S"]
    torso = [(hipP[0] - pw, hipP[1]), (hipP[0] + pw, hipP[1]),
             (shP[0] + sw2, shP[1]), (shP[0] - sw2, shP[1])]
    d.polygon(torso, fill=C_SHIRT, outline=C_OUTLINE)

    # neck + head
    capsule(d, P["S"], P["neck"], lw(0.07), C_SKIN, C_OUTLINE, SS)
    hr = SEG["head_r"] * SCALE
    hc = P["head"]
    d.ellipse([hc[0] - hr, hc[1] - hr, hc[0] + hr, hc[1] + hr], fill=C_SKIN, outline=C_OUTLINE, width=SS)
    # nose (faces +x)
    d.polygon([(hc[0] + hr * 0.9, hc[1] - hr * 0.1), (hc[0] + hr * 1.18, hc[1]),
               (hc[0] + hr * 0.9, hc[1] + hr * 0.15)], fill=C_SKIN_DK)

    # NEAR (right) arm -- the working arm
    capsule(d, P["S"], P["E"], lw(0.10), C_SKIN, C_OUTLINE, SS)
    capsule(d, P["E"], P["W"], lw(0.085), C_SKIN, C_OUTLINE, SS)
    capsule(d, P["W"], P["Hd"], lw(0.07), C_SKIN_DK, C_OUTLINE, SS)

    # dumbbell at the hand (bar + two plates)
    dbc = P["Hd"]
    bar_dir = (P["Hd"] - P["W"])
    bar_dir = bar_dir / (np.linalg.norm(bar_dir) + 1e-9)
    perp = np.array([-bar_dir[1], bar_dir[0]])
    half = SEG["db_r"] * 1.7 * SCALE
    b0, b1 = dbc - perp * half, dbc + perp * half
    capsule(d, b0, b1, lw(0.03), C_DB_HI)
    pr = SEG["db_r"] * SCALE
    for c in (b0, b1):
        d.ellipse([c[0] - pr, c[1] - pr, c[0] + pr, c[1] + pr], fill=C_DB, outline=C_OUTLINE, width=SS)
        d.ellipse([c[0] - pr * 0.4, c[1] - pr * 0.4, c[0] + pr * 0.4, c[1] + pr * 0.4], fill=C_DB_HI)

    return img.resize((W_OUT, H_OUT), Image.LANCZOS)


# ----------------------------------------------------------- world->pixel map
SCALE = 0.0   # px per metre (set in main)


def make_mapper(joints_ref):
    """Centre the figure horizontally, anchor feet near the bottom."""
    global SCALE
    # total standing height span (head top to ankle)
    span = SEG["torso"] + SEG["thigh"] + SEG["shank"] + SEG["neck"] + SEG["head_r"] * 2.2
    SCALE = (H * 0.82) / span
    # origin (shoulder) pixel: put feet ~6% above frame bottom
    feet_y_world = -(SEG["torso"] + SEG["thigh"] + SEG["shank"])
    margin = H * 0.06
    oy = H - margin + feet_y_world * SCALE
    ox = W * 0.42

    def w2p(p):
        return np.array([ox + p[0] * SCALE, oy - p[1] * SCALE])
    return w2p


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot", default=os.path.join(ROOT, "Data", "paper_minjerk_fatigue_10cycles.mot"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--load", default="2")
    ap.add_argument("--reps", default="10")
    ap.add_argument("--still", type=float, default=2.0, help="seconds still before/after")
    ap.add_argument("--subject", default="S01")
    args = ap.parse_args()

    cols, data = read_mot(args.mot)
    t = data[:, 0]
    elev = data[:, cols.index("r_shoulder_elev")]
    elb = data[:, cols.index("r_elbow_flex")]
    dur = float(t[-1] - t[0])

    out = args.out or os.path.join(
        ROOT, "Data", "synthetic_videos",
        f"{args.subject}_right_{args.load}kg_side_{args.reps}rep_take01.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fps = args.fps
    n_still = int(round(args.still * fps))
    n_move = int(round(dur * fps))
    # video sample times over the motion
    tv = np.linspace(t[0], t[-1], n_move)
    elb_v = np.interp(tv, t, elb)
    elev_v = np.interp(tv, t, elev)

    w2p = make_mapper(forward_kinematics(elev[0], elb[0]))

    ff = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{W_OUT}x{H_OUT}", "-framerate", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-preset", "medium", "-movflags", "+faststart", out,
    ]
    proc = subprocess.Popen(ff, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    total = n_still + n_move + n_still
    # cache the still frame
    still0 = render_frame(elb[0], elev[0], w2p).tobytes()
    stillN = render_frame(elb[-1], elev[-1], w2p).tobytes()

    done = 0

    def emit(b):
        nonlocal done
        proc.stdin.write(b)
        done += 1
        if done % 120 == 0 or done == total:
            sys.stdout.write(f"\r  frame {done}/{total} ({100*done/total:4.1f}%)")
            sys.stdout.flush()

    for _ in range(n_still):
        emit(still0)
    for k in range(n_move):
        emit(render_frame(elb_v[k], elev_v[k], w2p).tobytes())
    for _ in range(n_still):
        emit(stillN)

    proc.stdin.close()
    proc.wait()
    print()
    sz = os.path.getsize(out) / 1e6
    print(f"Wrote {out}  ({sz:.1f} MB, {total} frames, {total/fps:.1f} s @ {fps} fps)")
    print(f"  elbow ROM in clip: {elb.min():.1f}-{elb.max():.1f} deg over {dur:.1f} s")


if __name__ == "__main__":
    main()
