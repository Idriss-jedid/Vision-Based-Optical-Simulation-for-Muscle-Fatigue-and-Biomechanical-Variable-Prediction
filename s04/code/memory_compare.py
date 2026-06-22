# pyright: reportMissingImports=false
"""Mesure le pic de RAM (RSS) : mode 'ai' (load joblib + predict) vs mode 'osim' (load modele + SO court).
Usage: python memory_compare.py ai   |   python memory_compare.py osim"""
import sys, os, time, threading
import psutil
proc = psutil.Process()
ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"; B = os.path.join(ROOT, "batch")
mode = sys.argv[1] if len(sys.argv) > 1 else "ai"

base = proc.memory_info().rss; peak = [base]; stop = [False]
def sampler():
    while not stop[0]:
        try: peak[0] = max(peak[0], proc.memory_info().rss)
        except Exception: pass
        time.sleep(0.02)
th = threading.Thread(target=sampler); th.start()

if mode == "ai":
    import numpy as np, joblib
    b = joblib.load(os.path.join(B, "model_3d_final", "lgbm_3d_vision.joblib"))
    X = np.random.randn(1759, len(b["features"])).astype("float32")
    Xs = b["x_scaler"].transform(X)
    _ = b["y_scaler"].inverse_transform(b["model"].predict(Xs))
else:
    sys.path.insert(0, os.path.join(ROOT, "Code"))
    import run_stage2_pipeline as P
    motion = os.path.join(B, "s04", "motion", "curl.mot")
    scaled = os.path.join(B, "s04", "opensim", "arm26_s04_scaled.osim")
    tmp = os.path.join(B, "s04", "_mem"); os.makedirs(tmp, exist_ok=True)
    t0, _ = P.motion_range(motion)
    mp = P.prep_model(scaled, tmp)
    P.run_id(mp, motion, tmp, t0, t0 + 2.0)          # ID court
    P.run_so(mp, motion, tmp, "s04", t0, t0 + 2.0)   # SO court (capture le footprint)
    import shutil; shutil.rmtree(tmp, ignore_errors=True)

time.sleep(0.1); stop[0] = True; th.join()
print("%-6s base=%6.0f MB  peak=%6.0f MB  delta=%6.0f MB" % (mode, base / 1e6, peak[0] / 1e6, (peak[0] - base) / 1e6))
