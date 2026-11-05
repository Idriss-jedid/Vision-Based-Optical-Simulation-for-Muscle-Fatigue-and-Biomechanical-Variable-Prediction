# VisionBiomech — Application

Two services and two interfaces. One turns video into biomechanical signals; the
other keeps the clinical records that give those signals a purpose.

```
Application/
├── analysis/            the analysis service                        (port 8000)
│   ├── app.py           /analyze_single · /analyze · /health, and serves the patient app
│   ├── pose.py          RTMPose 2D keypoints from a video
│   ├── single_camera.py one view -> BiGRU -> 13 signals   (the deployed route)
│   ├── multi_camera.py  four views -> triangulation -> angles -> Level-1 model
│   ├── calibration.py   reading a camera-array Calib.toml
│   ├── signals.py       labelling and packaging the 13 outputs
│   ├── config.py        paths, strides, constants
│   └── weights/         single_camera_bigru.pt · multicamera_lgbm.joblib
│
├── care/                the care service: clinical records          (port 8100)
│   ├── app.py           router assembly and static mounts
│   ├── models.py        User · Exercise · Prescription · ExerciseSession
│   ├── schemas.py       request bodies
│   ├── security.py      hashing, tokens, the two authorisation dependencies
│   ├── ownership.py     the checks that isolate one clinician's records
│   ├── serialisers.py   rows -> what the interfaces receive
│   ├── database.py      engine and per-request session
│   ├── seed.py          demo accounts and starting catalogue
│   ├── config.py        paths, secret, roles
│   ├── routes/          auth · exercises · patients · prescriptions · my
│   ├── dashboard/       the clinician web interface
│   ├── rehab.db         SQLite: user · exercise · prescription · sess
│   └── videos/          recordings uploaded with a session
│
├── patient_app/         the patient progressive web app
│   ├── index.html       login, then record and analyse
│   ├── js/              app.js · rehab.js (care API) · cv.js (in-browser pipeline)
│   ├── models/          ONNX: yolox-tiny · rtmpose-s · l3
│   └── videos/          sample recordings, loaded by ?demo
│
├── tests/               test_care_api.py (25 tests) · smoke_live_api.py
└── requirements.txt
```

## Run

Two servers, both from this directory, both in the `biomech` environment.

```bash
# analysis + patient app
python -m uvicorn analysis.app:app --host 127.0.0.1 --port 8000

# care API + clinician dashboard
python -m uvicorn care.app:app --host 127.0.0.1 --port 8100
```

| | URL | Account |
|---|---|---|
| **Patient** | <http://127.0.0.1:8000/> | `patient@demo.com` / `demo1234` |
| Patient, with a sample video | <http://127.0.0.1:8000/?demo> | " |
| **Clinician** | <http://127.0.0.1:8100/> | `doctor@demo.com` / `demo1234` |

## How the two services relate

The patient app is served by the analysis service, so its call to
`/analyze_single` is same-origin. Everything to do with the care workflow —
signing in, reading prescriptions, filing a session — goes to the care service
on port 8100, which owns the database.

```
patient app (:8000) ── /analyze_single ──▶ analysis service   (video -> signals)
                    └─ /auth/login, /my/* ─▶ care service      (shared records)
clinician   (:8100) ── /patients, /prescriptions, /patients/{id}/sessions
```

That shared database is what makes them synchronise: the clinician assigns an
exercise on 8100 and it appears on the patient's home screen; the patient files a
session and it appears in the clinician's history.

## Tests

```bash
python -m pytest tests/test_care_api.py -q      # 25 tests, in-process, fresh database
```

`tests/smoke_live_api.py` is a separate end-to-end check that expects the care
service to already be running on port 8100.

## Notes

- Analysis runs on the CPU. A short clip takes a few seconds per camera; the
  first run also downloads the RTMPose weights into the rtmlib cache.
- `/analyze_single` is the deployed route and the one the patient app uses.
- **Known defect, `/analyze` only:** the multi-camera route builds the elbow
  angle as `180 - angle(upper_arm, forearm)`, so an extended arm reads 180°,
  while the Level-1 model was trained on the OpenSim convention where an
  extended arm is 0°. It therefore feeds an inverted angle and its numbers should
  not be trusted until that is corrected. The single-camera route is unaffected:
  it consumes raw 2D keypoints. See `analysis/multi_camera.build_feature_table`.
