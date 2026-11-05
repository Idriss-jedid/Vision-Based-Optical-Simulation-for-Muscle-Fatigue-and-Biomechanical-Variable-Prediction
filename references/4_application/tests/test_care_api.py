# -*- coding: utf-8 -*-
"""Backend test suite (pytest + FastAPI TestClient, in-process, fresh DB per test)."""
import os
import sys

# The environment must be set before care.config is imported: it reads both at
# import time, and a test must never touch the real database or sign tokens with
# the production secret.
os.environ["CARE_JWT_SECRET"] = "test-secret"
os.environ["CARE_DATABASE_URL"] = "sqlite:///" + os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_rehab.db"
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel

from care.app import app as care_app
from care.database import engine

DEMO = {"vars": {"elbow_moment": {"group": "torque", "label": "Elbow torque", "unit": "N·m",
                                  "pred": [1, 4, 8, 3, 7, 2, 6, 1]},
                 "MF_BIClong": {"group": "fatigue", "label": "Fatigue", "unit": "%", "pred": [0, 1, 2, 2.8]}},
        "time": [0, 1, 2, 3, 4, 5, 6, 7]}


@pytest.fixture()
def c():
    SQLModel.metadata.drop_all(engine)          # fresh database
    with TestClient(care_app) as client:             # startup -> create tables + seed
        yield client


def tok(c, email, pw="demo1234"):
    r = c.post("/auth/login", json={"email": email, "password": pw})
    assert r.status_code == 200, r.text
    return r.json()["token"]

def H(t):
    return {"Authorization": "Bearer " + t}


# ---------------- auth ----------------
def test_seed_accounts_exist(c):
    assert c.post("/auth/login", json={"email": "doctor@demo.com", "password": "demo1234"}).status_code == 200
    assert c.post("/auth/login", json={"email": "patient@demo.com", "password": "demo1234"}).status_code == 200

def test_login_wrong_password(c):
    assert c.post("/auth/login", json={"email": "doctor@demo.com", "password": "nope"}).status_code == 401

def test_login_unknown_email(c):
    assert c.post("/auth/login", json={"email": "ghost@x.com", "password": "x"}).status_code == 401

def test_register_patient_and_login(c):
    r = c.post("/auth/register", json={"name": "New", "email": "new@x.com", "password": "pw12345"})
    assert r.status_code == 200 and r.json()["user"]["role"] == "patient"
    assert c.post("/auth/login", json={"email": "new@x.com", "password": "pw12345"}).status_code == 200

def test_register_duplicate_email(c):
    assert c.post("/auth/register", json={"name": "A", "email": "doctor@demo.com", "password": "x"}).status_code == 400

def test_register_doctor_role(c):
    r = c.post("/auth/register", json={"name": "Doc2", "email": "doc2@x.com", "password": "pw12345", "role": "doctor"})
    assert r.json()["user"]["role"] == "doctor"


# ---------------- tokens / me ----------------
def test_me_requires_token(c):
    assert c.get("/me").status_code == 401

def test_me_invalid_token(c):
    assert c.get("/me", headers=H("garbage.token.here")).status_code == 401

def test_me_returns_profile(c):
    me = c.get("/me", headers=H(tok(c, "patient@demo.com"))).json()
    assert me["email"] == "patient@demo.com" and me["role"] == "patient"


# ---------------- exercises ----------------
def test_exercises_seeded(c):
    ex = c.get("/exercises", headers=H(tok(c, "doctor@demo.com"))).json()
    assert len(ex) >= 3 and any(e["name"] == "Biceps curl" for e in ex)

def test_doctor_creates_exercise(c):
    d = tok(c, "doctor@demo.com")
    r = c.post("/exercises", headers=H(d), json={"name": "Wrist curl", "muscles": "forearm", "level": "single"})
    assert r.status_code == 200 and r.json()["name"] == "Wrist curl"
    assert any(e["name"] == "Wrist curl" for e in c.get("/exercises", headers=H(d)).json())

def test_patient_cannot_create_exercise(c):
    assert c.post("/exercises", headers=H(tok(c, "patient@demo.com")), json={"name": "X"}).status_code == 403

def test_session_stores_mode(c):
    p = tok(c, "patient@demo.com"); ex = c.get("/exercises", headers=H(p)).json()
    c.post("/my/sessions", headers=H(p), json={"exercise_id": ex[0]["id"], "mode": "multi", "reps_done": 6, "results": DEMO})
    assert c.get("/my/sessions", headers=H(p)).json()[0]["mode"] == "multi"


# ---------------- doctor: patients ----------------
def test_doctor_sees_only_own_patients(c):
    d1 = tok(c, "doctor@demo.com")
    # a second doctor with their own patient
    c.post("/auth/register", json={"name": "Doc2", "email": "doc2@x.com", "password": "pw12345", "role": "doctor"})
    d2 = tok(c, "doc2@x.com", "pw12345")
    c.post("/patients", headers=H(d2), json={"name": "P2", "email": "p2@x.com", "password": "pw12345"})
    names1 = [p["name"] for p in c.get("/patients", headers=H(d1)).json()]
    names2 = [p["name"] for p in c.get("/patients", headers=H(d2)).json()]
    assert "Ahmed Patient" in names1 and "P2" not in names1
    assert "P2" in names2 and "Ahmed Patient" not in names2

def test_doctor_creates_patient(c):
    d = tok(c, "doctor@demo.com")
    r = c.post("/patients", headers=H(d), json={"name": "Sami", "email": "sami@x.com", "password": "pw12345"})
    assert r.status_code == 200 and r.json()["role"] == "patient"
    assert c.post("/auth/login", json={"email": "sami@x.com", "password": "pw12345"}).status_code == 200

def test_create_patient_duplicate_email(c):
    d = tok(c, "doctor@demo.com")
    assert c.post("/patients", headers=H(d), json={"name": "X", "email": "patient@demo.com", "password": "pw12345"}).status_code == 400


# ---------------- prescriptions ----------------
def test_doctor_assigns_and_patient_sees(c):
    d = tok(c, "doctor@demo.com")
    pid = c.get("/patients", headers=H(d)).json()[0]["id"]
    ex = c.get("/exercises", headers=H(d)).json()
    r = c.post("/prescriptions", headers=H(d), json={"patient_id": pid, "exercise_id": ex[1]["id"], "reps": 15, "sets": 4, "schedule": "Daily 09:00"})
    assert r.status_code == 200 and r.json()["reps"] == 15
    my = c.get("/my/prescriptions", headers=H(tok(c, "patient@demo.com"))).json()
    assert any(p["reps"] == 15 and p["exercise"]["id"] == ex[1]["id"] for p in my)

def test_cannot_prescribe_to_other_doctors_patient(c):
    c.post("/auth/register", json={"name": "Doc2", "email": "doc2@x.com", "password": "pw12345", "role": "doctor"})
    d2 = tok(c, "doc2@x.com", "pw12345")
    ahmed = c.get("/patients", headers=H(tok(c, "doctor@demo.com"))).json()[0]["id"]
    ex = c.get("/exercises", headers=H(d2)).json()
    assert c.post("/prescriptions", headers=H(d2), json={"patient_id": ahmed, "exercise_id": ex[0]["id"], "reps": 5}).status_code == 404


# ---------------- sessions / history ----------------
def test_patient_records_session_and_history(c):
    p = tok(c, "patient@demo.com")
    ex = c.get("/exercises", headers=H(p)).json()
    r = c.post("/my/sessions", headers=H(p), json={"exercise_id": ex[0]["id"], "reps_done": 9, "results": DEMO})
    assert r.status_code == 200 and r.json()["reps_done"] == 9
    hist = c.get("/my/sessions", headers=H(p)).json()
    assert len(hist) == 1 and hist[0]["results"]["time"] == DEMO["time"]

def test_doctor_sees_patient_sessions(c):
    p = tok(c, "patient@demo.com"); d = tok(c, "doctor@demo.com")
    ex = c.get("/exercises", headers=H(p)).json()
    c.post("/my/sessions", headers=H(p), json={"exercise_id": ex[0]["id"], "reps_done": 7, "results": DEMO})
    pid = c.get("/patients", headers=H(d)).json()[0]["id"]
    dh = c.get(f"/patients/{pid}/sessions", headers=H(d)).json()
    assert len(dh) == 1 and dh[0]["reps_done"] == 7

def test_results_json_roundtrip(c):
    p = tok(c, "patient@demo.com"); ex = c.get("/exercises", headers=H(p)).json()
    sid = c.post("/my/sessions", headers=H(p), json={"exercise_id": ex[0]["id"], "reps_done": 5, "results": DEMO}).json()["id"]
    got = [s for s in c.get("/my/sessions", headers=H(p)).json() if s["id"] == sid][0]
    assert got["results"]["vars"]["elbow_moment"]["pred"] == DEMO["vars"]["elbow_moment"]["pred"]

def test_video_upload(c):
    p = tok(c, "patient@demo.com"); ex = c.get("/exercises", headers=H(p)).json()
    sid = c.post("/my/sessions", headers=H(p), json={"exercise_id": ex[0]["id"], "reps_done": 5, "results": DEMO}).json()["id"]
    r = c.post(f"/my/sessions/{sid}/video", headers=H(p), files={"file": ("v.mp4", b"\x00\x01\x02fakevideo", "video/mp4")})
    assert r.status_code == 200 and r.json()["videos"][0].startswith("/videos/")
    assert any(s["id"] == sid and s["videos"] for s in c.get("/my/sessions", headers=H(p)).json())


# ---------------- security / roles ----------------
def test_patient_blocked_from_doctor_routes(c):
    p = H(tok(c, "patient@demo.com"))
    assert c.get("/patients", headers=p).status_code == 403
    assert c.post("/patients", headers=p, json={"name": "x", "email": "x@x.com", "password": "pw12345"}).status_code == 403
    assert c.post("/prescriptions", headers=p, json={"patient_id": 1, "exercise_id": 1, "reps": 5}).status_code == 403

def test_doctor_cannot_read_unknown_patient(c):
    d = H(tok(c, "doctor@demo.com"))
    assert c.get("/patients/9999/sessions", headers=d).status_code == 404

def test_no_token_blocks_protected(c):
    assert c.get("/patients").status_code == 401
    assert c.get("/my/sessions").status_code == 401
    assert c.post("/my/sessions", json={"exercise_id": 1, "reps_done": 1, "results": {}}).status_code == 401
