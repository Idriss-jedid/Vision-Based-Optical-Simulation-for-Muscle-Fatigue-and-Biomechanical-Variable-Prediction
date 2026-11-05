"""Request bodies.

These are separate from the table models on purpose: a client may never set an
id, a password hash, an owner or a creation date, and the only reliable way to
guarantee that is to not accept those fields in the first place.
"""
from typing import Optional

from pydantic import BaseModel

from care.config import RecordingMode, Role


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = Role.PATIENT


class LoginRequest(BaseModel):
    email: str
    password: str


class NewPatientRequest(BaseModel):
    name: str
    email: str
    password: str


class PatientUpdateRequest(BaseModel):
    """Every field optional: this is a partial update, and an omitted field
    means "leave it alone" rather than "clear it"."""

    name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None          # only rehashed when present


class ExerciseRequest(BaseModel):
    name: str
    description: str = ""
    muscles: str = ""
    level: str = RecordingMode.SINGLE_CAMERA


class NewPrescriptionRequest(BaseModel):
    patient_id: int
    exercise_id: int
    reps: int = 10
    sets: int = 3
    schedule: str = ""
    notes: str = ""


class PrescriptionUpdateRequest(BaseModel):
    exercise_id: Optional[int] = None
    reps: Optional[int] = None
    sets: Optional[int] = None
    schedule: Optional[str] = None
    notes: Optional[str] = None
    active: Optional[bool] = None


class NewSessionRequest(BaseModel):
    """A finished session. The patient is taken from the token, never from the
    body, so one patient cannot file a session against another."""

    exercise_id: int
    prescription_id: Optional[int] = None
    mode: str = RecordingMode.SINGLE_CAMERA
    reps_done: int = 0
    results: dict = {}
