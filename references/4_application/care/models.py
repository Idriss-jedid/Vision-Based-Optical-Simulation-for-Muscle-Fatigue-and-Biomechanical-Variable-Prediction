"""The four persistent entities of the care workflow.

A clinician owns patients; a patient receives prescriptions; performing one
produces a session. Ownership is carried by plain integer columns rather than
ORM relationships, because every access check in this service is an explicit
comparison and hiding it behind lazy loading would make it easy to forget.
"""
import datetime
from typing import Optional

from sqlmodel import Field, SQLModel

from care.config import RecordingMode, Role


def _utc_now() -> datetime.datetime:
    return datetime.datetime.utcnow()


class User(SQLModel, table=True):
    """A clinician or a patient.

    One table serves both, distinguished by `role`, so a single authentication
    path covers the two interfaces. `doctor_id` is the reflexive link that makes
    a patient belong to a clinician, and it is what every ownership check reads.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str = Field(index=True, unique=True)
    hashed_password: str
    role: str = Role.PATIENT
    doctor_id: Optional[int] = None
    created_at: datetime.datetime = Field(default_factory=_utc_now)


class Exercise(SQLModel, table=True):
    """A movement in the shared catalogue.

    Exercises are global: defined once and prescribed to any patient by any
    clinician. There is deliberately no owner column.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str = ""
    muscles: str = ""
    level: str = RecordingMode.SINGLE_CAMERA


class Prescription(SQLModel, table=True):
    """An exercise assigned to a patient by a clinician."""

    id: Optional[int] = Field(default=None, primary_key=True)
    patient_id: int = Field(index=True)
    exercise_id: int
    reps: int = 10
    sets: int = 3
    schedule: str = ""                      # free text, e.g. "Mon,Wed,Fri 18:00"
    notes: str = ""
    created_by: int = 0                     # the prescribing clinician
    active: bool = True
    created_at: datetime.datetime = Field(default_factory=_utc_now)


class ExerciseSession(SQLModel, table=True):
    """One completed exercise, with its analysis results and recordings.

    `prescription_id` is optional so a session survives the removal of the plan
    it was performed under: the clinical record outlives the prescription.

    The results and the video list are stored as JSON text rather than as
    related tables. They are written once, always read whole, and never queried
    field by field, so a column earns nothing over a document here.
    """

    # The table predates this class name and holds real records, so the name is
    # pinned rather than derived.
    __tablename__ = "sess"

    id: Optional[int] = Field(default=None, primary_key=True)
    patient_id: int = Field(index=True)
    exercise_id: int
    prescription_id: Optional[int] = None
    mode: str = RecordingMode.SINGLE_CAMERA
    reps_done: int = 0
    results_json: str = "{}"                # the thirteen biomechanical signals
    videos_json: str = "[]"                 # stored file names, not URLs
    created_at: datetime.datetime = Field(default_factory=_utc_now)
