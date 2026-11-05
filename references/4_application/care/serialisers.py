"""Turning stored rows into the shapes the two interfaces expect.

Kept in one place so a field can never be exposed by one endpoint and withheld
by another: the password hash is absent here, which is what makes it absent
everywhere.
"""
import json

from care.models import ExerciseSession, User


def public_user(user: User) -> dict:
    """A user as any client may see them. Never includes the password hash."""
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "doctor_id": user.doctor_id,
    }


def stored_video_names(session_row: ExerciseSession) -> list[str]:
    """The recordings of a session, as file names on disk."""
    return json.loads(session_row.videos_json or "[]")


def public_session(session_row: ExerciseSession) -> dict:
    """A session with its JSON columns decoded and its videos exposed as URLs.

    The database stores bare file names; the client needs paths it can request.
    Building them here means the storage layout can change without touching
    either interface.
    """
    return {
        "id": session_row.id,
        "patient_id": session_row.patient_id,
        "exercise_id": session_row.exercise_id,
        "prescription_id": session_row.prescription_id,
        "mode": session_row.mode,
        "reps_done": session_row.reps_done,
        "results": json.loads(session_row.results_json or "{}"),
        "videos": [f"/videos/{name}" for name in stored_video_names(session_row)],
        "created_at": session_row.created_at.isoformat(),
    }
