"""The shared exercise catalogue.

Any signed-in user may read it, because a patient needs the name and the muscles
of what they have been asked to do. Only a clinician may change it.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from care.database import get_session
from care.models import Exercise, Prescription, User
from care.schemas import ExerciseRequest
from care.security import get_current_user, require_doctor

router = APIRouter(tags=["exercises"])


@router.get("/exercises")
def list_exercises(
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    return session.exec(select(Exercise)).all()


@router.post("/exercises")
def create_exercise(
    body: ExerciseRequest,
    _: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    exercise = Exercise(**body.model_dump())
    session.add(exercise)
    session.commit()
    session.refresh(exercise)
    return exercise


@router.put("/exercises/{exercise_id}")
def update_exercise(
    exercise_id: int,
    body: ExerciseRequest,
    _: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    exercise = session.get(Exercise, exercise_id)
    if not exercise:
        raise HTTPException(404, "exercise not found")

    for field, value in body.model_dump().items():
        setattr(exercise, field, value)
    session.add(exercise)
    session.commit()
    session.refresh(exercise)
    return exercise


@router.delete("/exercises/{exercise_id}")
def delete_exercise(
    exercise_id: int,
    _: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    """Deleting an exercise cascades to every prescription that assigns it, for
    every patient of every clinician: a prescription without its exercise is a
    dangling row the patient app cannot render.

    Completed sessions are deliberately kept. A finished session is a clinical
    record and its results stay readable; its exercise_id is left in place and
    resolves to 'deleted exercise' in the interfaces.
    """
    exercise = session.get(Exercise, exercise_id)
    if not exercise:
        raise HTTPException(404, "exercise not found")

    orphaned = session.exec(
        select(Prescription).where(Prescription.exercise_id == exercise_id)
    ).all()
    for prescription in orphaned:
        session.delete(prescription)
    session.delete(exercise)
    session.commit()
    return {
        "deleted": "exercise",
        "id": exercise_id,
        "prescriptions_removed": len(orphaned),
    }
