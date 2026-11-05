"""Assigning, adjusting and withdrawing an exercise plan."""
from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from care.database import get_session
from care.models import ExerciseSession, Prescription, User
from care.ownership import get_owned_patient, get_owned_prescription
from care.schemas import NewPrescriptionRequest, PrescriptionUpdateRequest
from care.security import require_doctor

router = APIRouter(tags=["prescriptions"])


@router.post("/prescriptions")
def create_prescription(
    body: NewPrescriptionRequest,
    doctor: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    get_owned_patient(body.patient_id, doctor, session)

    prescription = Prescription(
        patient_id=body.patient_id,
        exercise_id=body.exercise_id,
        reps=body.reps,
        sets=body.sets,
        schedule=body.schedule,
        notes=body.notes,
        created_by=doctor.id,
    )
    session.add(prescription)
    session.commit()
    session.refresh(prescription)
    return prescription


@router.put("/prescriptions/{prescription_id}")
def update_prescription(
    prescription_id: int,
    body: PrescriptionUpdateRequest,
    doctor: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    prescription = get_owned_prescription(prescription_id, doctor, session)

    # exclude_none so an omitted field keeps its stored value: this is a partial
    # update, and sending only `active` must not blank the notes.
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(prescription, field, value)

    session.add(prescription)
    session.commit()
    session.refresh(prescription)
    return prescription


@router.delete("/prescriptions/{prescription_id}")
def delete_prescription(
    prescription_id: int,
    doctor: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    """Withdraw an assignment.

    Sessions already completed under it keep their results: their
    prescription_id is cleared rather than left pointing at a deleted row, so
    the history stays readable after the plan changes.
    """
    prescription = get_owned_prescription(prescription_id, doctor, session)

    completed = session.exec(
        select(ExerciseSession).where(ExerciseSession.prescription_id == prescription_id)
    ).all()
    for exercise_session in completed:
        exercise_session.prescription_id = None
        session.add(exercise_session)

    session.delete(prescription)
    session.commit()
    return {
        "deleted": "prescription",
        "id": prescription_id,
        "sessions_detached": len(completed),
    }
