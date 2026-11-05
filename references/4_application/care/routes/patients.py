"""Clinician-facing patient management.

Every endpoint here is reachable only by a clinician, and only for their own
patients. The pairing of `require_doctor` with `get_owned_patient` is what
enforces that, and it is repeated deliberately rather than hidden.
"""
import os

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from care.config import VIDEO_DIR, Role
from care.database import get_session
from care.models import ExerciseSession, Prescription, User
from care.ownership import get_owned_patient
from care.schemas import NewPatientRequest, PatientUpdateRequest
from care.security import find_user_by_email, hash_password, normalise_email, require_doctor
from care.serialisers import public_session, public_user, stored_video_names

router = APIRouter(tags=["patients"])


@router.get("/patients")
def list_my_patients(
    doctor: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    """This clinician's patients, each with how many sessions they have filed."""
    patients = session.exec(select(User).where(User.doctor_id == doctor.id)).all()
    listed = []
    for patient in patients:
        session_count = len(
            session.exec(
                select(ExerciseSession).where(ExerciseSession.patient_id == patient.id)
            ).all()
        )
        listed.append({**public_user(patient), "sessions": session_count})
    return listed


@router.post("/patients")
def create_patient(
    body: NewPatientRequest,
    doctor: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    """Create a patient account already attached to the calling clinician.

    Patients do not sign themselves up: the clinician issues the credentials,
    which is what makes the patient-clinician link trustworthy.
    """
    if find_user_by_email(session, body.email):
        raise HTTPException(400, "email already used")

    patient = User(
        name=body.name,
        email=normalise_email(body.email),
        hashed_password=hash_password(body.password),
        role=Role.PATIENT,
        doctor_id=doctor.id,
    )
    session.add(patient)
    session.commit()
    session.refresh(patient)
    return public_user(patient)


@router.put("/patients/{patient_id}")
def update_patient(
    patient_id: int,
    body: PatientUpdateRequest,
    doctor: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    patient = get_owned_patient(patient_id, doctor, session)

    if body.email and normalise_email(body.email) != patient.email:
        if find_user_by_email(session, body.email):
            raise HTTPException(400, "email already used")
        patient.email = normalise_email(body.email)
    if body.name:
        patient.name = body.name
    if body.password:
        patient.hashed_password = hash_password(body.password)

    session.add(patient)
    session.commit()
    session.refresh(patient)
    return public_user(patient)


@router.delete("/patients/{patient_id}")
def delete_patient(
    patient_id: int,
    doctor: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    """Delete a patient and everything that exists only because of them: their
    prescriptions, their sessions, and the recordings those sessions reference.

    Nothing removed here is shared with another patient, which is what makes the
    delete total, and also why it cannot be undone.
    """
    patient = get_owned_patient(patient_id, doctor, session)

    prescriptions = session.exec(
        select(Prescription).where(Prescription.patient_id == patient_id)
    ).all()
    sessions = session.exec(
        select(ExerciseSession).where(ExerciseSession.patient_id == patient_id)
    ).all()

    videos_removed = 0
    for exercise_session in sessions:
        for video_name in stored_video_names(exercise_session):
            try:
                os.remove(os.path.join(VIDEO_DIR, video_name))
                videos_removed += 1
            except OSError:
                pass                        # already gone; the row goes anyway
        session.delete(exercise_session)
    for prescription in prescriptions:
        session.delete(prescription)
    session.delete(patient)
    session.commit()

    return {
        "deleted": "patient",
        "id": patient_id,
        "prescriptions_removed": len(prescriptions),
        "sessions_removed": len(sessions),
        "videos_removed": videos_removed,
    }


@router.get("/patients/{patient_id}/prescriptions")
def list_patient_prescriptions(
    patient_id: int,
    doctor: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    get_owned_patient(patient_id, doctor, session)
    return session.exec(
        select(Prescription).where(Prescription.patient_id == patient_id)
    ).all()


@router.get("/patients/{patient_id}/sessions")
def list_patient_sessions(
    patient_id: int,
    doctor: User = Depends(require_doctor),
    session: Session = Depends(get_session),
):
    """A patient's history, newest first, as the dashboard displays it."""
    get_owned_patient(patient_id, doctor, session)
    rows = session.exec(
        select(ExerciseSession)
        .where(ExerciseSession.patient_id == patient_id)
        .order_by(ExerciseSession.created_at.desc())
    ).all()
    return [public_session(row) for row in rows]
