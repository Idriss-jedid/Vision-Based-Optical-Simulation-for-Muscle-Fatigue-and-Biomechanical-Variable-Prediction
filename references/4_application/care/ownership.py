"""The access checks that keep one clinician's records out of another's reach.

Both lookups answer 404 rather than 403 on a record belonging to someone else.
Saying "forbidden" would confirm that the id exists, which is enough to
enumerate another clinician's patients one number at a time.
"""
from fastapi import HTTPException
from sqlmodel import Session

from care.models import Prescription, User


def get_owned_patient(patient_id: int, doctor: User, session: Session) -> User:
    """The patient with this id, provided they belong to this clinician."""
    patient = session.get(User, patient_id)
    if not patient or patient.doctor_id != doctor.id:
        raise HTTPException(404, "patient not found")
    return patient


def get_owned_prescription(prescription_id: int, doctor: User, session: Session) -> Prescription:
    """The prescription with this id, provided its patient belongs to this
    clinician. Ownership of a prescription is never stored directly; it is
    always resolved through the patient, so there is one rule to reason about.
    """
    prescription = session.get(Prescription, prescription_id)
    if not prescription:
        raise HTTPException(404, "prescription not found")
    get_owned_patient(prescription.patient_id, doctor, session)
    return prescription
