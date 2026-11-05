"""Demo accounts and a starting catalogue.

Runs on every start but does nothing once the database holds a user, so a real
deployment is seeded once and never overwritten.
"""
from sqlmodel import Session, select

from care.config import RecordingMode, Role
from care.database import engine
from care.models import Exercise, Prescription, User
from care.security import hash_password

DEMO_PASSWORD = "demo1234"
DEMO_DOCTOR_EMAIL = "doctor@demo.com"
DEMO_PATIENT_EMAIL = "patient@demo.com"


def seed_demo_data() -> None:
    with Session(engine) as session:
        if session.exec(select(User)).first():
            return                          # already seeded

        exercises = [
            Exercise(
                name="Biceps curl",
                description="Dumbbell biceps curl, side view",
                muscles="BIClong, BICshort, BRA, BRD",
                level=RecordingMode.SINGLE_CAMERA,
            ),
            Exercise(
                name="Shoulder flexion",
                description="Raise the arm forward",
                muscles="Deltoid",
                level=RecordingMode.SINGLE_CAMERA,
            ),
            Exercise(
                name="Elbow extension",
                description="Straighten the elbow",
                muscles="Triceps",
                level=RecordingMode.SINGLE_CAMERA,
            ),
        ]
        for exercise in exercises:
            session.add(exercise)

        doctor = User(
            name="Dr. Demo",
            email=DEMO_DOCTOR_EMAIL,
            hashed_password=hash_password(DEMO_PASSWORD),
            role=Role.DOCTOR,
        )
        session.add(doctor)
        session.commit()
        session.refresh(doctor)

        patient = User(
            name="Ahmed Patient",
            email=DEMO_PATIENT_EMAIL,
            hashed_password=hash_password(DEMO_PASSWORD),
            role=Role.PATIENT,
            doctor_id=doctor.id,
        )
        session.add(patient)
        session.commit()
        session.refresh(patient)
        session.refresh(exercises[0])

        session.add(
            Prescription(
                patient_id=patient.id,
                exercise_id=exercises[0].id,
                reps=10,
                sets=3,
                schedule="Mon,Wed,Fri 18:00",
                notes="Slow and controlled",
                created_by=doctor.id,
            )
        )
        session.commit()
        print(f"seeded: {DEMO_DOCTOR_EMAIL} / {DEMO_PATIENT_EMAIL}  (password {DEMO_PASSWORD})")
