"""What a patient can reach about themselves.

Every endpoint scopes its query to the user carried by the token. There is no
patient id in any path here, which is what makes reaching another patient's
data impossible rather than merely forbidden.
"""
import json
import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlmodel import Session, select

from care.config import VIDEO_DIR
from care.database import get_session
from care.models import Exercise, ExerciseSession, Prescription, User
from care.schemas import NewSessionRequest
from care.security import get_current_user
from care.serialisers import public_session, stored_video_names

router = APIRouter(tags=["patient"])


@router.get("/my/prescriptions")
def list_my_prescriptions(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """The patient's active plans, each with its exercise inlined.

    The exercise travels with the prescription because the home screen shows
    both together, and a second round trip on a phone connection is worth
    avoiding.
    """
    prescriptions = session.exec(
        select(Prescription).where(
            Prescription.patient_id == user.id,
            Prescription.active == True,        # noqa: E712 - SQL needs the comparison
        )
    ).all()
    exercises = {exercise.id: exercise for exercise in session.exec(select(Exercise)).all()}

    return [
        {
            **prescription.model_dump(),
            "exercise": (
                exercises[prescription.exercise_id].model_dump()
                if prescription.exercise_id in exercises
                else None                        # the exercise was deleted
            ),
        }
        for prescription in prescriptions
    ]


@router.get("/my/sessions")
def list_my_sessions(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    rows = session.exec(
        select(ExerciseSession)
        .where(ExerciseSession.patient_id == user.id)
        .order_by(ExerciseSession.created_at.desc())
    ).all()
    return [public_session(row) for row in rows]


@router.post("/my/sessions")
def create_my_session(
    body: NewSessionRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Record a finished session.

    This is the first of the two requests that save a session: the results are
    stored here, and the recording is uploaded separately afterwards. That order
    is deliberate — if the upload fails, the clinically useful part is already
    safe.
    """
    exercise_session = ExerciseSession(
        patient_id=user.id,
        exercise_id=body.exercise_id,
        prescription_id=body.prescription_id,
        mode=body.mode,
        reps_done=body.reps_done,
        results_json=json.dumps(body.results),
    )
    session.add(exercise_session)
    session.commit()
    session.refresh(exercise_session)
    return public_session(exercise_session)


@router.post("/my/sessions/{session_id}/video")
def upload_my_session_video(
    session_id: int,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Attach a recording to a session the caller owns.

    The stored name is derived from the session id and the position in the list,
    never from the uploaded name alone, so two patients cannot collide and an
    uploaded name cannot escape the video directory.
    """
    exercise_session = session.get(ExerciseSession, session_id)
    if not exercise_session or exercise_session.patient_id != user.id:
        raise HTTPException(404, "session not found")

    video_names = stored_video_names(exercise_session)
    original = os.path.basename(file.filename or "video.mp4")
    stored_name = f"sess_{session_id}_{len(video_names)}_{original}"

    with open(os.path.join(VIDEO_DIR, stored_name), "wb") as destination:
        destination.write(file.file.read())

    video_names.append(stored_name)
    exercise_session.videos_json = json.dumps(video_names)
    session.add(exercise_session)
    session.commit()
    return {"videos": [f"/videos/{name}" for name in video_names]}
