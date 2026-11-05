"""Registration, sign-in, and the identity of the caller.

Paths are written in full at each decorator rather than assembled from a router
prefix, so the URL a client calls is readable here without looking elsewhere.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from care.config import Role
from care.database import get_session
from care.models import User
from care.schemas import LoginRequest, RegisterRequest
from care.security import (
    create_access_token,
    find_user_by_email,
    get_current_user,
    hash_password,
    normalise_email,
    verify_password,
)
from care.serialisers import public_user

router = APIRouter(tags=["auth"])


@router.post("/auth/register")
def register(body: RegisterRequest, session: Session = Depends(get_session)):
    if find_user_by_email(session, body.email):
        raise HTTPException(400, "email already used")

    # Only two roles exist, and anything unrecognised becomes a patient: a
    # client must not be able to award itself clinician rights by typo.
    role = Role.DOCTOR if body.role == Role.DOCTOR else Role.PATIENT
    user = User(
        name=body.name,
        email=normalise_email(body.email),
        hashed_password=hash_password(body.password),
        role=role,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return {"token": create_access_token(user.id, user.role), "user": public_user(user)}


@router.post("/auth/login")
def login(body: LoginRequest, session: Session = Depends(get_session)):
    user = find_user_by_email(session, body.email)
    # One message for both failures, so the response cannot be used to discover
    # which addresses have an account.
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "wrong email or password")
    return {"token": create_access_token(user.id, user.role), "user": public_user(user)}


@router.get("/me")
def read_current_user(user: User = Depends(get_current_user)):
    return public_user(user)
