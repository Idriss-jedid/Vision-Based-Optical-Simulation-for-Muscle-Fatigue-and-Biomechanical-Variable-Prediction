"""Password hashing, access tokens, and the two authorisation dependencies.

Every protected endpoint depends on `get_current_user`, and every clinician-only
endpoint on `require_doctor`. Neither reads the request body, so no client can
influence who it is by what it sends.
"""
import datetime
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlmodel import Session, select

from care.config import JWT_ALGORITHM, JWT_SECRET, TOKEN_LIFETIME_HOURS, Role
from care.database import get_session
from care.models import User

password_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_token = OAuth2PasswordBearer(tokenUrl="/auth/login")


def normalise_email(raw: str) -> str:
    """Emails are case-insensitive in practice and a phone keyboard will
    capitalise the first letter on its own. Every lookup and every insert goes
    through here, so an account created as 'Name@example.com' can sign in as
    'name@example.com'."""
    return (raw or "").strip().lower()


def hash_password(plain: str) -> str:
    return password_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return password_context.verify(plain, hashed)


def create_access_token(user_id: int, role: str) -> str:
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_LIFETIME_HOURS)
    return jwt.encode(
        {"sub": str(user_id), "role": role, "exp": expires_at},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def find_user_by_email(session: Session, email: str) -> Optional[User]:
    return session.exec(select(User).where(User.email == normalise_email(email))).first()


def get_current_user(
    token: str = Depends(bearer_token),
    session: Session = Depends(get_session),
) -> User:
    """Resolve the bearer token to a stored user.

    The role is re-read from the database rather than trusted from the token, so
    revoking a clinician takes effect on the next request instead of when the
    token expires.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(401, "invalid token")

    user = session.get(User, user_id)
    if not user:
        raise HTTPException(401, "user not found")
    return user


def require_doctor(user: User = Depends(get_current_user)) -> User:
    if user.role != Role.DOCTOR:
        raise HTTPException(403, "doctor only")
    return user
