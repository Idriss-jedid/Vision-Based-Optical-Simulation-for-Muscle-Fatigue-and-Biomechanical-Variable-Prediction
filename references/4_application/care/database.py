"""Database engine and the per-request session dependency."""
from collections.abc import Iterator

from sqlmodel import Session, SQLModel, create_engine

from care.config import DATABASE_URL

# check_same_thread=False because FastAPI serves requests from a thread pool and
# SQLite would otherwise refuse a connection opened on another thread.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def create_tables() -> None:
    """Create any missing table. Existing tables and their rows are untouched."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding one database session per request."""
    with Session(engine) as session:
        yield session
