"""Configuration for the care service.

Everything the service needs to locate or authenticate is resolved here, once,
so no other module builds a path or reads an environment variable.
"""
import os
import secrets

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the clinical records live. Overridable so the tests can point at a
# throwaway file instead of the real database.
DATABASE_URL = os.environ.get(
    "CARE_DATABASE_URL", f"sqlite:///{os.path.join(PACKAGE_DIR, 'rehab.db')}"
)

# Uploaded session recordings. Created on import: the upload endpoint must not
# be the thing that discovers the directory is missing.
VIDEO_DIR = os.path.join(PACKAGE_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

DASHBOARD_DIR = os.path.join(PACKAGE_DIR, "dashboard")

# A generated secret means tokens do not survive a restart, which is the safe
# default for a demo. A deployment sets CARE_JWT_SECRET and keeps sessions alive.
JWT_SECRET = os.environ.get("CARE_JWT_SECRET", "change-me-in-production-" + secrets.token_hex(8))
JWT_ALGORITHM = "HS256"
TOKEN_LIFETIME_HOURS = 24 * 7


class Role:
    """The two account types.

    These strings are stored in the database and sent to the browser, so they
    are part of the contract and cannot be renamed without a migration.
    """

    PATIENT = "patient"
    DOCTOR = "doctor"


class RecordingMode:
    """How a session was captured. Stored on the session row."""

    SINGLE_CAMERA = "single"
    MULTI_CAMERA = "multi"
