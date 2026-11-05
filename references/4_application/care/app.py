"""The care service: clinical records, plus the clinician dashboard that reads them.

    uvicorn care.app:app --host 127.0.0.1 --port 8100

Assembly only. Every behaviour lives in the module it belongs to; what this file
decides is which routers exist and what is served as static files.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from care.config import DASHBOARD_DIR, VIDEO_DIR
from care.database import create_tables
from care.routes import auth, exercises, my, patients, prescriptions
from care.seed import seed_demo_data


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_tables()
    seed_demo_data()
    yield


app = FastAPI(title="VisionBiomech Care API", lifespan=lifespan)

# The patient app is served from another origin (the analysis service, port
# 8000) and must be able to call this one.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.include_router(auth.router)
app.include_router(exercises.router)
app.include_router(patients.router)
app.include_router(prescriptions.router)
app.include_router(my.router)


class UncachedStaticFiles(StaticFiles):
    """Static files the browser is told never to keep.

    The dashboard is one HTML file carrying its own script, so a stale copy in
    the cache looks exactly like a broken feature: the button is there, the
    click does nothing, and no error is shown.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


# Uploaded recordings keep normal caching: they are immutable once written.
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

if os.path.isdir(DASHBOARD_DIR):
    app.mount("/", UncachedStaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
