# The analysis service, with the patient app it serves.
#
#   docker build -t visionbiomech-analysis .
#   docker run -p 8000:8000 visionbiomech-analysis
#
# CPU only: the deployed single-camera route is RTMPose-s through onnxruntime
# followed by a small recurrent network, and neither needs a GPU.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# libgl and glib are OpenCV's runtime dependencies; ffmpeg decodes the uploads.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependencies first, so a code change does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The service, its weights, and the app it serves.
COPY analysis/ /srv/analysis/
COPY patient_app/ /srv/patient_app/

EXPOSE 8000
CMD ["uvicorn", "analysis.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
