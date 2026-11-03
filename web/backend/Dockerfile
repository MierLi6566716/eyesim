# Build context is the REPO ROOT, not this directory, so this image can
# `pip install -e` the eyesim package directly instead of reimplementing
# any physics/ML in the web layer:
#
#   docker build -f web/backend/Dockerfile -t eyesim-backend .
#   (or: fly deploy --dockerfile web/backend/Dockerfile, run from repo root)

FROM python:3.11-slim

WORKDIR /repo

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# eyesim itself (physics + PupilNet) — installed as a real dependency, not
# copy-pasted into the web layer.
COPY pyproject.toml README.md ./
COPY eyesim/ ./eyesim/
RUN pip install --no-cache-dir -e ".[ml]"

# Web backend's own dependencies (FastAPI, SQLAlchemy, Supabase client, ...)
COPY web/backend/requirements.txt ./web/backend/requirements.txt
RUN pip install --no-cache-dir -r web/backend/requirements.txt

# App code, migrations, the trained checkpoint, and the precomputed
# training-distribution stats used by Diagnose mode's confidence score.
COPY web/backend/app ./web/backend/app
COPY web/backend/alembic ./web/backend/alembic
COPY web/backend/alembic.ini ./web/backend/alembic.ini
COPY checkpoints/best.pt ./checkpoints/best.pt

WORKDIR /repo/web/backend
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
