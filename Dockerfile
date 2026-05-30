# ---------------------------------------------------------------------------
# RAG Demo — reproducible environment
#
# We pin Python 3.12 here on purpose. The host machine might have Python 3.14
# (or 3.9, or none at all) — it doesn't matter. Everything runs INSIDE this
# image with the exact same interpreter and dependency versions on every
# computer: Windows, macOS, or Linux.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# Keep Python output unbuffered (so logs show up immediately) and skip .pyc files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Some Python wheels (chromadb's deps, pypdf, etc.) occasionally need basic
# build tools. Installing them keeps `pip install` from failing on edge cases.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (this layer is cached unless requirements change),
# which makes rebuilds fast.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project.
COPY . .

# The local vector database is persisted to /app/chroma_db, which we mount as a
# volume in docker-compose so your embeddings survive container restarts.
ENV CHROMA_DIR=/app/chroma_db

# Default to showing the CLI help. Real commands are passed in via
# docker compose run (see README).
CMD ["python", "-m", "src.cli", "--help"]