# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — installs all Python deps with uv
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Install system build tools needed for llama-cpp-python & faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and install into /workspace/.venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — download models during build, then serve
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Install minimal runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy installed packages from builder stage
COPY --from=builder /workspace/.venv /workspace/.venv

# Make the venv the active Python
ENV PATH="/workspace/.venv/bin:$PATH"
ENV PYTHONPATH="/workspace"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HF cache lives inside /workspace so it's baked into the image layer
ENV HF_HOME=/workspace/.cache/huggingface

# ── Copy scripts and download models NOW (at build time) ──────────────────
COPY scripts/ /workspace/scripts/
RUN python scripts/04_download_model.py

# ── Copy data (pre-built graph + FAISS index) ─────────────────────────────
COPY data/ /workspace/data/

# ── Copy application source ────────────────────────────────────────────────
COPY app/ /workspace/app/

# ── Expose port (Mandatory for Hackathon) ──────────────────────────────────
EXPOSE 8000

# ── Health check so orchestrators know when the model is loaded ────────────
HEALTHCHECK --interval=15s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Start server ───────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]