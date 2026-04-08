# ── Base ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Set up Hugging Face User (Required for permissions) ──────────────────────
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# ── System deps (Run as root, then switch back) ──────────────────────────────
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
USER user

# ── Python deps ──────────────────────────────────────────────────────────────
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY --chown=user . .

# ── Expose port (Hugging Face Spaces uses 7860) ──────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────────
# Fixed the syntax error by putting this on one line
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Launch ───────────────────────────────────────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]