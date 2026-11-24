# syntax=docker/dockerfile:1
FROM python:3.10-slim

LABEL org.opencontainers.image.title="Credit Risk Scorecard" \
      org.opencontainers.image.description="FastAPI service and web UI for credit default scoring" \
      org.opencontainers.image.source="https://github.com/thaitri2005/Credit-Scorecard-for-Default-Prediction" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Apply pending security updates to the base image. No extra system packages
# are needed: every Python dependency ships a wheel and the health check uses
# Python itself.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get upgrade -y

# Install dependencies before copying the source so this layer is reused until
# requirements.txt changes. Then remove the packaging tools: the running
# service never installs anything, and they only add to the attack surface.
RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
    && pip uninstall -y setuptools wheel pip \
    && rm -rf /usr/local/lib/python3.10/ensurepip/_bundled

# Run as an unprivileged user that owns nothing it does not need to write.
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --no-create-home app

COPY --chown=1000:1000 app/ ./app/
COPY --chown=1000:1000 models/ ./models/

USER app

# Azure App Service routes traffic to port 80.
EXPOSE 80

# /api/v1/health returns 503 when the model is not loaded, which fails this check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:80/api/v1/health', timeout=5)"]

ENTRYPOINT ["uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "80"]
