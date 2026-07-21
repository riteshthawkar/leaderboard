FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV WEB_CONCURRENCY=1
ENV GUNICORN_THREADS=4
ENV GUNICORN_TIMEOUT=180
ENV FLASK_ENV=production
ENV DEPLOYMENT_MODE=public
ENV DISABLE_SUBMISSION_AUTH=false
ENV AUTH_DEV_MODE=false
ENV LEADERBOARD_DATA_DIR=/data
ENV RESULTS_DIR=/data/results
ENV DATABASE_URL=sqlite:////data/leaderboard.db
ENV AUTH_DATABASE_URL=
ENV SUBMISSION_DATABASE_URL=
ENV SQLITE_BUSY_TIMEOUT_MS=5000
ENV AUTO_BACKUP_ENABLED=true
ENV AUTO_BACKUP_DIR=/data/backups
ENV AUTO_BACKUP_MIRROR_DIR=
ENV REQUIRE_OFFSITE_BACKUP=true
ENV AUTO_BACKUP_INTERVAL_HOURS=48
ENV AUTO_BACKUP_RETENTION_COUNT=15
ENV AUTO_BACKUP_POLL_SECONDS=300
ENV AUTO_BACKUP_RUN_ON_START=true
ENV MIN_FREE_DISK_BYTES=1073741824
ENV MIN_FREE_DISK_PERCENT=5
ENV LIMITER_STORAGE_URI=memory://
ENV TRUST_PROXY_HOPS=0
ENV SESSION_COOKIE_SECURE=true
ENV SESSION_COOKIE_SAMESITE=Lax
ENV GROUND_TRUTHS_DIR=/data/ground_truths
ENV GROUND_TRUTHS_SOURCE=hf
ENV SPATIAL_GROUND_TRUTH_FILE=/data/ground_truths/spatial_v1/ground_truth.json
ENV REQUIRE_OFFICIAL_SPATIAL=false
ENV MAX_CONTENT_LENGTH=52428800
ENV MAX_FILE_SIZE_PER_SUBMISSION=52428800
ENV MAX_SPATIAL_ARCHIVE_BYTES=10485760
ENV MAX_SPATIAL_MULTIPART_BYTES=12582912
ENV MAX_SPATIAL_SUBMISSION_BYTES=20971520
ENV MAX_SPATIAL_MANIFEST_BYTES=1048576
ENV MAX_SPATIAL_ZIP_COMPRESSION_RATIO=100

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user

WORKDIR /home/user/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user backend ./backend
COPY --chown=user:user tasks ./tasks
COPY --chown=user:user evaluation/spatial_reasoning ./evaluation/spatial_reasoning
COPY --chown=user:user README.md ./README.md

RUN mkdir -p /data/results /data/backups /data/ground_truths /data/ground_truths_cache \
    && chown -R user:user /data /home/user/app \
    && chmod 700 /data /data/results /data/backups /data/ground_truths /data/ground_truths_cache

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT:-7860}/api/health/live || exit 1

CMD ["sh", "-c", "gunicorn backend.web:app --bind 0.0.0.0:${PORT:-7860} --workers ${WEB_CONCURRENCY:-1} --worker-class gthread --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-180}"]
