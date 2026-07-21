FROM node:22-bookworm-slim AS frontend-build

WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./

ENV VITE_SAME_ORIGIN=1
ENV VITE_API_BASE_URL=
ENV VITE_PRIVACY_POLICY_URL=/privacy
RUN npm run build


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user
ENV PORT=7860
ENV API_PORT=5050
ENV WEB_CONCURRENCY=1
ENV GUNICORN_THREADS=4
ENV GUNICORN_TIMEOUT=180
ENV FLASK_ENV=production
ENV DEPLOYMENT_MODE=public
ENV DISABLE_SUBMISSION_AUTH=false
ENV AUTH_DEV_MODE=false
ENV TEST_SUBMISSION_USER=test-deployment@ms-vista.local
ENV LEADERBOARD_DATA_DIR=/data
ENV RESULTS_DIR=/data/results
ENV LEADERBOARD_LOG_DIR=/data/logs
ENV DATABASE_URL=sqlite:////data/leaderboard.db
ENV AUTH_DATABASE_URL=
ENV SUBMISSION_DATABASE_URL=
ENV SQLITE_BUSY_TIMEOUT_MS=5000
ENV AUTO_BACKUP_ENABLED=true
ENV AUTO_BACKUP_DIR=/data/backups
ENV AUTO_BACKUP_MIRROR_DIR=/backup
ENV REQUIRE_OFFSITE_BACKUP=true
ENV AUTO_BACKUP_INTERVAL_HOURS=48
ENV AUTO_BACKUP_RETENTION_COUNT=15
ENV AUTO_BACKUP_POLL_SECONDS=300
ENV AUTO_BACKUP_RUN_ON_START=true
ENV MIN_FREE_DISK_BYTES=1073741824
ENV MIN_FREE_DISK_PERCENT=5
ENV LIMITER_STORAGE_URI=memory://
ENV TRUST_PROXY_HOPS=1
ENV SESSION_COOKIE_SECURE=true
ENV SESSION_COOKIE_SAMESITE=Lax
ENV SESSION_COOKIE_NAME=ms_vista_session
ENV GROUND_TRUTHS_DIR=/data/ground_truths
ENV GROUND_TRUTHS_SOURCE=hf
ENV GROUND_TRUTHS_HF_CACHE_DIR=/data/ground_truths_cache
ENV HF_HOME=/data/hf_cache
ENV SPATIAL_GROUND_TRUTH_FILE=/data/ground_truths/spatial_v1/ground_truth.json
ENV REQUIRE_OFFICIAL_SPATIAL=false
ENV HF_REQUIRE_PERSISTENT_VOLUMES=true
ENV MAX_CONTENT_LENGTH=52428800
ENV MAX_FILE_SIZE_PER_SUBMISSION=52428800
ENV MAX_SPATIAL_ARCHIVE_BYTES=10485760
ENV MAX_SPATIAL_MULTIPART_BYTES=12582912
ENV MAX_SPATIAL_SUBMISSION_BYTES=20971520
ENV MAX_SPATIAL_MANIFEST_BYTES=1048576
ENV MAX_SPATIAL_ZIP_COMPRESSION_RATIO=100

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl dumb-init libmagic1 nginx supervisor util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default \
    && useradd --create-home --uid 1000 user

WORKDIR /home/user/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user backend ./backend
COPY --chown=user:user tasks ./tasks
COPY --chown=user:user evaluation/spatial_reasoning ./evaluation/spatial_reasoning
COPY --chown=user:user deployment/huggingface ./deployment/huggingface
COPY --chown=user:user README.md ./README.md
COPY --chown=user:user --from=frontend-build /build/frontend/dist ./frontend-dist

RUN mkdir -p /data/results /data/backups /data/logs /data/ground_truths_cache /data/hf_cache /backup \
    && chown -R user:user /data /backup /home/user/app \
    && chmod 700 /data /backup \
    && chmod 755 /home/user/app/deployment/huggingface/start.sh

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT:-7860}/api/health/live || exit 1

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["/home/user/app/deployment/huggingface/start.sh"]
