---
title: MS VISTA Leaderboard
sdk: docker
app_port: 7860
fullWidth: true
header: mini
short_description: Visual perception, cognition, and spatial reasoning model evaluation
startup_duration_timeout: 30m
---

# Combined Leaderboard

React + Flask application for the MSR visual cognition and spatial reasoning leaderboard. It evaluates VLM submissions across Do-You-See-Me, Mind's-Eye, and Spatial Reasoning, then exposes track-aware rankings, metadata filters, comparison views, and model reports.

## Current Architecture

- `backend/` - API-only Flask service for auth, scoring, submissions, persistence, backups, and leaderboard data.
- `frontend/src/` - independently hosted React/Vite application source with Tailwind and Visx components.
- `frontend/tests/` - frontend unit and component tests, kept outside production source.
- `backend/leaderboard_store.py` - JSON-backed per-model task store used by the React leaderboard.
- `backend/scoring/task_scorer.py` - per-task scoring and spatial diagnostics.
- `tasks/` - public question bundles, manifests, and JSONL submission templates.
- `Ground_truths/` - ignored local mount point for private answer keys; only its README is committed.
- `evaluation/` - isolated, benchmark-specific evaluation packages for Do You See Me, Mind's Eye, and Spatial Reasoning.
- `tests/` - backend, evaluation, integration, and opt-in live E2E verification.
- `deployment/` - standalone API and Hugging Face deployment configuration.
- `docs/` - architecture, deployment, and operating documentation.

Flask does not render pages or expose frontend static files. The frontend calls the public API origin configured through `VITE_API_BASE_URL`.

## Quick Start

```bash
cd Combined-Leaderboard
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Edit .env and set SECRET_KEY. Optional: OAuth client vars.
python backend/web/app.py
```

The API defaults to `http://localhost:5050`. Its root and `/api` return service metadata; application data is exposed below `/api/*`.

In another terminal:

```bash
cd Combined-Leaderboard/frontend
npm install
cp .env.example .env.development.local
npm run dev
```

Open `http://localhost:5173`. The example frontend environment sends requests directly to `http://localhost:5050`.

For a production frontend build, provide the public API origin:

```bash
cd Combined-Leaderboard/frontend
cp .env.example .env.production
# Set VITE_API_BASE_URL=https://api.example.com in .env.production
npm run build
```

The build writes to `frontend/dist/`. Deploy that directory to the frontend host; Flask never serves it.

## Local SQLite Backend

Use SQLite for local development and the intended low-concurrency,
single-instance production deployment.

```env
LEADERBOARD_DATA_DIR=.
DATABASE_URL=sqlite:///leaderboard.db
AUTH_DATABASE_URL=
SUBMISSION_DATABASE_URL=
LIMITER_STORAGE_URI=memory://
DISABLE_SUBMISSION_AUTH=false
AUTH_DEV_MODE=true
SESSION_COOKIE_SECURE=false
SESSION_COOKIE_SAMESITE=Lax
FRONTEND_BASE_URL=http://localhost:5173
API_BASE_URL=http://localhost:5050
OAUTH_REDIRECT_BASE_URL=http://localhost:5050
CORS_ORIGINS=http://localhost:5173,http://localhost:5174
GROUND_TRUTHS_DIR=./Ground_truths
GROUND_TRUTHS_SOURCE=auto
MAX_CONTENT_LENGTH=52428800
MAX_FILE_SIZE_PER_SUBMISSION=52428800
```

If you want the scorer to force Hugging Face ground-truth loading instead of
local files, set `GROUND_TRUTHS_SOURCE=hf`, `GROUND_TRUTHS_HF_REPO`, and
`HF_TOKEN`. Otherwise `auto` uses local `Ground_truths/` first and only falls
back to Hugging Face when configured.

## Production Defaults

- Submission auth is enabled by default. Keep `DISABLE_SUBMISSION_AUTH=false` for production.
- Set `DISABLE_SUBMISSION_AUTH=true` only for short local UI testing.
- Normal submissions use the signed Flask session cookie created by `/login`.
- New passwords must contain 15 to 128 characters; existing shorter passwords remain valid until they are reset.
- Password resets are single-use and revoke every previously issued session for the account.
- Google and Microsoft buttons are visible, but OAuth works only when the corresponding client IDs/secrets and callback URLs are configured.
- Email/password registration and password reset require either Azure Communication Services Email or SMTP. Without a provider, production delivery fails instead of silently logging account links.
- Use persistent storage for `LEADERBOARD_DATA_DIR`, or set `RESULTS_DIR` and `LEADERBOARD_STORE_FILE` explicitly.
- The low-concurrency production deployment uses SQLite with one web worker, WAL mode, a five-second busy timeout, and persistent storage.
- Verified SQLite backups run every 48 hours by default. The newest 15 archives are retained under `LEADERBOARD_DATA_DIR/backups`; admins can inspect or run backups from `/admin`.

The root [`Dockerfile`](Dockerfile) builds the same-origin Hugging Face Space
with Nginx in front of the API. [`deployment/api/Dockerfile`](deployment/api/Dockerfile) and
[`frontend/Dockerfile`](frontend/Dockerfile) remain available for split
hosting. See the [deployment guide](docs/deployment.md) for storage, secrets, and
temporary no-login test configuration.

## Important Environment Variables

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Required Flask secret for production. |
| `DEPLOYMENT_MODE` | `local` for localhost or `public` to fail closed on production HTTPS, privacy, admin, and backup requirements. |
| `FLASK_PORT` | Backend port; defaults to `5050`. |
| `DISABLE_SUBMISSION_AUTH` | Keep `false` unless doing temporary local testing. |
| `ADMIN_EMAILS` | Comma-separated verified account emails allowed to use `/admin` and admin APIs. |
| `LEADERBOARD_DATA_DIR` | Persistent runtime data directory for SQLite and scored results. |
| `GROUND_TRUTHS_DIR` | Private answer-key directory used by the scorer; do not expose publicly. |
| `GROUND_TRUTHS_SOURCE` | `auto`, `local`, or `hf`; `auto` uses local GT and falls back to HF. |
| `GROUND_TRUTHS_HF_REPO` | Private Hugging Face repo containing `<subset>/ground_truth.jsonl`. |
| `HF_TOKEN` | Hugging Face token with read access to the private GT repo. |
| `DATABASE_URL` | SQLite database URL for auth, quota, submission audit, and rescore state. |
| `SUBMISSION_DAILY_LIMIT_PER_BENCHMARK` | Successful submissions allowed per account and benchmark in each rolling 24-hour window; defaults to 1. |
| `SQLITE_BUSY_TIMEOUT_MS` | Time SQLite waits for a writer before returning a lock error; defaults to 5000 ms. |
| `AUTO_BACKUP_ENABLED` | Enables retained, verified server-side SQLite backups; defaults to true in production. |
| `AUTO_BACKUP_INTERVAL_HOURS` | Scheduled backup interval; defaults to 48 hours. |
| `AUTO_BACKUP_RETENTION_COUNT` | Number of scheduled backup archives retained; defaults to 15. |
| `AUTO_BACKUP_MIRROR_DIR` | Second mounted filesystem receiving a verified copy of every retained archive. |
| `REQUIRE_OFFSITE_BACKUP` | Requires a current mirror on a different filesystem; always enabled in public mode. |
| `CORS_ORIGINS` | Comma-separated allowed origins. |
| `LIMITER_STORAGE_URI` | Use `redis://...` for production rate limits. |
| `TRUST_PROXY_HOPS` | Exact trusted reverse-proxy count used for client IP and HTTPS detection; keep `0` for direct traffic. |
| `FRONTEND_BASE_URL` | Required public React origin for reset links and post-auth redirects. |
| `API_BASE_URL` | Required public Flask API origin for verification links. |
| `PRIVACY_POLICY_URL` | Required HTTPS privacy-policy URL in public mode. |
| `ACS_*` or `SMTP_*` | Transactional email provider for account verification and password reset. |
| `SESSION_COOKIE_SECURE` | Set `true` when served over HTTPS. |
| `SESSION_COOKIE_SAMESITE` | Use `Lax` for same-site subdomains; use `None` with HTTPS when frontend and API are on different sites. |
| `MAX_FILE_SIZE_PER_SUBMISSION` | Per-upload JSONL size limit in bytes. |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Enables Google OAuth. |
| `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET` | Enables Microsoft OAuth. |
| `OAUTH_REDIRECT_BASE_URL` | Public base URL for OAuth callbacks. |

## Frontend / Backend Connection

Create `frontend/.env.production` from `frontend/.env.example` and set the API service origin:

```env
VITE_API_BASE_URL=https://your-api-domain.example
```

Then set backend production env accordingly:

```env
CORS_ORIGINS=https://your-frontend-domain.example
SESSION_COOKIE_SECURE=true
FRONTEND_BASE_URL=https://your-frontend-domain.example
API_BASE_URL=https://your-api-domain.example
OAUTH_REDIRECT_BASE_URL=https://your-api-domain.example
```

Use `SESSION_COOKIE_SAMESITE=Lax` when frontend and API use same-site HTTPS subdomains such as `app.example.com` and `api.example.com`. Use `SESSION_COOKIE_SAMESITE=None` with `SESSION_COOKIE_SECURE=true` only when the two origins are on different sites. Exact frontend origins must be listed in `CORS_ORIGINS`; wildcard origins are rejected because authentication uses credentialed cookies.

OAuth provider callback URLs must point at the backend, for example `https://your-api-domain.example/api/auth/oauth/google/callback`.

## Submission Flow

1. Register or sign in on the frontend `/login` route.
2. Register the model once, then select its stable model identity in the submission workspace.
3. Download questions/templates from the frontend `/submit` route.
4. Run the model locally, ideally through the harness for that benchmark.
5. Upload one benchmark's JSONL or spatial ZIP plus run-specific metadata.
6. For Do You See Me and Mind's Eye, the backend validates sample coverage and scores final answers against private ground truth. For Spatial, it validates official harness provenance, public per-sample evidence, and aggregate arithmetic without independently grading the answers. Every accepted result is linked to the selected model and published to the appropriate leaderboard.

Visual submission JSONL rows must include `question_id` and `answer`; `sample_id` is accepted for legacy local bundles. `condition` defaults to `standard`. Spatial users upload only the versioned ZIP emitted by the official harness.

Combined generator exports that contain the `subset` and `output` fields can be
split into canonical task files with the checked scorer before upload:

```bash
python scripts/normalize_model_outputs.py \
  ../FInal_model_output_files/results_q35__main_noncot.jsonl \
  --model-name "Qwen3.5"
```

The command validates JSONL syntax, duplicate IDs, exact private sample
coverage, scalar answers, and both task scores. Empty model outputs fail by
default. For a reviewed administrative import, `--empty-policy incorrect`
replaces each blank with an explicit no-response token so it remains present
and scores as incorrect. The source file is never modified, and a SHA-256
normalization report is written beside the canonical files.

The Do-You-See-Me and Mind's-Eye public `questions.jsonl` bundles mirror the public Hugging Face dataset `amolharsh/visual-intelligence-leaderboard`; private answers are loaded separately from `GROUND_TRUTHS_DIR`.

## Visual Evaluation

On a Linux machine with a free 40 GB-class BF16 NVIDIA GPU, choose a shipped model profile and run both benchmarks with one command:

```bash
python evaluation/run_public_evaluation.py \
  --profile evaluation/model_profiles/qwen35-9b.json \
  --gpus 0,1
```

The command provisions the pinned environment and dataset, runs DYS and Mind's Eye inference with resume checkpoints, unloads the visual model, runs model-only answer extraction, and writes both upload-ready submission files. The extractor receives only each saved model response and its expected answer format; it returns the formatted answer or an empty string. There is no deterministic answer recovery. Copy `evaluation/model_profiles/custom-model.template.json` to add a new vLLM-compatible checkpoint. See [`evaluation/README.md`](evaluation/README.md) for profiles, protocols, outputs, and advanced controls.

## Score Semantics

`accuracy` and `micro_accuracy` are the sample-weighted result and are retained
for auditing. Public ranking uses `macro_accuracy` with a benchmark-specific
aggregation contract:

| Benchmark | Public ranking score |
| --- | --- |
| Do You See Me | Mean task accuracy within 2D and 3D, then an equal mean across the two dimensions |
| Mind's Eye | Unweighted mean of the eight task accuracies |
| Spatial Reasoning | Unweighted mean of main non-CoT accuracy across datasets |

`task_spread` is the descriptive standard deviation across heterogeneous task
or dataset scores. It is not repeated-run uncertainty and must not be displayed
with a plus/minus symbol. `accuracy_std` remains as a backward-compatible alias.

Paper statistics and release-suite statistics are intentionally distinct. The
published Do You See Me evaluation contains 2,612 questions, while the current
leaderboard release contains 4,500 validated questions. The Mind's Eye paper
describes 800 items; the current validated release contains 799 because Visual
Conceptual Slippage contains 99 rows. `/api/tasks/<task_id>/info` reports both
`paper_total_samples` and the actual `total_samples` used for scoring.

Private ground truth can also be loaded from a private Hugging Face repo. Set `GROUND_TRUTHS_SOURCE=hf` or leave `auto` to fall back when local files are absent, then set `GROUND_TRUTHS_HF_REPO`, `GROUND_TRUTHS_HF_REVISION` if needed, and `HF_TOKEN`. Downloaded files are cached under `GROUND_TRUTHS_HF_CACHE_DIR` or `<LEADERBOARD_DATA_DIR>/ground_truths_cache`.

Raw upload files are not stored. For audit and re-scoring, the database stores one structured row per final answer (`submission_id`, `row_index`, `question_id`, `condition`, `answer`, answer hash). Authenticated users can regenerate their submitted final-answer JSONL from `/api/submissions/<submission_id>/export.jsonl`.

Canonical model metadata includes organization, source status, paper link, and optional parameter count. It is registered once. Each benchmark submission separately records CoT usage, method description, prompt template, and change log. Accounts receive an independent rolling quota for each benchmark, so all three tracks can be submitted for the same model without competing for one shared allowance.

Signed-in users can review scored uploads at `/submissions`. Admin users listed
in `ADMIN_EMAILS` can review all submissions, hide/restore entries, soft-delete
test rows, rescore individual submissions, and rebuild public rankings from the
stored per-sample answers at `/admin`.

The server creates a validated SQLite backup at startup and every 48 hours,
retaining the newest 15 archives by default. Each archive is CRC checked and
each SQLite snapshot must pass `PRAGMA quick_check` before it is retained.
Admins can inspect backup status, create an immediate retained backup, or
download an on-demand ZIP from `/admin`. Archives contain online SQLite
snapshots and leaderboard JSON cache files. They intentionally exclude `.env`
secrets and private ground-truth files.

Public mode also requires `AUTO_BACKUP_MIRROR_DIR` on a different mounted
filesystem. Verify or unpack a retained archive with `python -m
backend.backup_cli verify <archive>` and `python -m backend.backup_cli restore
<archive> --destination <empty-directory>`.

## Spatial Public Evidence Contract

`backend/build_tasks.py` creates a small synthetic spatial bundle so the app can boot. The API always rejects submissions against this demo bundle. Build a versioned official bundle on a trusted administrator machine after preparing all 13 normalized datasets:

```bash
cd evaluation/spatial_reasoning
python3 prepare_data.py --lmudata ./LMUData --hf-token "$HF_TOKEN"
python3 build_server_bundle.py \
  --lmudata ./LMUData \
  --benchmark-version 2026-07-12 \
  --ground-truth-output ../../Ground_truths/spatial_v1/ground_truth.json
```

This writes the public manifest, identifier-only questions, six-condition template, and a private administrator QA key under the ignored `Ground_truths/` directory. The backend does not use that private key to grade Spatial uploads. Restart the API and confirm `/api/health` reports `spatial_bundle: healthy` before opening submissions.

The spatial harness produces one user-facing upload, `spatial_reasoning_submission.zip`, containing `submission.jsonl`, `run_manifest.json`, and `leaderboard.json`. The API validates the package in memory, confirms the per-sample correctness evidence agrees with the aggregate report, and stores the exact ZIP and all three members in SQLite. It also retains each official public benchmark contract once, links every Spatial submission to its manifest hash, and uses that immutable version for future rescoring. Visible spatial results expose public evidence metadata, hashes, per-sample results, and the original package under `/api/public/submissions/<submission_id>/`.

The checked-in spatial bundle is intentionally small. Treat spatial launch as blocked until the official bundle has been generated from the pinned TSV files and all dataset preparation checks pass.

`REQUIRE_OFFICIAL_SPATIAL` defaults to false while this track is pending, so the visual leaderboard can pass readiness independently. Spatial uploads remain closed against the demo bundle. After mounting the official public manifest, question identifiers, and template, set the flag to true and confirm the spatial component is healthy before announcing the track.

## Production Build

```bash
cd Combined-Leaderboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
gunicorn --workers 1 --bind 0.0.0.0:5050 --timeout 180 backend.web:app
```

Build the frontend independently:

```bash
cd Combined-Leaderboard/frontend
npm ci
VITE_API_BASE_URL=https://api.example.com npm run build
# Deploy frontend/dist to the frontend host.
```

Required production settings:

- `SECRET_KEY` set to a generated random value.
- `DEPLOYMENT_MODE=public`.
- `SESSION_COOKIE_SECURE=true` behind HTTPS.
- `SESSION_COOKIE_SAMESITE=Lax` for same-site frontend/API subdomains, or `None` for different sites.
- `CORS_ORIGINS=https://your-frontend-domain.example`.
- `FRONTEND_BASE_URL` set to the public frontend HTTPS origin.
- `API_BASE_URL` and `OAUTH_REDIRECT_BASE_URL` set to the public backend HTTPS origin.
- `PRIVACY_POLICY_URL` and `VITE_PRIVACY_POLICY_URL` set to the approved public policy.
- `ACS_CONNECTION_STRING`/`ACS_ENDPOINT` plus `ACS_SENDER_ADDRESS`, or `SMTP_HOST` credentials.
- `DATABASE_URL=sqlite:////persistent/path/leaderboard.db`.
- `WEB_CONCURRENCY=1` when using SQLite. Keep a small `GUNICORN_THREADS` value
  such as `4` so liveness and read requests remain responsive during scoring.
- `LIMITER_STORAGE_URI=memory://` for local/private testing, or Redis later if the deployment becomes multi-instance.
- `TRUST_PROXY_HOPS` set to the exact reverse-proxy count (`0` when the API is reached directly).
- Persistent `LEADERBOARD_DATA_DIR` or explicit persistent `RESULTS_DIR` and `LEADERBOARD_STORE_FILE`.
- `AUTO_BACKUP_MIRROR_DIR` mounted on a different filesystem from `LEADERBOARD_DATA_DIR`.
- `REQUIRE_OFFICIAL_SPATIAL=true` after replacing the demo spatial bundle with official data.

## Useful Checks

```bash
cd Combined-Leaderboard/frontend
npm audit --omit=dev
VITE_API_BASE_URL=https://api.example.com npm run build

cd ..
python3 -m pip-audit -r requirements.txt
python3 -m pytest -q
python3 -m py_compile backend/web/app.py backend/leaderboard_store.py backend/auth_db.py backend/submission_store.py
curl -fsS http://localhost:5050/api/health
```
