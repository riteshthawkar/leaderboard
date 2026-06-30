# Combined Leaderboard

React + Flask application for the MSR visual cognition and spatial reasoning leaderboard. It evaluates VLM submissions across Do-You-See-Me, Mind's-Eye, and Spatial Reasoning, then exposes track-aware rankings, metadata filters, comparison views, and model reports.

## Current Architecture

- `frontend/` - React/Vite app with Tailwind/shadcn-style primitives and bklit/visx chart components.
- `backend/web/app.py` - Flask API and React build host.
- `backend/leaderboard_store.py` - JSON-backed per-model task store used by the React leaderboard.
- `backend/scoring/task_scorer.py` - per-task scoring and spatial diagnostics.
- `tasks/` - public question bundles, hidden ground truth, and submission templates.
- `dysm_harness/`, `minds_eye_harness/`, `spatial_harness/` - user-facing response-generation harnesses.

Legacy Jinja templates still exist as a fallback when the React build is absent, but the primary UI is the Vite app.

## Quick Start

```bash
cd Combined-Leaderboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set SECRET_KEY. Optional: OPENAI_API_KEY and OAuth client vars.
python backend/web/app.py
```

The backend defaults to `http://localhost:5050` so the Vite dev proxy works without extra configuration.

In another terminal:

```bash
cd Combined-Leaderboard/frontend
npm install
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`. If that port is busy, Vite will choose the next free port.

For a production frontend build:

```bash
cd Combined-Leaderboard/frontend
npm run build
```

The build writes to `frontend/static/react-app/`, which Flask serves automatically.

## Production Defaults

- Submission auth is enabled by default. Keep `DISABLE_SUBMISSION_AUTH=false` for production.
- Set `DISABLE_SUBMISSION_AUTH=true` only for short local UI testing.
- Normal submissions use registered user bearer tokens from `/login`.
- `API_TOKENS` is optional admin/legacy access, not required for ordinary users.
- Google and Microsoft buttons are visible, but OAuth works only when the corresponding client IDs/secrets and callback URLs are configured.
- Use persistent storage for `users.db` and `results/leaderboard_store.json` in deployment.
- Use Redis for `LIMITER_STORAGE_URI` when running more than one process or container.

## Important Environment Variables

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Required Flask secret for production. |
| `FLASK_PORT` | Backend port; defaults to `5050`. |
| `OPENAI_API_KEY` | Enables GPT-4o extraction/judging; deterministic fallback is used when unset. |
| `DISABLE_SUBMISSION_AUTH` | Keep `false` unless doing temporary local testing. |
| `CORS_ORIGINS` | Comma-separated allowed origins. |
| `LIMITER_STORAGE_URI` | Use `redis://...` for production rate limits. |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Enables Google OAuth. |
| `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET` | Enables Microsoft OAuth. |
| `OAUTH_REDIRECT_BASE_URL` | Public base URL for OAuth callbacks. |

## Submission Flow

1. Register or sign in at `/login`.
2. Download questions/templates from `/submit`.
3. Run the model locally, ideally through the harness for that benchmark.
4. Upload JSON or CSV predictions plus required metadata.
5. The backend scores the file, stores the task result, and updates the leaderboard.

Required metadata includes organization, source status, parameter count, base model, training/fine-tuning summary, CoT usage, method description, prompt template, and change log.

## Spatial Ground Truth

`backend/build_tasks.py` creates a small synthetic spatial bundle so the app can boot. Do not use that for official spatial results. For real spatial scoring, run `spatial_harness/build_ground_truth.py` against the official dataset sources and replace `tasks/spatial/ground_truth.json` before launch.

## Useful Checks

```bash
cd Combined-Leaderboard/frontend
npm run build

cd ..
python3 -m py_compile backend/web/app.py backend/leaderboard_store.py backend/auth_db.py
curl -fsS http://localhost:5050/api/health
```

Known non-blocker: the frontend build currently emits a Vite warning that the main JS chunk is larger than 500 kB. Code splitting can be added later if startup weight becomes a problem.
