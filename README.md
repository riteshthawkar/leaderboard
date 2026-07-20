# MSR Vision Leaderboard

A full-stack research leaderboard for evaluating Vision-Language Models (VLMs) across three benchmarks from Microsoft Research. Models are scored against paper-faithful ground truth and ranked in real time.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Flask 3.0](https://img.shields.io/badge/flask-3.0-green)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Benchmarks

| Track | Benchmark | Tasks | Metric |
|---|---|---|---|
| **Visual Cognition** | Do-You-See-Me | 7 visual perception dimensions | Accuracy → VCI score |
| **Visual Cognition** | Mind's-Eye | 8 mental imagery tasks | Accuracy → VCI score |
| **Spatial Reasoning** | Spatial CoT | 13 datasets, 4 evaluation conditions | Accuracy + robustness diagnostics |

The **Visual Cognition Index (VCI)** is a weighted composite of Do-You-See-Me (perception) and Mind's-Eye (imagery) scores.

---

## Features

- **User accounts** — register / login to submit; rate-limited per account (3/hour · 10/day)
- **Paper-faithful scoring** — exact same evaluation logic as the source papers; GPT-4o judge with deterministic fallback
- **Live leaderboard** — two tracks (Visual Cognition + Spatial), updated on every submission
- **Robustness diagnostics** — Spatial track reports CoT degradation, shortcut sensitivity, hallucination rate alongside accuracy
- **Model harnesses** — ready-to-run harnesses for Do-You-See-Me, Mind's-Eye, and Spatial (VLMEvalKit backend)
- **Security** — constant-time login, bcrypt password hashing, per-user token rate limiting, atomic leaderboard writes, security headers on all responses

---

## Quick Start

```bash
cd Combined-Leaderboard
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env: set SECRET_KEY and optionally OPENAI_API_KEY

python backend/web/app.py   # -> http://localhost:5050

# In another terminal, run the React app during development
cd frontend
npm install
npm run dev                 # -> http://localhost:5173, or the next free port
```

> **Windows:** use `run.bat`  
> **Unix/macOS:** use `run.sh`

### Environment variables (`.env`)

| Variable | Required | Description |
|---|---|---|
| `SECRET_KEY` | Yes | Flask secret — generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `FLASK_PORT` | Local/dev | Backend port. Defaults to `5050` to match the Vite proxy. |
| `OPENAI_API_KEY` | No | Enables GPT-4o judge; falls back to deterministic scoring if unset |
| `CORS_ORIGINS` | Production | Comma-separated allowed origins, e.g. `https://yoursite.com` |
| `LIMITER_STORAGE_URI` | Production | `redis://...` for persistent rate limits across restarts (default: `memory://`) |
| `API_TOKENS` | Optional | Comma-separated admin tokens for bypass access |
| `DISABLE_SUBMISSION_AUTH` | Local only | Keep `false` for production. Set `true` only for temporary unauthenticated UI testing. |
| `GOOGLE_CLIENT_ID` / `MICROSOFT_CLIENT_ID` | Optional | Enables OAuth buttons when paired with provider secrets and callback URLs. |

---

## Project Structure

```
leaderboard/
└── Combined-Leaderboard/          # Main application
    ├── backend/
    │   ├── web/app.py             # Flask app, all routes
    │   ├── auth_db.py             # User registration / login (SQLite)
    │   ├── leaderboard_store.py   # Atomic JSON leaderboard store
    │   ├── scoring/
    │   │   ├── task_scorer.py     # Per-task scoring engine
    │   │   └── llm_grader.py      # GPT-4o / deterministic judge
    │   ├── data_handlers/
    │   │   └── ground_truth.py    # Ground truth loader
    │   └── constants.py           # Rate limits, file constraints
    ├── frontend/
    │   ├── src/                   # React/Vite app: pages, components, data, API helpers
    │   ├── static/css/site.css    # Shared design system used by React
    │   ├── static/react-app/      # Production build output from npm run build
    │   └── templates/             # Legacy Jinja fallback when React build is absent
    ├── tasks/                     # Ground truth + question sets per benchmark
    │   ├── do_you_see_me/
    │   ├── minds_eye/
    │   └── spatial/
    ├── dysm_harness/              # Do-You-See-Me evaluation harness
    ├── minds_eye_harness/         # Mind's-Eye evaluation harness
    ├── spatial_harness/           # Spatial reasoning harness (VLMEvalKit)
    ├── requirements.txt
    ├── .env.example
    ├── run.sh / run.bat
    └── test_system.py             # End-to-end tests
```

---

## Submitting a Model

1. **Register** at `/login` — you get an API token (stored in browser, used automatically)
2. **Download** the question set for a benchmark from the Submit page
3. **Run your model** — use the harness scripts in `dysm_harness/`, `minds_eye_harness/`, or `spatial_harness/` for standardised output
4. **Upload** the response file (JSON or CSV) — results appear on the leaderboard immediately

### Submission format

```json
[
  {"question_id": "q001", "answer": "A"},
  {"question_id": "q002", "answer": "C"}
]
```

CSV format is also accepted (`question_id,answer` columns).

### Rate limits

| Limit | Value |
|---|---|
| Submissions per hour (per account) | 3 |
| Submissions per day (per account) | 10 |
| Register attempts per hour (per IP) | 5 |
| Login attempts per hour (per IP) | 10 |

---

## Running the Harnesses

### Do-You-See-Me

```bash
cd Combined-Leaderboard/dysm_harness
python run_harness.py --model gpt-4o --output ../submissions/my_model_dysm.json
```

### Mind's-Eye

```bash
cd Combined-Leaderboard/minds_eye_harness
python run_harness.py --model gpt-4o --output ../submissions/my_model_me.json
```

### Spatial (VLMEvalKit backend)

```bash
cd Combined-Leaderboard/spatial_harness
python run_harness.py --model Qwen2.5-VL-7B --conditions standard cot no_image
```

See each harness's `README.md` for full options.

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/auth/register` | — | Register; returns `{username, api_token}` |
| `POST` | `/api/auth/login` | — | Login; returns `{username, api_token}` |
| `POST` | `/api/tasks/{task_id}/submit` | Bearer token | Submit predictions |
| `GET` | `/api/leaderboard/visual-cognition` | — | VCI rankings |
| `GET` | `/api/leaderboard/spatial` | — | Spatial rankings |
| `GET` | `/api/tasks` | — | List available tasks |
| `GET` | `/api/tasks/{task_id}/questions` | — | Download question set |
| `GET` | `/api/tasks/{task_id}/template.json` | — | Download submission template |
| `GET` | `/api/statistics/overview` | — | Aggregate stats |
| `GET` | `/api/health` | — | Health check |

---

## Deployment

See `.env.example` for all configuration options.

**Recommended (Railway / Render):**
1. Connect your GitHub repo
2. Set env vars in the platform dashboard (`SECRET_KEY`, `CORS_ORIGINS`, `LIMITER_STORAGE_URI=redis://...`)
3. Mount a persistent volume so `users.db` and `results/leaderboard_store.json` survive redeploys
4. HTTPS is provided automatically

**Self-hosted (nginx + Waitress):**
```nginx
location / {
    proxy_pass http://127.0.0.1:5050;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $remote_addr;
}
```
The app runs Waitress by default — no extra config needed.

---

## Papers

- **Do You See Me?** — arXiv:2506.02022  
- **Mind's Eye** — arXiv:2604.16054  
- **CoT Degrades Spatial Reasoning** — arXiv:2604.16060





The standard visual evaluation runner and final-results workflow are documented in [Combined-Leaderboard/evaluation/README.md](Combined-Leaderboard/evaluation/README.md). Run models into staging roots with `evaluation/run_visual_suite.sh`, then validate and consolidate the newest complete DYS and Mind's Eye tracks:

```bash
cd Combined-Leaderboard
python -m evaluation.finalize_visual_results --dry-run
python -m evaluation.finalize_visual_results
```

The authoritative local inventory is `evaluation/results/final/index.json`. Source runs are pruned only after canonical validation and never while an `.active-run.json` PID is alive.