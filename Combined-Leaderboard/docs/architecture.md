# Project Architecture

## Source boundaries

| Path | Ownership |
|---|---|
| `backend/` | Flask API, auth, submission validation, scoring, persistence, backups, and health checks |
| `frontend/src/` | React application source and UI assets |
| `frontend/tests/` | Frontend unit and component tests |
| `evaluation/` | Participant-facing benchmark runners and output packaging, separated by track |
| `tasks/` | Public questions, manifests, and submission templates |
| `tests/` | Backend, evaluation, integration, and opt-in live E2E tests |
| `scripts/` | Operator and maintenance commands |
| `deployment/` | API and Hugging Face container configuration |
| `docs/` | Architecture, deployment, and operating documentation |

`Ground_truths/` is an ignored local mount point for private answer keys. It is
not application source and only its README may be committed.

## Runtime boundaries

Production runtime state must be mounted outside the source tree through
`LEADERBOARD_DATA_DIR`. This includes SQLite databases, leaderboard cache
files, logs, backup archives, and downloaded private ground-truth caches.
Uploaded participant files are validated in memory; normalized final answers
and audit metadata are stored in SQLite.

Local ignored databases, logs, backups, model outputs, virtual environments,
dependency directories, and generated frontend bundles are development state.
They must not be copied into a deployment image or committed to Git.

## Deployment entry points

- Root `Dockerfile`: single-origin Hugging Face deployment with Nginx and one API worker.
- `deployment/api/Dockerfile`: standalone API image for split hosting.
- `frontend/Dockerfile`: standalone static frontend image.
- `deployment/huggingface/`: supervisor, Nginx, and startup configuration for the root image.
