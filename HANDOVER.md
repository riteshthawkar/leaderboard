# Project Handover — Combined VLM Leaderboard

_Last updated: 2026-06-18_

This document is the single source of truth for picking up this project. It
describes what the system is, how it is laid out, how to run it, what has been
verified, and what is still open.

---

## 1. What this project is

A web leaderboard that benchmarks Vision-Language Models (VLMs) across **three
tasks**, grouped into **two reporting sections**:

| Section | Task id | Source benchmark | Score type |
|---|---|---|---|
| **Visual Cognition** | `do_you_see_me` | Do-You-See-Me (perception) | pure accuracy |
| **Visual Cognition** | `minds_eye` | Mind's-Eye (mental imagery) | pure accuracy |
| **Spatial Reasoning** | `spatial` | 13 public spatial benchmarks (paper arXiv:2604.16060) | accuracy + diagnostics |

- **Visual Cognition Index (VCI)** = equal-weight mean of the two Visual
  Cognition tasks (perception + imagery).
- **Spatial** is scored on accuracy plus three robustness diagnostics:
  - **CoT-Δ** — change in accuracy when chain-of-thought is requested.
  - **Shortcut score** — accuracy with a blank/gray image (lower is better).
  - **Hallucination resistance** — rate of correctly choosing "Cannot
    determine" when the image is blank and that option is appended.

Users **download public sample sets (no ground truth)**, run their model
locally with the provided harnesses, and **upload a single response file** per
task. Ground truth stays server-side. Scoring is deterministic via a single
`TaskScorer`.

---

## 2. Repository layout (what is in this repo)

```
Leaderboard-Project/                  <- repo root
  HANDOVER.md                         <- this file
  .gitignore
  Combined-Leaderboard/               <- THE APPLICATION
    backend/
      web/app.py                      <- Flask app (served by Waitress)
      config.py                       <- TASKS, SECTIONS, SPATIAL_DATASETS (13), paths, weights
      constants.py
      request_models.py               <- Pydantic request models (renamed from validators.py*)
      scoring/
        task_scorer.py                <- single source of truth for scoring + diagnostics
        engine.py
      leaderboard_store.py            <- JSON-backed store keyed by model_name
      leaderboard_manager.py
      data_handlers/                  <- ground-truth + submission loaders
      models/                         <- dataclasses (tasks.py: TaskScore, Diagnostics, ...)
      build_tasks.py                  <- builds tasks/<id>/ bundles (questions, GT, templates)
      build_golden_set.py             <- legacy data loaders (still used by build_tasks)
      file_security.py, database.py, logging_config.py
    frontend/                         <- index.html + static/ (2-tab UI, canvas charts, no JS deps)
    tasks/                            <- GENERATED bundles served to the UI
      do_you_see_me/  minds_eye/  spatial/
    dysm_harness/                     <- user harness for Do-You-See-Me
    minds_eye_harness/                <- user harness for Mind's-Eye
    spatial_harness/                  <- user + maintainer harness for Task 3 (VLMEvalKit-backed)
    requirements.txt
    run.bat / run.sh
    .env.example                      <- copy to .env and set API_TOKENS
```

\* `backend/validators.py` was renamed to `request_models.py` because the name
collided with the PyPI `validators` package that VLMEvalKit imports.

**Excluded from this repo (by `.gitignore`)** — they are separate third-party
repos / runtime data / secrets:

- `Do-You-See-Me/` and `Mind-s-Eye/` — large third-party benchmark repos (~280 MB,
  their own git history and licenses; some datasets are non-redistributable).
- `env/` — local virtual environment.
- `results/`, `uploads/`, `**/logs/`, `**/__pycache__/`, `.env`.

---

## 3. Environment & install

- **Python env:** conda env `leaderboard`
  (`C:\Users\b-rthawkar\mc3\envs\leaderboard\python.exe`, Python 3.12).
  conda is **not on PATH**; use the full python path, or
  `C:\Users\b-rthawkar\mc3\condabin\conda.bat`.
- **App deps:** `pip install -r Combined-Leaderboard/requirements.txt`
  (Flask, Waitress, pandas, numpy 1.26.4, pydantic 2, etc.).

### VLMEvalKit (only needed for the Spatial / Task-3 harness)

VLMEvalKit is **not on PyPI** and has Windows-specific install pitfalls. The
working recipe (already applied on this machine):

1. **polygon3 needs a C++ compiler** — install a prebuilt binary first:
   `conda install -n leaderboard -c conda-forge polygon3 -y`
2. **The wheel build drops subpackages** (e.g. `megabench/parsing`) → use an
   **editable install from a clone** instead of `pip install git+...`:
   ```
   git clone --depth 1 https://github.com/open-compass/VLMEvalKit.git C:\Users\b-rthawkar\VLMEvalKit
   <env-python> -m pip install -e C:\Users\b-rthawkar\VLMEvalKit --no-deps
   ```
   (Runtime deps such as torch/transformers/datasets get installed by the first
   failed wheel attempt, or install them separately.)
3. **Windows import crash** — patch
   `vlmeval/dataset/utils/hipho_verifier.py`: its `timeout()` only returns a
   decorator on POSIX and returns `None` on Windows. An `else: def
   decorator(func): return func` fallback was added.

> The VLMEvalKit clone lives **outside** this repo at `C:\Users\b-rthawkar\VLMEvalKit`.
> A new machine must repeat the steps above. The harness only needs
> `vlmeval.dataset.build_dataset`, `SUPPORTED_DATASETS`, and
> `vlmeval.config.supported_VLM`.

---

## 4. How to run the server

From `Combined-Leaderboard/`:

```powershell
$env:API_TOKENS = 'your-secret-token'   # bearer token required for submissions
& "C:\Users\b-rthawkar\mc3\envs\leaderboard\python.exe" backend\web\app.py
```

- Serves **Waitress** on `0.0.0.0:5000` (no auto-reload — restart to apply edits).
- Set `$env:FLASK_DEBUG=1` to use the Flask dev server with reload instead.
- Check / free port 5000 before restarting.

**Endpoints (selected):**
`GET /api/health`, `/api/sections`, `/api/tasks/<id>/info|questions|template.<fmt>`,
`/api/spatial/manifest`, `POST /api/tasks/<id>/submit` (Bearer auth),
`GET /api/leaderboard/visual-cognition`, `/api/leaderboard/spatial`,
`/api/statistics/overview`, `/api/model/<name>/report`.

**Rebuild task bundles** (do this if `tasks/` is missing or data changed):
```powershell
& "C:\Users\b-rthawkar\mc3\envs\leaderboard\python.exe" backend\build_tasks.py
```
This writes `tasks/{do_you_see_me,minds_eye,spatial}/`. DYSM=175 samples,
Mind's-Eye=175, spatial=manifest + 39 **synthetic placeholder** samples so the
server boots. Real spatial ground truth comes from the spatial harness (below).

---

## 5. The three harnesses

Each benchmark has its own harness folder; a user runs it to produce one
upload-ready response file.

- **`dysm_harness/`, `minds_eye_harness/`** — read `tasks/<id>/questions.json`,
  resolve local image paths, call a pluggable `model_generate(payload)` adapter,
  support `--dry-run`/`--limit`, and emit
  `{task_id, model_meta, predictions:{standard:{sample_id:pred}}}`.

- **`spatial_harness/`** — VLMEvalKit-backed (Task 3 has **no source codebase**;
  the paper is an analysis/meta paper aggregating 13 public benchmarks). Files:
  - `registry.py` — maps our 13 dataset ids → VLMEvalKit registry names, validates
    against the installed registry, skips+warns unknowns.
  - `vlmeval_backend.py` — `build_dataset`, `iter_samples` →
    `{sample_id, index, question, options, answer, images}`; `make_sample_id = f"{ds_id}:{index}"`.
  - `conditions.py` — builds the 4 conditions (standard / cot / no_image /
    no_image_plus) and the gray placeholder image (PIL).
  - `adapters.py` — `load_adapter`: `example_adapter` (offline stub) |
    vlmeval `supported_VLM[name]` | a custom module exposing `model_generate(messages)`.
  - `build_ground_truth.py` — **maintainer** script: downloads datasets via
    VLMEvalKit and writes `tasks/spatial/ground_truth.json` + metadata-only
    `questions.json` + refreshed `manifest.json`.
  - `run_harness.py` — **user** script: `--model --datasets --conditions --limit
    --dry-run --score`; emits `{task_id:"spatial", model_meta, predictions:{4 conditions}}`.

  **Verified registry names** (`SUPPORTED_DATASETS` has 600 entries):
  `blink=BLINK`, `cvbench2d=CV-Bench-2D`, `cvbench3d=CV-Bench-3D`, `mmvp=MMVP`,
  `realworldqa=RealWorldQA`, `vstar=VStarBench`, `3dsrbench=3DSRBench`,
  `mindcube=MindCubeBench_raw_qa`, `mmsibench=MMSIBench_wo_circular`,
  `omnispatial=OmniSpatialBench`, `vsr=VSR-zeroshot`.
  **Not in the registry (skipped):** `spatialbench`, `satreal`.

---

## 6. Verification status

- **Visual Cognition (Tasks 1 & 2):** full end-to-end tested — bundles build
  (175 + 175), harness outputs parse and match all ids via `TaskScorer`,
  leaderboard ranks complete > partial submissions.
- **Spatial (Task 3):** real end-to-end tested (not dry-run) —
  `build_ground_truth.py --datasets mmvp --limit 5` downloaded real MMVP and
  wrote 5 GT entries; `run_harness.py --model example_adapter --datasets mmvp
  --limit 5 --score` → accuracy 0.600 (3/5) with diagnostics computed.
- **Server smoke test:** `/api/health` reports `healthy` (database +
  ground_truth); `/api/leaderboard` returns entries.

---

## 7. Known issues / open items

- **Benign boot noise:** server logs `Ground truth file not found ... 3D_DoYouSeeMe/...`
  for 3D DYSM tasks (3D data not present). Health still reports healthy. Could be
  silenced in `data_handlers/ground_truth.py`.
- **Spatial placeholder vs real GT:** `build_tasks.build_spatial_task` writes
  **synthetic** spatial GT so the server boots; `spatial_harness/build_ground_truth.py`
  is the real one and overrides it. Run the real builder before serving real
  spatial scores.
- **Dataset licensing:** RealWorldQA is CC BY-ND — do **not** host a merged
  Task-3 dataset blob. The manifest + harness model (download-locally,
  upload-responses) is intentional to respect this.
- **Latent 3D path bug:** the DYSM handler builds `3D_DoYouSeeMe/<task>/...`
  while the real folders are prefixed `3D_<task>` — 3D never loads. 2D works.
- **Dead/legacy files left on disk** (unused): `golden_leaderboard.py`,
  `build_golden_set.py` is still imported by `build_tasks.py` (keep it),
  `golden_set/`.
- **Deprecation warnings:** `datetime.utcnow()` and Pydantic v1 `.dict()` in
  `app.py` — cosmetic.

---

## 8. Paper reference (Task 3)

arXiv:2604.16060 — "CoT Degrades Visual Spatial Reasoning" (Microsoft Research +
IIT-H). 17 models × 13 spatial benchmarks, pass@1 greedy (temp 0, 3 seeds),
evaluated via VLMEvalKit with a uniform MCQ prompt; LLM-as-judge =
Qwen3-30B-A3B-Instruct-2507. No-Image = full gray image (shortcut probe);
No-Image++ = gray image + appended "Cannot determine from the image" option
(hallucination-resistance probe).
