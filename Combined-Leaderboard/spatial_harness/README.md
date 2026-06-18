# Spatial Reasoning Harness (Task 3) — VLMEvalKit-backed

Task 3 of the leaderboard reproduces an **analysis paper** (arXiv:2604.16060)
that aggregates **13 public spatial-reasoning benchmarks** and probes models with
four diagnostic conditions. The paper ships **no evaluation code**, so this
harness reconstructs the framework end to end using
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) as the data + inference
backbone and layering the paper's conditions on top.

No dataset is redistributed: VLMEvalKit downloads each benchmark from its
official source on *your* machine.

## Install

```bash
pip install vlmeval        # data loaders, downloaders, 200+ model wrappers
```

VLMEvalKit pulls in `torch`, `transformers`, `datasets`, `Pillow`, etc. A GPU is
recommended for open-weight models; API models (GPT-4o, Gemini, …) need only the
relevant API key in a `.env` file (see VLMEvalKit's Quickstart).

## How it works

```
VLMEvalKit.build_dataset(name)   ──►  downloads + loads each benchmark
        │                              (.data DataFrame: question, A/B/C/D, answer, image)
        ▼
conditions.build_messages()      ──►  standard / cot / no_image / no_image_plus
        │
        ▼
your model (.generate / adapter) ──►  raw answers
        │
        ▼
spatial_responses.json           ──►  upload to the leaderboard
```

| File | Role |
|------|------|
| `registry.py` | maps our 13 ids → VLMEvalKit registry names; validates against the **installed** registry and skips+warns unknowns |
| `vlmeval_backend.py` | builds a dataset, normalises rows, extracts options + ground truth, decodes images |
| `conditions.py` | builds the 4 condition prompts + gray placeholder images |
| `adapters.py` | resolves `--model` to `example` / a VLMEvalKit model / a custom module |
| `build_ground_truth.py` | **maintainer-only**: extracts private ground truth + public metadata |
| `run_harness.py` | **user-facing**: runs your model and writes the submission file |

## Evaluation conditions

| Condition       | Image  | Prompt | Diagnostic |
|-----------------|--------|--------|------------|
| `standard`      | real   | question + options + "answer with the option letter" | accuracy |
| `cot`           | real   | + "think step by step" | CoT‑Δ (does reasoning help/hurt?) |
| `no_image`      | gray   | same text, image blanked | shortcut score (lower = less blind guessing) |
| `no_image_plus` | gray   | + "Cannot determine from the image" as the **correct** option | hallucination resistance (higher = better) |

Only `standard` is required; the other three unlock the diagnostics column on the
Spatial leaderboard.

## Datasets

The 13 benchmarks are declared in `backend/config.py` (`SPATIAL_DATASETS`) and
mapped to VLMEvalKit names in `registry.py`. VLMEvalKit ships most of them
(BLINK, MMVP, RealWorldQA, CV-Bench 2D/3D, 3DSRBench, MindCube, MMSI-Bench,
OmniSpatial, V*Bench). A few (SpatialBench, VSR, SAT) are **not** in the current
registry — the harness reports and skips those rather than guessing. Update the
candidate lists in `registry.py` when VLMEvalKit adds them.

## Usage

### Maintainer — build ground truth once

```bash
# Downloads the resolvable datasets and writes the private ground truth +
# public metadata the server scores against.
python build_ground_truth.py
# subset / smoke test:
python build_ground_truth.py --datasets blink mmvp --limit 50
```

Outputs (paths from `backend/config.py`):
- `tasks/spatial/ground_truth.json` — private; `{sample_id: {answer, dataset, group, tags}}`
- `tasks/spatial/questions.json` — public **metadata only** (no questions/images)
- `tasks/spatial/manifest.json` — refreshed real sample counts

### User — evaluate a model

```bash
# A VLMEvalKit-supported model (name from vlmeval.config.supported_VLM):
python run_harness.py --model Qwen2-VL-7B-Instruct --out spatial_responses.json

# Your own model: a module exposing model_generate(messages) -> str
python run_harness.py --model my_model

# Validate the whole pipeline offline (no vlmeval, no downloads, stub model):
python run_harness.py --dry-run --score
```

`messages` is VLMEvalKit's interleaved format:
`[{"type": "image", "value": "/path.png"}, {"type": "text", "value": "..."}]`.

## Output

```json
{
  "task_id": "spatial",
  "model_meta": {"name": "Qwen2-VL-7B-Instruct"},
  "predictions": {
    "standard":      {"blink:0": "B", "...": "..."},
    "cot":           {"...": "..."},
    "no_image":      {"...": "..."},
    "no_image_plus": {"...": "..."}
  }
}
```

Upload it on the leaderboard's **Spatial Reasoning** submit card. Sample ids are
produced by the same backend the maintainer used for the ground truth, so they
always align. Ground truth stays private server-side.
