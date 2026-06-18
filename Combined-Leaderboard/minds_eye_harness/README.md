# Mind's-Eye Harness (Mental Imagery)

Generates the **Mind's-Eye** submission file for the Visual Cognition track.
Reads the public sample set, builds a prompt per sample, queries your model, and
writes one response file to upload.

Mind's-Eye is multiple-choice, but the options are rendered **inside each image**
(labelled A/B/C/D, etc.), so the model answers with the option label. The harness
produces only the `standard` condition.

## Usage

```bash
# 1. Validate the pipeline end-to-end with a stub model (no inference):
python run_harness.py --dry-run --out minds_eye_responses.json

# 2. Implement an adapter, then run for real:
python run_harness.py --adapter my_model --out minds_eye_responses.json
```

You implement exactly one thing — a **model adapter**: a module exposing
`model_generate(payload) -> str` (see `example_adapter` in `run_harness.py`).
The payload is `{sample_id, image, text, task, capability}`; `image` is an
absolute path to the question image (with the options drawn in).

## Output

```json
{
  "task_id": "minds_eye",
  "model_meta": {"name": "my-model"},
  "predictions": {
    "standard": { "minds_eye:mental_rotation:0000": "B", "...": "..." }
  }
}
```

Upload it on the leaderboard's **Mind's-Eye** submit card. Use the **same model
name** across all three tasks so the leaderboard merges them into your VCI.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--questions` | `../tasks/minds_eye/questions.json` | public sample set |
| `--adapter` | `example_adapter` | your `model_generate` module |
| `--out` | `minds_eye_responses.json` | output file |
| `--image-root` | workspace root | folder containing `Mind-s-Eye/` |
| `--limit` | `0` | only process the first N samples |
| `--dry-run` | — | stub model, no inference |
