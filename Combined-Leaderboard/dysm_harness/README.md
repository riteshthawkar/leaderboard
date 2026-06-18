# Do-You-See-Me Harness (Perception)

Generates the **Do-You-See-Me** submission file for the Visual Cognition track.
Reads the public sample set, builds a prompt per sample, queries your model, and
writes one response file to upload.

Do-You-See-Me is free-form perception (no MCQ options), so the harness produces
only the `standard` condition.

## Usage

```bash
# 1. Validate the pipeline end-to-end with a stub model (no inference):
python run_harness.py --dry-run --out dysm_responses.json

# 2. Implement an adapter, then run for real:
python run_harness.py --adapter my_model --out dysm_responses.json
```

You implement exactly one thing — a **model adapter**: a module exposing
`model_generate(payload) -> str` (see `example_adapter` in `run_harness.py`).
The payload is `{sample_id, image, text, task, capability}`; `image` is an
absolute path to the question image.

## Output

```json
{
  "task_id": "do_you_see_me",
  "model_meta": {"name": "my-model"},
  "predictions": {
    "standard": { "do_you_see_me:visual_spatial:0000": "2", "...": "..." }
  }
}
```

Upload it on the leaderboard's **Do-You-See-Me** submit card. Use the **same
model name** across all three tasks so the leaderboard merges them into your VCI.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--questions` | `../tasks/do_you_see_me/questions.json` | public sample set |
| `--adapter` | `example_adapter` | your `model_generate` module |
| `--out` | `dysm_responses.json` | output file |
| `--image-root` | workspace root | folder containing `Do-You-See-Me/` |
| `--limit` | `0` | only process the first N samples |
| `--dry-run` | — | stub model, no inference |
