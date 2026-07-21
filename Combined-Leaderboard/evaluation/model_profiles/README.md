# Model profiles

Run both public benchmarks, extraction, and submission export with one command:

```bash
python evaluation/run_public_evaluation.py \
  --profile evaluation/model_profiles/qwen35-9b.json \
  --gpus 0,1
```

The ranked-model profiles select audited built-in integrations. To evaluate a new
vLLM-compatible model, copy `custom-model.template.json`, fill in its immutable
repository revision and model settings, then pass that file to the same command.

The command runs all DYS and Mind's Eye inference first, unloads the visual
model, then runs the pinned model-only extractor. The extractor receives only the
saved response and expected answer format and returns an answer or an empty
string; no deterministic answer recovery is used.

Successful runs write ready-to-upload files under
`evaluation/results/public-runs/<slug>/`:

- `do_you_see_me_submission.jsonl`
- `minds_eye_submission.jsonl`

`serving_mode: replica` places one complete model replica on each selected GPU.
`serving_mode: tensor_parallel` shards one model across all selected GPUs.