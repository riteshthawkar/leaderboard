# Mind's Eye evaluation

This package runs the visual cognition benchmark and creates one upload-ready JSONL file. Run commands from the repository root.

## OpenAI-compatible or vLLM endpoint

```bash
pip install -r evaluation/requirements-vllm.txt
python -m evaluation.minds_eye.run_vllm \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --endpoints http://localhost:8000/v1
```

Use `--image-root /path/to/dataset` for local images. Without it, the runner uses the public `image_url` in the question bundle.

## Hugging Face transformers

```bash
pip install -r evaluation/requirements-hf.txt
python -m evaluation.minds_eye.run_hf \
  --model Qwen/Qwen3-VL-8B-Instruct
```

For multi-GPU sharding, run one process per shard with `--shard` and `--nshards`, then merge every generated diagnostics file:

```bash
python -m evaluation.minds_eye.merge_shards \
  evaluation/minds_eye/results/*_shard_*_of_*.diagnostics.jsonl
```

The final file is `evaluation/minds_eye/results/minds_eye_submission.jsonl` by default. CoT selection is recorded in submission metadata, not in the filename or JSONL condition. The file contains exactly:

```jsonl
{"question_id":"t2_dynamic_isomorphism_0000","condition":"standard","answer":"B"}
```

Partial runs and runs with failed or empty responses create diagnostics only. They never create a submission file.
