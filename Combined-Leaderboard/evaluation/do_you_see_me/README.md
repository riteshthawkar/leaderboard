# Do You See Me evaluation

This package runs the perception benchmark and creates one upload-ready JSONL file. Run commands from the repository root.

## OpenAI-compatible or vLLM endpoint

```bash
pip install -r evaluation/requirements-vllm.txt
python -m evaluation.do_you_see_me.run_vllm \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --endpoints http://localhost:8000/v1
```

Use `--image-root /path/to/dataset` for local images. Without it, the runner uses the public `image_url` in the question bundle.

## Hugging Face transformers

```bash
pip install -r evaluation/requirements-hf.txt
python -m evaluation.do_you_see_me.run_hf \
  --model Qwen/Qwen3-VL-8B-Instruct
```

For multi-GPU sharding, run one process per shard with `--shard` and `--nshards`, then merge every generated diagnostics file:

```bash
python -m evaluation.do_you_see_me.merge_shards \
  evaluation/do_you_see_me/results/*_shard_*_of_*.diagnostics.jsonl
```

The final file is `evaluation/do_you_see_me/results/do_you_see_me_submission.jsonl` by default. CoT selection is recorded in submission metadata, not in the filename or JSONL condition. The file contains exactly:

```jsonl
{"question_id":"t1_2d_shape_discrimination_easy_0000","condition":"standard","answer":"3"}
```

Partial runs and runs with failed or empty responses create diagnostics only. They never create a submission file.
