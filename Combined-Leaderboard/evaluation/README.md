# Evaluation pipelines

Evaluation tooling is isolated from the production API and frontend. Each benchmark has its own package:

- `do_you_see_me/` produces the perception submission JSONL.
- `minds_eye/` produces the visual cognition submission JSONL.
- `spatial_reasoning/` runs the six-condition spatial evaluation and produces `spatial_reasoning_submission.zip`.
- `common/` contains strict shared code used by the two visual runners.

The visual pipelines emit raw diagnostic records separately from canonical submission files. Canonical files are written only after complete question coverage, unique identifiers, successful inference, and nonempty final answers have been verified.

The backend and frontend do not import these evaluation packages at runtime. The API packages `spatial_reasoning/` only when a user requests the official spatial harness download.

## One-command visual evaluation suite

`run_visual_suite.sh` provisions an isolated Python environment, downloads the pinned public images and model revisions, starts one model at a time on one NVIDIA GPU, evaluates both visual tracks, validates the outputs, and stops the model before loading the next one.

### GPU host prerequisites

- Linux with a working NVIDIA driver and a 24 GB-class GPU.
- At least 32 GB of system RAM, with no other memory-heavy model process running.
- Python 3.10 through 3.14 with the `venv` module.
- `git`, `curl`, and at least 60 GiB of free disk space.
- Internet access to Hugging Face. `HF_TOKEN` is optional for these public repositories but recommended to avoid anonymous download rate limits.

The script installs pinned Python dependencies inside `.venv/visual-suite`. It does not require a separately installed CUDA toolkit, but the NVIDIA driver must be compatible with the PyTorch wheel selected by `uv`.

### Clone and run

```bash
git clone https://github.com/riteshthawkar/leaderboard.git
cd leaderboard/Combined-Leaderboard
export HF_TOKEN="hf_your_token"  # Recommended; omit for anonymous downloads.
bash evaluation/run_visual_suite.sh
```

Run the process inside `tmux`, `screen`, or another persistent terminal because the full suite contains 5,299 samples per model and 37,093 model requests across seven models.

The default sequence is:

| Slug | Model | Loading | Prompt |
| --- | --- | --- | --- |
| `qwen35-9b` | `Qwen/Qwen3.5-9B` | 4-bit BitsAndBytes | Non-CoT with thinking disabled |
| `internvl35-8b` | `OpenGVLab/InternVL3_5-8B` | Full precision with a 4,096-token context | Non-CoT |
| `glm41v-9b-thinking` | `zai-org/GLM-4.1V-9B-Thinking` | 4-bit BitsAndBytes | CoT |
| `minicpm-v46` | `openbmb/MiniCPM-V-4.6` | 4-bit BitsAndBytes | Non-CoT |
| `qwen25-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | 4-bit BitsAndBytes | Non-CoT |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | 4-bit BitsAndBytes | Non-CoT |
| `phi4-multimodal` | `microsoft/Phi-4-multimodal-instruct` | BF16, or FP16 when BF16 is unavailable | Non-CoT |

InternVL3.5 remains full precision because its vLLM model class does not support the BitsAndBytes loader. Phi-4 multimodal also remains full precision because its built-in vision adapter is incompatible with BitsAndBytes quantization. Both use a 4,096-token context to fit a 24 GB card; the other models use an 8,192-token server context. Quantized runs must be disclosed as 4-bit inference in the leaderboard method description because quantization can affect scores.

### Reliability behavior

- Every track starts with a strict 20-sample smoke test. A model with empty or unparseable responses does not proceed to the full track.
- Diagnostics are saved atomically every 25 new responses. Re-running the same command keeps valid responses and retries only missing, failed, or unparseable samples.
- Each full track receives up to three retry passes. One failed model does not prevent later models from running, but the script returns a nonzero exit status if any model remains failed.
- Exact model and dataset revisions are pinned. A run fingerprint prevents checkpoints from different prompts, revisions, dtypes, or context limits from being mixed.
- Model caches are deleted after each attempted model by default to keep disk use bounded. Set `KEEP_MODEL_CACHE=1` when retaining completed or failed weights is more important than disk use.
- Canonical submission JSONL is created only after complete question coverage, exact ordering, schema, and nonempty answers have passed validation.

Re-run the same command after a disconnect or model failure. Do not use `FORCE=1` unless the existing checkpoints should be replaced.

### Useful controls

```bash
# Validate one model and one track with only the strict smoke test.
MODELS=qwen35-9b TRACKS=minds_eye SMOKE_ONLY=1 \
  bash evaluation/run_visual_suite.sh

# Run a subset. Values are comma-separated model slugs and track names.
MODELS=internvl35-8b,qwen3-vl-8b TRACKS=do_you_see_me \
  bash evaluation/run_visual_suite.sh

# Print the resolved plan without installing or downloading anything.
DRY_RUN=1 bash evaluation/run_visual_suite.sh

# Keep downloaded weights after successful models.
KEEP_MODEL_CACHE=1 bash evaluation/run_visual_suite.sh
```

Use `bash evaluation/run_visual_suite.sh --help` for the complete short reference. The conservative defaults use one active request and 88 percent of GPU memory. Increase concurrency only after proving the selected model has adequate memory headroom.

### Outputs

Each model writes to `evaluation/results/visual_suite/<model-slug>/`:

- `do_you_see_me_submission.jsonl`: upload-ready perception responses.
- `minds_eye_submission.jsonl`: upload-ready cognition responses.
- `<track>.diagnostics.jsonl`: raw model responses and per-sample errors for audit and resume.
- `run_manifest.json`: pinned model revision, inference settings, output row counts, and SHA-256 hashes.
- `vllm.log`: model download, startup, and server errors.

The two `*_submission.jsonl` files are the files submitted to their corresponding leaderboard tracks. Diagnostics and manifests should be retained with the experimental record but are not uploaded as visual-track responses.
