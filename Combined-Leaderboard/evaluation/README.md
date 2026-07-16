# Evaluation pipelines

Evaluation tooling is isolated from the production API and frontend:

- `do_you_see_me/` produces the perception submission JSONL.
- `minds_eye/` produces the visual cognition submission JSONL.
- `spatial_reasoning/` produces the spatial reasoning proof bundle.
- `common/` contains strict shared loading, inference, and export code.

Canonical visual submission files are written only after complete question coverage, unique identifiers, successful inference, and parseable nonempty final answers have been verified. Raw model responses remain in separate diagnostic files.

## Research profile

The visual suite uses original checkpoint tensors with BF16 computation. In this documentation, **unquantized BF16** means the checkpoint's original released BF16 precision, not 4-bit or 8-bit weight loading. Converting a BF16 checkpoint to FP32 would not restore information that is absent from the released tensors and would only increase memory use.

The two benchmark protocols are intentionally different:

| Track | Primary prompt | Temperature | Top P | Maximum output tokens | Reference |
| --- | --- | ---: | ---: | ---: | --- |
| Do You See Me | Direct answer, non-CoT | 1.0 | 0.95 | 200 | [Paper, Appendix C](https://arxiv.org/abs/2506.02022) |
| Mind's Eye | Structured CoT | 0.1 | 1.0 | 1,000 | [Paper and released evaluation code](https://arxiv.org/abs/2604.16054) |

The Do You See Me paper applies the same temperature, nucleus sampling, and token cap to its local and API models. Its main benchmark is direct visual questioning; CoT is a separate ablation. The Mind's Eye main table reports CoT prompting, and its released open-model handlers use roughly 1,000 output tokens and a temperature of 0.1. The Mind's Eye release is not fully uniform about sampling across model-specific handlers, so the leaderboard fixes one documented profile for all models.

All models also use neutral shared sampling defaults: `top_k=-1`, `min_p=0`, presence and frequency penalties of `0`, and repetition penalty `1`. Repository-specific generation defaults and model-card sampling recommendations are not silently applied because that would give different decoding policies to different leaderboard entries.

Both papers use an expert LLM to extract answers from free-form model output. This suite instead asks for an explicit final-answer format and applies a strict local parser. That avoids a second judge model and ground-truth leakage, but it is a deliberate protocol difference. These results are therefore leaderboard evaluations on the pinned public bundle, not exact reproductions of the papers' historical tables.

For Qwen3.5, thinking is disabled on Do You See Me to implement the direct-answer condition and enabled on Mind's Eye to implement the CoT condition. No model-specific thinking override is applied to other checkpoints.

## What is not changed for speed

- No BitsAndBytes, AWQ, GPTQ, FP8, or other weight quantization. The KV cache is also pinned to BF16 rather than FP8.
- No benchmark image resize, tiling override, downsampling, or recompression in the runner.
- No reduced 4,096 or 8,192 token context; every configured model receives a 32,768 token context.
- No speculative decoding, LoRA adapter, CPU weight offload, synthetic answer, or invalid-answer sentinel.
- No shorter output budget on retries. Every attempt uses the same track protocol.

The runner supplies the original image bytes without its own resize or recompression step. A checkpoint's native vision processor may still resize, crop, or tile the image as defined by that checkpoint; the suite does not override those model-specific operations.

The performance-only controls are vLLM serving, one active request by default, checkpoint/resume, model cache reuse when requested, and optional tensor parallelism. Tensor parallelism distributes the same unquantized tensors over multiple GPUs. It does not quantize the model, although parallel floating-point reductions can produce very small numerical differences near tied token probabilities.

The papers' released code uses model-specific Transformers handlers, while this suite uses vLLM's supported multimodal path to make the multi-model run operationally consistent. This is not expected to reduce model capability, but it is not bit-for-bit equivalent to the paper handlers and can produce small output differences. Use the original paper repositories when an exact historical reproduction is required.

## Host prerequisites

- Linux with a working NVIDIA driver and at least one free 40 GB-class GPU with native BF16 support.
- At least 32 GB of system RAM.
- Python 3.10 through 3.14 with `venv` support.
- `curl` and at least 50 GiB of free disk per concurrently downloaded model.
- Internet access to Hugging Face. `HF_TOKEN` is recommended to avoid anonymous rate limits.

The script creates one shared pinned environment at `.venv/visual-suite`. It does not require a separately installed CUDA toolkit, but the installed NVIDIA driver must support the PyTorch CUDA runtime selected by `uv`.

## One model at a time

```bash
git clone https://github.com/riteshthawkar/leaderboard.git
cd leaderboard/Combined-Leaderboard
export HF_TOKEN="hf_your_token"

# Prepare one shared environment and the pinned dataset without loading a model.
GPU_IDS=0 SETUP_ONLY=1 bash evaluation/run_visual_suite.sh

# Review the exact resolved protocol without starting a model.
DRY_RUN=1 MODELS=internvl35-8b bash evaluation/run_visual_suite.sh

# Run one unquantized model on one GPU.
GPU_IDS=0 MODELS=internvl35-8b FORCE=1 \
  bash evaluation/run_visual_suite.sh
```

The available slugs are:

| Slug | Checkpoint | Weight loading | Context |
| --- | --- | --- | ---: |
| `qwen35-9b` | `Qwen/Qwen3.5-9B` | Original, unquantized BF16 | 32,768 |
| `internvl35-8b` | `OpenGVLab/InternVL3_5-8B` | Original, unquantized BF16 | 32,768 |
| `glm46v-flash` | `zai-org/GLM-4.6V-Flash` | Original, unquantized BF16 | 32,768 |
| `minicpm-v46` | `openbmb/MiniCPM-V-4.6` | Original, unquantized BF16 | 32,768 |
| `qwen25-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | Original, unquantized BF16 | 32,768 |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | Original, unquantized BF16 | 32,768 |
| `phi4-multimodal` | `microsoft/Phi-4-multimodal-instruct` | Original, unquantized BF16 | 32,768 |

Run inside `tmux` or `screen`. Each complete model receives 5,299 requests across the two tracks.

## Two models on two A100 GPUs

Use one shared environment and dataset cache, with one independent model per GPU:

```bash
GPU_GROUPS='0;1' \
MODEL_LIST=internvl35-8b,minicpm-v46 \
FORCE=1 \
  bash evaluation/run_visual_suite_multi_gpu.sh
```

To split each model over two GPUs instead, use two comma-separated GPU IDs inside each semicolon-separated worker group:

```bash
GPU_GROUPS='0,1;2,3' \
MODEL_LIST=internvl35-8b,minicpm-v46 \
FORCE=1 \
  bash evaluation/run_visual_suite_multi_gpu.sh
```

Run a dry check first by adding `DRY_RUN=1`. `GPU_IDS=0,1` remains supported by the multi-GPU wrapper as legacy shorthand for two independent one-GPU workers; use `GPU_GROUPS` whenever tensor parallel grouping is needed.

## Reliability and audit behavior

- A strict 20-sample smoke test runs before each full track.
- Diagnostics are saved atomically every 25 new responses.
- A rerun preserves parseable responses and requests only missing, failed, or unparseable samples.
- Format recovery uses a deterministic seed sequence beginning at seed 0. It never checks correctness when deciding whether to retry.
- Every failed formatting pass is archived as `<track>.attempt-N.diagnostics.jsonl`; its hash is included in the final manifest.
- After the configured attempts, any remaining invalid response fails the track. It is never replaced with a scoreable placeholder.
- Model, dataset, dependency, prompt, and source revisions are pinned or hashed.
- A schema-v4 run fingerprint prevents checkpoints from different dtypes, prompts, generation settings, context limits, or tensor-parallel topologies from being mixed.
- The final manifest records weight loading, BF16 compute and KV-cache dtypes, GPU assignment, tensor parallel size, per-track generation settings, and output hashes.
- MiniCPM-V-4.6 applies a version-checked vLLM 0.25.1 weight mapping correction. The patch ID and source hash are recorded.
- Model caches are removed after each attempt by default. Set `KEEP_MODEL_CACHE=1` only when disk capacity is sufficient.

`FORCE=1` is required when replacing earlier 4-bit results. The unquantized runner defaults to `evaluation/results/visual_suite_bf16/`, so old quantized outputs are not silently mixed into the new record.

## Outputs

Each model writes to `evaluation/results/visual_suite_bf16/<model-slug>/`:

- `do_you_see_me_submission.jsonl`: canonical perception responses.
- `minds_eye_submission.jsonl`: canonical cognition responses.
- `<track>.diagnostics.jsonl`: raw model responses, finish reasons, token counts, and errors.
- `.run_config.json`: immutable checkpoint compatibility fingerprint.
- `run_manifest.json`: final protocol, hardware, provenance, and output hashes.
- `vllm.log`: model download, startup, and serving diagnostics.

Retain diagnostics and manifests with the experimental record. Only the two submission JSONL files are uploaded as visual benchmark responses.
