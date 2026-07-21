# Evaluation pipelines

Evaluation tooling is isolated from the production API and frontend:

- `do_you_see_me/` produces the perception submission JSONL.
- `minds_eye/` produces the visual cognition submission JSONL.
- `spatial_reasoning/` produces the spatial reasoning proof bundle.
- `common/` contains strict shared loading, inference, and export code.

Canonical visual submission files are written only after complete question coverage, unique identifiers, and one gold-blind LLM extraction decision for every model response. `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218` is the default dedicated extractor with native thinking disabled. A clear commitment becomes the normalized answer; an empty, missing, truncated, conflicting, or ambiguous response becomes the standardized `UNRESOLVED` token. Raw model responses remain unchanged in separate diagnostic files.

## Standard evaluation lifecycle

`run_visual_suite.sh` is the standard runner for every model family. Benchmark semantics are shared; model-specific code is limited to a pinned profile that declares checkpoint revision, context, reasoning mode, completion budget, chat-template arguments, serving topology, and any audited compatibility override. Adding a model must not create a separate benchmark prompt, image path, parser, retry policy, or export implementation.

Use this lifecycle for every evaluation:

1. Run `DRY_RUN=1` and verify the resolved checkpoint, context, precision, topology, and per-track generation policy.
2. Run both tracks in a unique staging output root. A live run writes `.active-run.json`; cleanup tools refuse to prune while that PID exists.
3. Let the strict smoke test and complete visual pass run once. Every visual response is immediately followed by mandatory text-only extraction; there are no answer-format reruns, archived-response selection, local-parser answer selection, or raw-output fallback.
4. Dry-run final selection, then atomically rebuild the canonical result set:

```bash
python -m evaluation.finalize_visual_results --dry-run
python -m evaluation.finalize_visual_results
python -m evaluation.finalize_visual_results --verify-only
```

5. After every intended model appears in `evaluation/results/final/index.json`, remove superseded `visual_suite*` staging roots:

```bash
python -m evaluation.finalize_visual_results --prune-source-runs --prune-cache
```

The finalizer accepts only the current unquantized BF16 pipeline revision, excludes live staging roots, validates exact question order and coverage for both benchmarks, verifies every submission answer against its diagnostic source, selects the newest valid track for a pinned model revision, and preserves track-specific run configurations. It treats the existing verified canonical root as a baseline, so a future staging run adds or updates models after source pruning instead of replacing earlier final results. This allows DYS and Mind's Eye to come from different serving topologies without falsely presenting them as one run. Cache pruning is explicit because it removes the downloaded dataset and model weights; future runs recreate them from pinned sources.

## Model integration contract

A new model profile is complete only when all of the following are explicit and tested:

- Immutable repository revision, original checkpoint loading, BF16 compute, resolved BF16 KV cache, and a context no larger than the checkpoint contract.
- Thinking classification, supported chat-template arguments, per-track completion ceilings, stop handling, and the separate 200-token extracted-answer check.
- A valid single-replica, tensor-parallel, built-in data-parallel, or independent-replica topology with model identity checked at every endpoint.
- Gated access, required adapters, alternate official checkpoint sources, legacy engine mode, or config overrides represented in `.run_config.json` and the final manifest.
- A profile dry-run regression plus mandatory extraction, `UNRESOLVED`, resume, manifest, and finalizer tests.

Model-specific compatibility work may restore omitted released defaults or required adapters, but it must never change checkpoint tensors, benchmark images, prompts, sampling policy, or answers.

## Research profile

The visual suite uses original checkpoint tensors with BF16 computation. In this documentation, **unquantized BF16** means the checkpoint's original released BF16 precision, not 4-bit or 8-bit weight loading. Converting a BF16 checkpoint to FP32 would not restore information that is absent from the released tensors and would only increase memory use.

The two benchmark prompt and sampling protocols are intentionally different:

| Track | Primary prompt | Temperature | Top P | Reference |
| --- | --- | ---: | ---: | --- |
| Do You See Me | Direct answer, non-CoT | 1.0 | 0.95 | [Paper, Appendix C](https://arxiv.org/abs/2506.02022) |
| Mind's Eye | Structured CoT | 0.1 | 1.0 | [Paper and released evaluation code](https://arxiv.org/abs/2604.16054) |

Generation budgets follow the checkpoint rather than the benchmark name. Qwen3.5, InternVL3.5, GLM-4.6V-Flash, MiniCPM-V-4.6, Qwen2.5-VL, and Qwen3-VL use a hard 8,192-token completion ceiling on both tracks, configurable with `INTERNVL35_MAX_TOKENS`; Qwen3.6 uses the same default ceiling through `QWEN36_MAX_TOKENS`. Gemma 3, Kimi-VL, and Llama 3.2 Vision keep the paper-compatible 200-token DYS cap but use `INTERNVL35_MAX_TOKENS` for Mind's Eye. After each generation, the dedicated extractor emits one answer or `UNRESOLVED`; its extracted answer is checked separately against the 200-token final-answer limit. The raw model response is never truncated or rewritten. Every run fingerprint records both model contracts and the extraction decision provenance.

The Do You See Me paper applies temperature 1.0, top-p 0.95, and a 200-token completion cap to its evaluated models. The Mind's Eye main table reports CoT prompting, and its released open-model handlers use roughly 1,000 total completion tokens and temperature 0.1. The model-based budget is a deliberate cross-benchmark policy requested for this suite: models observed to produce longer supported reasoning receive enough room to reach an explicit answer without allowing malformed generations to consume the full context. For CoT prompts, vLLM also stops on and retains the closing `</answer>` delimiter.

All models also use neutral shared sampling defaults: `top_k=-1`, `min_p=0`, presence and frequency penalties of `0`, and repetition penalty `1`. Repository-specific generation defaults and model-card sampling recommendations are not silently applied because that would give different decoding policies to different leaderboard entries.

Both papers use an expert LLM to extract answers from free-form model output. This suite makes that extraction authoritative for every sample. The default extractor is pinned `Qwen/Qwen3-8B`, served separately from the evaluated visual checkpoint with `enable_thinking=false`, temperature 0, top-p 1, a 200-token completion cap, and an exact `<answer>...</answer>` contract. It receives the public question, answer type, allowed answer domain, required output format, response finish metadata, and raw candidate response. It receives no image, image reference, or ground truth. It must not solve the question; it may only identify the response's final clear commitment. Empty, missing, truncated-before-commitment, conflicting, ambiguous, malformed, or unsupported responses become `UNRESOLVED`. Extractor transport failures stop finalization rather than silently producing missing answers. The raw response, extractor transcript, model/revision, status, and response hash remain in diagnostics.

InternVL3.5 is explicitly classified as a thinking checkpoint. Qwen3.5 and Qwen3.6 are unified checkpoints that think by default, but these profiles explicitly evaluate their supported nonthinking mode with `enable_thinking=false`; benchmark-level CoT is supplied by the shared prompt rather than hidden model-native reasoning. GLM-4.6V-Flash is also classified as nonthinking and receives the pinned template's supported `enable_thinking=false` argument. InternVL3.5's pinned template has no `enable_thinking` switch, so no unsupported template argument is injected; its completion is capped at 8,192 tokens by default and its extracted answer is checked separately. The remaining configured instruction checkpoints are classified as nonthinking unless their pinned model contract is updated and revalidated.

## What is not changed for speed

- No BitsAndBytes, AWQ, GPTQ, FP8, or other weight quantization. The KV cache is also pinned to BF16 rather than FP8.
- No benchmark image resize, tiling override, downsampling, or recompression in the runner.
- No checkpoint is configured beyond or below its released context window. DeepSeek-VL2 uses its native 4,096-token limit; every other configured model receives a 32,768-token context. InternVL's 8,192-token completion ceiling is not a context-window reduction.
- No speculative decoding, speed-only LoRA adapter, CPU weight offload, synthetic answer, local-parser answer selection, or raw-response substitution. `UNRESOLVED` explicitly represents no extractable final commitment. Phi's checkpoint-required vision adapter is the documented exception.
- No answer-format generation retries. Transient HTTP retries may repeat a failed transport request, but they do not select among different model answers.

The runner supplies the original image bytes without its own resize or recompression step. A checkpoint's native vision processor may still resize, crop, or tile the image as defined by that checkpoint; the suite does not override those model-specific operations.

The performance-only controls are vLLM serving, one active request by default, checkpoint/resume, model cache reuse when requested, and optional tensor parallelism. Tensor parallelism distributes the same unquantized tensors over multiple GPUs. It does not quantize the model, although parallel floating-point reductions can produce very small numerical differences near tied token probabilities.

The papers' released code uses model-specific Transformers handlers, while this suite uses vLLM's supported multimodal path to make the multi-model run operationally consistent. This is not expected to reduce model capability, but it is not bit-for-bit equivalent to the paper handlers and can produce small output differences. Use the original paper repositories when an exact historical reproduction is required.

Phi-4 Multimodal is served with the checkpoint's bundled `vision-lora` adapter, as required by Microsoft's official vLLM recipe. The base checkpoint intentionally excludes those LoRA tensors during vLLM weight loading, so requests target the registered `vision` adapter alias. The adapter source and checkpoint revision are recorded in the run fingerprint and manifest.

If Hugging Face's repository transport is unavailable, `PHI_OFFICIAL_SNAPSHOT_PATH` may point to an immutable checkout of Microsoft's official ModelScope repository at commit `7641bf905e6965ee54166808d275266371e28343`. The runner verifies the SHA-256 of every base-model shard and the vision adapter before loading it. The canonical Hugging Face model and revision remain in the fingerprint, while `checkpoint_source` records the actual provider, commit, and object hashes. An unverified mirror or a moving branch is never accepted.

## Host prerequisites

- Linux with a working NVIDIA driver and enough BF16 GPU memory for the evaluated checkpoint plus a dedicated Qwen3-8B extractor. The managed default uses a separate GPU; an externally managed extractor endpoint is also supported.
- At least 32 GB of system RAM.
- Python 3.10 through 3.14 with `venv` support.
- `curl` and at least 96 GiB of free disk for a single-model run. Concurrent launches reserve 64 GiB for the host plus 32 GiB for every model cache.
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

# Override InternVL's hard completion ceiling when required.
DRY_RUN=1 MODELS=internvl35-8b INTERNVL35_MAX_TOKENS=4096 \
  bash evaluation/run_visual_suite.sh

# Run one unquantized model on GPU 0 and the managed extractor on GPU 7.
GPU_IDS=0 ANSWER_EXTRACTOR_GPU_IDS=7 MODELS=internvl35-8b FORCE=1 \
  bash evaluation/run_visual_suite.sh

# Run one 8B model as four replicas for higher throughput without changing
# prompts, sampling, token policy, image bytes, or checkpoint precision.
GPU_IDS=0,1,2,3 ANSWER_EXTRACTOR_GPU_IDS=7 \
TENSOR_PARALLEL_SIZE=1 DATA_PARALLEL_SIZE=4 CONCURRENCY=4 \
MODELS=internvl35-8b FORCE=1 \
  bash evaluation/run_visual_suite.sh

# Migrate every raw response in an existing complete diagnostics file to the
# mandatory extractor contract without loading the visual checkpoint.
GPU_IDS=7 ANSWER_EXTRACTOR_GPU_IDS=7 MODELS=internvl35-8b \
TRACKS=do_you_see_me EXTRACT_EXISTING_ONLY=1 \
  bash evaluation/run_visual_suite.sh

# Alternatively, use a separately supervised Qwen3-8B endpoint.
GPU_IDS=0 ANSWER_EXTRACTOR_ENDPOINTS=http://127.0.0.1:8035/v1 \
MODELS=internvl35-8b FORCE=1 bash evaluation/run_visual_suite.sh
```

For checkpoints that fit on one A100, data parallelism is the preferred throughput topology: `TENSOR_PARALLEL_SIZE=1`, `DATA_PARALLEL_SIZE` equal to the number of visual-model GPUs, and matching `CONCURRENCY`. Managed extractor GPUs must not overlap visual-model GPUs during normal inference. Tensor parallelism is reserved for checkpoints that need multiple GPUs for one request. Both serving contracts and request concurrency are recorded in the schema-v11 fingerprint and final manifest.

The available slugs are:

| Slug | Checkpoint | Weight loading | Context |
| --- | --- | --- | ---: |
| `qwen35-9b` | `Qwen/Qwen3.5-9B` | Original, unquantized BF16 | 32,768 |
| `qwen36-27b` | `Qwen/Qwen3.6-27B` | Original, unquantized BF16 | 32,768 |
| `internvl35-8b` | `OpenGVLab/InternVL3_5-8B` | Original, unquantized BF16 | 32,768 |
| `glm46v-flash` | `zai-org/GLM-4.6V-Flash` | Original, unquantized BF16 | 32,768 |
| `minicpm-v46` | `openbmb/MiniCPM-V-4.6` | Original, unquantized BF16 | 32,768 |
| `qwen25-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | Original, unquantized BF16 | 32,768 |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | Original, unquantized BF16 | 32,768 |
| `phi4-multimodal` | `microsoft/Phi-4-multimodal-instruct` | Original, unquantized BF16 | 32,768 |
| `gemma3-12b-it` | `google/gemma-3-12b-it` | Original, unquantized BF16 | 32,768 |
| `gemma3-27b-it` | `google/gemma-3-27b-it` | Original, unquantized BF16 | 32,768 |
| `kimi-vl-a3b-instruct` | `moonshotai/Kimi-VL-A3B-Instruct` | Original, unquantized BF16 | 32,768 |
| `deepseek-vl2` | `deepseek-ai/deepseek-vl2` | Original, unquantized BF16 | 4,096 |
| `llama32-11b-vision-instruct` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Original, unquantized BF16 | 32,768 |

Gemma 3 is license-gated on Hugging Face. Accept Google's Gemma usage license for the account associated with `HF_TOKEN` before selecting `gemma3-12b-it` or `gemma3-27b-it`; the runner remains pinned to the recorded checkpoint revision after access is granted. The unquantized 27B checkpoint requires tensor parallelism across two 40 GB A100s.

Kimi-VL-A3B-Instruct uses three independent single-GPU vLLM processes when three A100s are assigned (`TENSOR_PARALLEL_SIZE=1`, `DATA_PARALLEL_SIZE=3`, `CONCURRENCY=3`, `SERVING_REPLICA_MODE=independent`). Built-in vLLM MoE DP flattens the three-way DP world into tensor sharding and fails because the model's 1,408-wide MoE intermediate dimension is not divisible by three; tensor parallel size 3 is also invalid for its 16 attention heads and 64 routed experts. The independent replica mode preserves full model math, exposes one endpoint per GPU, and is recorded in the run fingerprint and manifest. DYS retains the 200-token direct-answer cap, while Mind's Eye uses the 8,192-token structured-CoT allowance. Set `ALLOW_SMOKE_RAW_OUTPUT_FALLBACK=1` to admit a smoke only after all retries and same-model extraction when every smoke record is complete, non-error, and nonempty; this admission changes no canonical output and is recorded in provenance.

The full DeepSeek-VL2 checkpoint has 27.5B total MoE parameters and a native 4,096-token context. Its original BF16 tensors require tensor parallelism across two 40 GB A100s. The pinned JSON relies on defaults from DeepSeek's remote configuration class, but vLLM 0.25.1 replaces the omitted `kv_lora_rank=512` with `None` and the omitted `num_hidden_layers=30` with `32`. The immutable tensor index contains exactly 30 MLA layers. The runner therefore supplies `--hf-overrides '{"text_config":{"kv_lora_rank":512,"num_hidden_layers":30}}'`; the literal override and compatibility ID are recorded in provenance and do not alter any checkpoint tensor. The validated launch contract uses only GPUs 0 and 1, one active sequence, and PyNCCL collectives:

```bash
GPU_IDS=0,1 TENSOR_PARALLEL_SIZE=2 DATA_PARALLEL_SIZE=1 CONCURRENCY=1 \
MAX_NUM_SEQS_PER_REPLICA=1 DISABLE_CUSTOM_ALL_REDUCE=1 \
MODELS=deepseek-vl2 FORCE=1 bash evaluation/run_visual_suite.sh
```

Llama 3.2 11B Vision Instruct is Meta's latest image-capable checkpoint that fits comfortably on a 40 GB A100 and is accessible to the configured Hugging Face account. Meta released it on September 25, 2024 with a 128K native window; this suite evaluates the same 32,768-token context used by its other compatible profiles. The checkpoint is license-gated, so accept the Llama 3.2 Community License before running it. Current vLLM removed the encoder-decoder `MllamaForConditionalGeneration` implementation after v0.10.2, so this profile uses an isolated pinned vLLM 0.10.2 environment with `VLLM_USE_V1=0`. The older CLI's `--kv-cache-dtype auto` resolves to BF16 for this BF16 model; both the resolved dtype and CLI value are recorded in provenance. Meta's repository also contains a redundant original-format `consolidated.pth`; prefetch excludes it because vLLM loads the five pinned Transformers safetensor shards. Two independent replicas provide throughput without changing model math:

```bash
VENV_DIR="$PWD/.venv/visual-suite-mllama-vllm-0.10.2" \
VLLM_VERSION=0.10.2 HUGGINGFACE_HUB_VERSION=0.34.4 \
GPU_IDS=0,1 TENSOR_PARALLEL_SIZE=1 DATA_PARALLEL_SIZE=2 CONCURRENCY=2 \
MAX_NUM_SEQS_PER_REPLICA=1 SERVING_REPLICA_MODE=independent \
MODELS=llama32-11b-vision-instruct FORCE=1 \
  bash evaluation/run_visual_suite.sh
```

Qwen3.6-27B BF16 requires tensor parallelism across two 40 GB A100s. On the validated PCIe-connected GPU pair, set `TENSOR_PARALLEL_SIZE=2` and `DISABLE_CUSTOM_ALL_REDUCE=1`; the latter routes collectives through PyNCCL after vLLM's custom all-reduce kernel failed during CUDA-graph warmup. This serving choice is recorded in the run fingerprint.

Run inside `tmux` or `screen`. Each complete model receives 5,299 requests across the two tracks.

## Two models on two A100 GPUs

Use one shared environment and dataset cache, with one independent model per GPU:

```bash
GPU_GROUPS='0;1' \
MODEL_LIST=internvl35-8b,minicpm-v46 \
ANSWER_EXTRACTOR_ENDPOINTS=http://127.0.0.1:8035/v1 \
FORCE=1 \
  bash evaluation/run_visual_suite_multi_gpu.sh
```

To split each model over two GPUs instead, use two comma-separated GPU IDs inside each semicolon-separated worker group:

```bash
GPU_GROUPS='0,1;2,3' \
MODEL_LIST=internvl35-8b,minicpm-v46 \
ANSWER_EXTRACTOR_ENDPOINTS=http://127.0.0.1:8035/v1 \
FORCE=1 \
  bash evaluation/run_visual_suite_multi_gpu.sh
```

Run a dry check first by adding `DRY_RUN=1`. `GPU_IDS=0,1` remains supported by the multi-GPU wrapper as legacy shorthand for two independent one-GPU workers; use `GPU_GROUPS` whenever tensor parallel grouping is needed.

Do not start concurrent single-model runners on the same `PORT`. The multi-GPU launcher allocates a distinct visual-model port for every worker. Concurrent workers share an externally supervised Qwen3-8B extractor through `ANSWER_EXTRACTOR_ENDPOINTS`; the endpoint identity is checked by every worker before evaluation.

The multi-model launcher also protects 64 GiB of host free space and budgets 32 GiB for each checkpoint being downloaded. If the model cache filesystem is small, run models sequentially or set `CACHE_ROOT` to a larger mounted filesystem. Do not lower these guards merely to bypass a failed preflight.

## Reliability and audit behavior

- A strict 20-sample smoke test runs before each full track.
- Diagnostics are saved atomically every 25 new responses.
- A resume preserves rows that already have the current mandatory extractor decision and requests only missing rows.
- Each sample has one visual response and one authoritative Qwen3-8B extraction decision. No alternate visual answer is generated for formatting recovery.
- Empty, missing, conflicting, ambiguous, unsupported, or truncated-before-commitment responses become `UNRESOLVED`. Extractor service or transport failures stop finalization.
- Existing complete diagnostics may be migrated with `EXTRACT_EXISTING_ONLY=1`; this sends every stored raw response to Qwen3-8B without loading the visual checkpoint or consulting archived attempts.
- Model, dataset, dependency, prompt, and source revisions are pinned or hashed.
- A schema-v11 fingerprint prevents checkpoints from different dtypes, prompts, reasoning profiles, completion contracts, extractor models/revisions, context limits, or parallel topologies from being mixed. The only supported in-place upgrade is extraction-only migration from complete schema-v10 diagnostics; raw response bytes are preserved.
- The final manifest records weight loading, BF16 compute and KV-cache dtypes, GPU assignment, tensor parallel size, per-track generation settings, and output hashes.
- MiniCPM-V-4.6 applies a version-checked vLLM 0.25.1 weight mapping correction. The patch ID and source hash are recorded.
- Visual-model caches are removed after each completed model by default. Set `KEEP_MODEL_CACHE=1` only when disk capacity is sufficient.

`FORCE=1` is required when replacing earlier 4-bit results. The unquantized runner defaults to `evaluation/results/visual_suite_bf16/`, so old quantized outputs are not silently mixed into the new record. For repeated experiments, set a unique `OUTPUT_ROOT`; the finalizer discovers valid staging roots recursively.

## Outputs

Each staging run writes to `<OUTPUT_ROOT>/<model-slug>/`:

- `do_you_see_me_submission.jsonl`: canonical perception responses.
- `minds_eye_submission.jsonl`: canonical cognition responses.
- `<track>.diagnostics.jsonl`: raw model responses, finish reasons, token counts, inference errors, and mandatory extractor provenance.
- `.run_config.json`: immutable checkpoint compatibility fingerprint.
- `run_manifest.json`: final protocol, hardware, provenance, and output hashes.
- `vllm.log`: model download, startup, and serving diagnostics.

The canonical root is `evaluation/results/final/`:

- `index.json`: authoritative model inventory, pinned revisions, row counts, resolved-answer counts, `UNRESOLVED` counts, legacy raw-fallback counts, and per-model manifest hashes.
- `<model-slug>/final_manifest.json`: track-specific source run, topology, generation contract, answer provenance, and every retained artifact hash.
- `<model-slug>/<track>_submission.jsonl`: uploadable canonical benchmark response.
- `<model-slug>/<track>.diagnostics.jsonl`, run configurations, and source manifests: retained audit record. Legacy canonical models may also retain pre-schema-v11 retry archives.

Retain the canonical diagnostics and manifests with the experimental record. Only the two submission JSONL files are uploaded as benchmark responses. Staging roots and `.cache` are disposable after finalization and after all active runs have stopped.

## Gold-blind canonical response audit

Every stored visual response receives one authoritative response-commitment audit, including responses that already match the output contract. This is not a second attempt to solve the image. The extraction process receives only public question text, the task-specific answer contract, response metadata, and the stored model response. It has no ground-truth CLI argument and receives neither images nor answer keys.

Pinned `Qwen/Qwen3-8B` must return `COMMITTED` with the selected answer and a literal source quote, or `UNRESOLVED` with empty answer and evidence fields. Deterministic code independently validates the quote, commitment language, truncation state, and task-specific answer domain. Unsupported or ambiguous responses become `UNRESOLVED`; explicit out-of-domain commitments become `__INVALID_FORMAT__`. Request and schema failures block production packaging.

```bash
python -m evaluation.extract_canonical_answers \
  --canonical-root evaluation/results/final-extracted-v11 \
  --endpoint http://127.0.0.1:8035/v1 \
  --endpoint http://127.0.0.1:8036/v1 \
  --model Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --policy all \
  --concurrency 64 \
  --max-tokens 256 \
  --output /share/data/visual-answer-extraction/qwen3-8b-evidence-v4/audit.jsonl
```

Resume is append-only and requires the same method, extractor contract, candidate set, and response hashes. Earlier extraction checkpoints are intentionally incompatible. If repeated extractor attempts leave schema failures, an offline finalization pass may classify only literal terminal source commitments; it fails closed on every other blocker and preserves all extractor attempts:

```bash
python -m evaluation.extract_canonical_answers \
  --canonical-root evaluation/results/final-extracted-v11 \
  --policy all \
  --max-tokens 256 \
  --output /share/data/visual-answer-extraction/qwen3-8b-evidence-v4/audit.jsonl \
  --finalize-checkpoint
```

After all rows are complete, build a separate production tree; the v11 source remains unchanged:

```bash
python -m evaluation.build_production_visual_results \
  --source-root evaluation/results/final-extracted-v11 \
  --audit /share/data/visual-answer-extraction/qwen3-8b-evidence-v4/audit.jsonl \
  --output-root evaluation/results/final-extracted-v12
```

The builder hash-checks every source artifact, requires exact audit coverage, reruns the deterministic classifier, excludes the two Qwen3.5 mode ablations, and publishes atomically only after canonical verification. Verify and score the resulting 12-model tree before import:

```bash
python -m evaluation.finalize_visual_results \
  --verify-only \
  --output-root evaluation/results/final-extracted-v12

python scripts/import_canonical_visual_results.py \
  --result-root evaluation/results/final-extracted-v12
```

The importer is dry-run only unless `--apply` is supplied. It is the first step in this flow that loads private ground truth, and it requires production scorer invalid-format counts to match each retained manifest exactly.
