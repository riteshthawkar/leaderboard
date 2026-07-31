# Remote GPU Runbook

These commands assume four 40 GB A100 GPUs and the repository root as the
current directory. Replace GPU IDs only when `nvidia-smi` confirms they are
free.

## Recommended automated run

Use the queue below for the current controlled experiments. It keeps one target
model loaded across all image conditions, runs two model families
sequentially, then loads the fixed extractor once:

```bash
mkdir -p /share/data/drive_3/omkar/ms-vista-analysis-v2/logs

setsid env \
  GPU_ID=2 \
  ANALYSIS_ROOT=/share/data/drive_3/omkar/ms-vista-analysis-v2 \
  DATASET_ROOT="$PWD/evaluation/results/.cache/visual-intelligence-dataset" \
  QWEN_TARGET_HF_HOME=/share/data/drive_3/hf_cache \
  INTERNVL_TARGET_HF_HOME=/share/data/drive_1/omkar/ms-vista-track3-v2/internvl-hf-home \
  EXTRACTOR_HF_HOME="$PWD/evaluation/results/.cache/models/qwen3-8b-extractor" \
  TARGET_SLUGS=qwen3-vl-8b,internvl35-8b \
  bash evaluation/research/run_single_gpu_analysis_queue.sh \
  >/share/data/drive_3/omkar/ms-vista-analysis-v2/logs/bootstrap.log \
  2>&1 </dev/null &
```

Monitor it with:

```bash
cat /share/data/drive_3/omkar/ms-vista-analysis-v2/status/current.tsv
tail -F /share/data/drive_3/omkar/ms-vista-analysis-v2/logs/supervisor.log
nvidia-smi -i 2
```

The queue runs deterministic paired inference at BF16 with a 32,768-token
server context. It deliberately omits the request-level completion cap. Raw
diagnostics are resumable and extraction is deferred until both target models
finish. Every final condition must pass
`analysis/research/audit_inference_quality.py`.

After completion, copy the analysis root to the trusted host without copying
private ground truth to the GPU server. Run:

```bash
.venv/bin/python analysis/research/score_analysis_queue.py \
  --root PATH_TO_DOWNLOADED_ANALYSIS_ROOT
```

The remaining sections document manual execution and the optional training
experiments.

## Scheduled Qwen3.5 protocol ablation

The follow-up watcher starts only after the controlled queue above records
successful completion and releases GPU 2:

```bash
setsid env \
  GPU_ID=2 \
  CURRENT_ANALYSIS_ROOT=/share/data/drive_3/omkar/ms-vista-analysis-v2 \
  ABLATION_ROOT=/share/data/drive_3/omkar/ms-vista-qwen35-thinking-v1 \
  DATASET_ROOT="$PWD/evaluation/results/.cache/visual-intelligence-dataset" \
  TARGET_HF_HOME=/share/data/drive_3/hf_cache \
  EXTRACTOR_HF_HOME="$PWD/evaluation/results/.cache/models/qwen3-8b-extractor" \
  bash evaluation/research/queue_qwen35_followup.sh \
  >/share/data/drive_3/omkar/ms-vista-qwen35-thinking-v1/logs/bootstrap.log \
  2>&1 </dev/null &
```

It runs the full Mind's Eye benchmark under direct/reasoning prompts crossed
with Qwen3.5 internal thinking disabled/enabled. It uses the same BF16,
deterministic, uncapped-context and fixed-extractor policy as the controlled
queue. Score the downloaded results on the trusted host:

```bash
.venv/bin/python analysis/research/analyze_qwen35_thinking_ablation.py \
  --root PATH_TO_DOWNLOADED_ABLATION_ROOT
```

## 1. Prepare the shared inference environment

```bash
GPU_IDS=0 SETUP_ONLY=1 KEEP_MODEL_CACHE=1 \
  bash evaluation/run_visual_suite.sh
```

Recreate and validate the CPU-generated inputs on the remote host:

```bash
.venv/visual-suite/bin/python evaluation/research/generate_causal_transformations.py
.venv/visual-suite/bin/python evaluation/research/prepare_visual_fidelity.py --workers 12
.venv/visual-suite/bin/python evaluation/research/prepare_state_interventions.py
.venv/visual-suite/bin/python evaluation/research/generate_perception_curriculum.py

.venv/visual-suite/bin/python evaluation/research/validate_experiment_bundle.py \
  --manifest evaluation/research/results/causal_transformations/manifest.json \
  --manifest evaluation/research/results/visual_fidelity/manifest.json \
  --manifest evaluation/research/results/state_interventions/manifest.json \
  --manifest evaluation/research/results/perception_curriculum/manifest.json
```

The fidelity preparation downloads about 560 source images and writes about
830 MB. Some original figures exceed 100 megapixels, so this CPU step can take
10 to 20 minutes.

## 2. Start a target VLM and the fixed extractor

The example uses the exact Qwen3-VL-8B and Qwen3-8B revisions already pinned by
the main evaluation pipeline.

Terminal A:

```bash
export TARGET_MODEL=Qwen/Qwen3-VL-8B-Instruct
export TARGET_REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

CUDA_VISIBLE_DEVICES=0 .venv/visual-suite/bin/vllm serve "${TARGET_MODEL}" \
  --revision "${TARGET_REVISION}" \
  --served-model-name "${TARGET_MODEL}" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --port 8000 \
  > evaluation/research/results/target-vllm.log 2>&1
```

Terminal B:

```bash
export EXTRACTOR_MODEL=Qwen/Qwen3-8B
export EXTRACTOR_REVISION=b968826d9c46dd6066d109eabc6255188de91218

CUDA_VISIBLE_DEVICES=3 .venv/visual-suite/bin/vllm serve "${EXTRACTOR_MODEL}" \
  --revision "${EXTRACTOR_REVISION}" \
  --served-model-name "${EXTRACTOR_MODEL}" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.70 \
  --port 8001 \
  > evaluation/research/results/extractor-vllm.log 2>&1
```

Check readiness:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8001/v1/models
```

The 8B target fits on one A100. For a larger checkpoint, expose two GPUs and
add `--tensor-parallel-size 2`. Do not quantize the checkpoint.

In every later terminal, set:

```bash
export TARGET_MODEL=Qwen/Qwen3-VL-8B-Instruct
export TARGET_REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
export EXTRACTOR_MODEL=Qwen/Qwen3-8B
export EXTRACTOR_REVISION=b968826d9c46dd6066d109eabc6255188de91218
```

Before every full condition matrix, run the same command once with `LIMIT=5`.
The runner validates five model and extractor responses, writes resumable
diagnostics, and deliberately does not create a partial submission. Remove
`LIMIT=5` for the full run.

## 3. Run the causal transformation experiment

```bash
ROOT=evaluation/research/results/causal_transformations
RUN=evaluation/research/results/gpu_runs/qwen3-vl-8b/causal
mkdir -p "${RUN}"

.venv/visual-suite/bin/python -m evaluation.minds_eye.run_vllm \
  --model "${TARGET_MODEL}" \
  --endpoints http://127.0.0.1:8000/v1 \
  --extractor-model "${EXTRACTOR_MODEL}" \
  --extractor-revision "${EXTRACTOR_REVISION}" \
  --extractor-endpoints http://127.0.0.1:8001/v1 \
  --extractor-chat-template-kwargs '{"enable_thinking":false}' \
  --questions "${ROOT}/questions.jsonl" \
  --image-root "${ROOT}" \
  --prompt-mode cot \
  --temperature 0.1 \
  --top-p 1.0 \
  --max-final-answer-tokens 200 \
  --concurrency 8 \
  --request-timeout 900 \
  --checkpoint-every 10 \
  --resume \
  --out "${RUN}/submission.jsonl" \
  --diagnostics "${RUN}/diagnostics.jsonl"

.venv/visual-suite/bin/python analysis/research/score_causal_transformations.py \
  --ground-truth "${ROOT}/private_ground_truth.jsonl" \
  --pairs "${ROOT}/pairs.jsonl" \
  --submission "${RUN}/submission.jsonl" \
  --output "${RUN}/score"
```

For the first causal run, add `--limit 5 --strict-partial`. After it passes,
remove those two arguments and rerun with the same diagnostics path; the five
validated rows are resumed.

Run the same 600 questions for every model selected for the paper. The primary
metric is `full_triplet_success`, which requires a correct base answer, the
correct answer change under the causal edit, and invariance under nuisance
styling.

## 4. Run the visual fidelity experiment

Do You See Me with the paper-aligned direct-answer protocol:

```bash
TRACK=do_you_see_me \
CONDITION_ROOT=evaluation/research/results/visual_fidelity/do_you_see_me \
CONDITIONS=native,downsample_50,downsample_25 \
PROMPT_MODES=noncot \
MODEL="${TARGET_MODEL}" \
MODEL_REVISION="${TARGET_REVISION}" \
EXTRACTOR_MODEL="${EXTRACTOR_MODEL}" \
EXTRACTOR_REVISION="${EXTRACTOR_REVISION}" \
ENDPOINT=http://127.0.0.1:8000/v1 \
EXTRACTOR_ENDPOINT=http://127.0.0.1:8001/v1 \
OUTPUT_ROOT=evaluation/research/results/gpu_runs/qwen3-vl-8b/fidelity/do_you_see_me \
CONCURRENCY=8 \
bash evaluation/research/run_visual_condition_matrix.sh
```

Mind's Eye with the paper-aligned reasoning protocol:

```bash
TRACK=minds_eye \
CONDITION_ROOT=evaluation/research/results/visual_fidelity/minds_eye \
CONDITIONS=native,downsample_50,downsample_25 \
PROMPT_MODES=cot \
MODEL="${TARGET_MODEL}" \
MODEL_REVISION="${TARGET_REVISION}" \
EXTRACTOR_MODEL="${EXTRACTOR_MODEL}" \
EXTRACTOR_REVISION="${EXTRACTOR_REVISION}" \
ENDPOINT=http://127.0.0.1:8000/v1 \
EXTRACTOR_ENDPOINT=http://127.0.0.1:8001/v1 \
OUTPUT_ROOT=evaluation/research/results/gpu_runs/qwen3-vl-8b/fidelity/minds_eye \
CONCURRENCY=8 \
bash evaluation/research/run_visual_condition_matrix.sh
```

Score each matched matrix:

```bash
.venv/visual-suite/bin/python analysis/research/score_condition_matrix.py \
  --track do_you_see_me \
  --baseline native \
  --condition native=evaluation/research/results/gpu_runs/qwen3-vl-8b/fidelity/do_you_see_me/native/noncot/submission.jsonl \
  --condition downsample_50=evaluation/research/results/gpu_runs/qwen3-vl-8b/fidelity/do_you_see_me/downsample_50/noncot/submission.jsonl \
  --condition downsample_25=evaluation/research/results/gpu_runs/qwen3-vl-8b/fidelity/do_you_see_me/downsample_25/noncot/submission.jsonl \
  --output evaluation/research/results/gpu_runs/qwen3-vl-8b/fidelity/do_you_see_me/score
```

Repeat with `--track minds_eye` and the corresponding `cot` paths.

## 5. Run correct and mismatched abstraction cues

```bash
TRACK=minds_eye \
CONDITION_ROOT=evaluation/research/results/state_interventions \
CONDITIONS=baseline,oracle_abstraction,mismatched_abstraction \
PROMPT_MODES=cot \
MODEL="${TARGET_MODEL}" \
MODEL_REVISION="${TARGET_REVISION}" \
EXTRACTOR_MODEL="${EXTRACTOR_MODEL}" \
EXTRACTOR_REVISION="${EXTRACTOR_REVISION}" \
ENDPOINT=http://127.0.0.1:8000/v1 \
EXTRACTOR_ENDPOINT=http://127.0.0.1:8001/v1 \
OUTPUT_ROOT=evaluation/research/results/gpu_runs/qwen3-vl-8b/abstraction_cues \
CONCURRENCY=8 \
bash evaluation/research/run_visual_condition_matrix.sh
```

Score with:

```bash
.venv/visual-suite/bin/python analysis/research/score_condition_matrix.py \
  --track minds_eye \
  --baseline baseline \
  --condition baseline=evaluation/research/results/gpu_runs/qwen3-vl-8b/abstraction_cues/baseline/cot/submission.jsonl \
  --condition oracle_abstraction=evaluation/research/results/gpu_runs/qwen3-vl-8b/abstraction_cues/oracle_abstraction/cot/submission.jsonl \
  --condition mismatched_abstraction=evaluation/research/results/gpu_runs/qwen3-vl-8b/abstraction_cues/mismatched_abstraction/cot/submission.jsonl \
  --output evaluation/research/results/gpu_runs/qwen3-vl-8b/abstraction_cues/score
```

## 6. Run perception-reasoning module swaps

First extract validated, answer-blind states from the 199 eligible Mind's Eye
items:

```bash
STATE_DIR=evaluation/research/results/gpu_runs/qwen3-vl-8b/module_swap
mkdir -p "${STATE_DIR}"

.venv/visual-suite/bin/python evaluation/research/run_state_extractor_vllm.py \
  --questions tasks/minds_eye/questions.jsonl \
  --schemas evaluation/research/results/state_interventions/state_schemas.json \
  --endpoint http://127.0.0.1:8000/v1 \
  --model "${TARGET_MODEL}" \
  --model-revision "${TARGET_REVISION}" \
  --chat-template-kwargs '{}' \
  --output "${STATE_DIR}/states.jsonl" \
  --concurrency 8 \
  --resume
```

Produce the image-plus-state condition:

```bash
.venv/visual-suite/bin/python evaluation/research/build_state_augmented_questions.py \
  --questions tasks/minds_eye/questions.jsonl \
  --states "${STATE_DIR}/states.jsonl" \
  --output "${STATE_DIR}/image_plus_state_questions.jsonl"
```

Run that question file with `evaluation.minds_eye.run_vllm` exactly as in the
causal command, retaining the image and the state.

Produce the state-only condition:

```bash
.venv/visual-suite/bin/python evaluation/research/run_text_intervention_vllm.py \
  --questions tasks/minds_eye/questions.jsonl \
  --states "${STATE_DIR}/states.jsonl" \
  --endpoint http://127.0.0.1:8000/v1 \
  --model "${TARGET_MODEL}" \
  --model-revision "${TARGET_REVISION}" \
  --extractor-endpoint http://127.0.0.1:8001/v1 \
  --extractor-model "${EXTRACTOR_MODEL}" \
  --extractor-revision "${EXTRACTOR_REVISION}" \
  --extractor-chat-template-kwargs '{"enable_thinking":false}' \
  --reasoning-mode cot \
  --condition state_only \
  --output "${STATE_DIR}/state_only_submission.jsonl" \
  --diagnostics "${STATE_DIR}/state_only_diagnostics.jsonl" \
  --concurrency 16 \
  --resume
```

For a two-by-two module swap, repeat state extraction with a second VLM, then
run each saved state file through both reasoner checkpoints. Name outputs
`states-<perception-model>__reasoner-<reasoner-model>.jsonl`. Off-diagonal cells
are essential: they distinguish state quality from reasoning quality.

## 7. Run the matched curriculum versus recognition-control training

This uses a separate training environment because the official Qwen trainer
pins a different PyTorch stack from vLLM.

The official source describes an unreleased `transformers 4.57.0.dev0`; the
setup script uses the published compatible patch release `4.57.1`. It requires
Python 3.12, `nvcc`, at least 80 GB of free disk, and BF16-capable GPUs.

```bash
bash evaluation/research/setup_qwen3vl_training.sh
```

Smoke-test one condition for two optimizer steps in a disposable output root:

```bash
GPU_IDS=0,1 SEED=20260723 MAX_STEPS=2 SKIP_SETUP=1 \
OUTPUT_ROOT=/tmp/ms-vista-lora-smoke \
  bash evaluation/research/train_perception_lora.sh perception_curriculum
```

Run both conditions at three independent seeds. Each run uses two A100s,
unquantized BF16 weights, official Qwen3-VL training code, and LoRA:

```bash
for seed in 20260723 20260724 20260725; do
  GPU_IDS=0,1 SEED="${seed}" SKIP_SETUP=1 \
    bash evaluation/research/train_perception_lora.sh perception_curriculum
  GPU_IDS=0,1 SEED="${seed}" SKIP_SETUP=1 \
    bash evaluation/research/train_perception_lora.sh recognition_control
done
```

The adapters are written under:

```text
evaluation/research/results/perception_lora/seed-<seed>/<condition>
```

The official LoRA branch trains attention `q_proj`, `k_proj`, `v_proj`, and
`o_proj` adapters and freezes the vision encoder. The causal comparison is
therefore perception-focused supervision versus matched recognition
supervision, not a claim that the vision encoder itself was retrained.

First evaluate base, curriculum, and control on the fixed 560-item native
fidelity subsets. If the curriculum-control difference is consistent over all
three seeds, run the full two benchmarks as confirmation. Serve an adapter with
vLLM using:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/visual-suite/bin/vllm serve \
  Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules perception=PATH_TO_ADAPTER \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

Set `MODEL=perception` when running the condition matrix against that adapter.
Record the base checkpoint, adapter path, training seed, inference seed, and
generated-data manifest hash in the paper artifacts.

## 8. Minimum paper-ready execution order

1. Run causal controls on at least three representative model families.
2. Run fidelity and abstraction cues on the same models.
3. Run the two-by-two module swap on one strong and one contrasting model.
4. Run three curriculum and three recognition-control training seeds.
5. Confirm any transfer effect on the full benchmarks.
6. Freeze the analysis before inspecting task-level results.

Do not interpret a single model, a single LoRA seed, or an item oracle as a
submission-ready causal finding.
