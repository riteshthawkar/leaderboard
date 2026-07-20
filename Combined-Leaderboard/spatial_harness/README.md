# Track-3 v2 Spatial Reasoning Evaluation

This package evaluates the 13 spatial benchmarks from the Spatial-CoT analysis
with an OpenAI-compatible vision endpoint and a separate judge endpoint. It
implements the v2 protocol from `TRACK3-v1-to-v2-changes.md`.

## V2 contract

- Explicit datasets: `BLINK,CV-Bench-2D,CV-Bench-3D,MMVP,RealWorldQA,VStarBench,MMSIBench_wo_circular,3DSRBench,VSR_MCQ,SpatialBench,MindCube,OmniSpatial,SAT-Real`.
- Conditions: `main` and No-Image++ (`noimgpp`), each with `noncot` and `cot` prompts.
- Equal 16,384-token completion budgets for CoT and non-CoT.
- `answer_type=mcq` uses the paper's Appendix A.3 MCQ judge.
- `answer_type=vqa` uses the paper's Appendix A.3 VQA judge and runs only in `main`.
- SpatialBench and SAT-Real use all-rotations-correct circular scoring.
- No-Image++ runs on every valid MCQ item. There is no frozen probe subset.

## Installed environment

The dedicated environment on this host is:

```bash
conda activate /share/data/drive_3/conda_envs/track3-v2
```

VLMEvalKit is installed editable from
`/share/data/drive_3/vendor/VLMEvalKit-7055d301` at commit
`7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f`. The environment uses CPU PyTorch
because model and judge generation happen in external endpoints; dataset
preparation and judging do not need a second CUDA runtime.

For a fresh Linux x86-64 environment, use the exact locks:

```bash
./spatial_harness/install_track3_env.sh
```

`environment-track3-v2.yml` is the small human-readable Conda specification;
the two lock files record the exact installed Linux package set. The installer
uses the Conda lock for NumPy and the Python 3.11 build of decord, the pip lock for
pip-managed packages, and a clean editable VLMEvalKit checkout whose commit is
verified before installation.

## Prepare data

Accept the SpatialBench terms at
<https://huggingface.co/datasets/RussRobin/SpatialBench> first. Then run from
the `Combined-Leaderboard` root:

```bash
export HF_TOKEN=hf_xxx
/share/data/drive_3/conda_envs/track3-v2/bin/python \
  -m spatial_harness.prepare_data \
  --lmudata /share/data/drive_3/track3-v2/LMUData \
  --cache /share/data/drive_3/track3-v2/cache
```

Preparation pins custom source revisions, writes SHA-256 provenance to
`track3_data_manifest.json`, verifies unique indices and answer labels, and
rejects incorrect v2 counts. The shared bundle contains all 13 datasets under
`/share/data/drive_3/track3-v2/LMUData` and passes full manifest verification.

| Dataset | Rows | V2 detail |
| --- | ---: | --- |
| MMVP | 300 | Official pinned MMVP source; MCQ |
| RealWorldQA | 765 | 438 MCQ and 327 VQA |
| 3DSRBench | 5,157 | Flip-augmented source; runner selects 2,625 base questions |
| MindCube | 1,040 | Paper's 1K MCQ split; path-backed assets pinned separately |
| SpatialBench | 174 | Mixed MCQ/VQA; parser repairs two malformed positional items |
| OmniSpatial | 1,533 | 1,304 MCQ and 229 VQA |
| SAT-Real | 150 | MCQ; multiple images retained |

SpatialBench remains gated for fresh installations. Preparation uses standard
HTTPS for that source and stops with an actionable error until the Hugging Face
account associated with `HF_TOKEN` has accepted its terms.

## Run evaluation

The vision server and judge server must expose `/v1/models` and
`/v1/chat/completions`. Run:

```bash
./spatial_harness/run_eval.sh \
  MODEL_NAME \
  http://127.0.0.1:8000/v1 \
  http://127.0.0.1:9000/v1 \
  JUDGE_MODEL_NAME
```

The fourth argument is optional and defaults to `JUDGE_MODEL`, then to the VLM
model name. `VLM_ENDPOINTS` may contain comma-separated replicas. Useful
overrides include `OUT`, `LMUDATA`, `VLM_CONCURRENCY`, `JUDGE_CONCURRENCY`,
`VLM_API_KEY`, and `JUDGE_API_KEY`.

The runner validates model identity at every endpoint, checkpoints predictions
atomically, resumes completed rows, and records every input TSV hash. The judge
also resumes atomically. Final metrics are written to `leaderboard.json` with
`main_noncot`, `main_cot`, `main_delta`, `npp_noncot`, and `npp_cot` for each
dataset.

Monitor a running inference directory with:

```bash
/share/data/drive_3/conda_envs/track3-v2/bin/python \
  -m spatial_harness.watch_progress \
  /share/data/drive_3/track3-v2/results/MODEL_SLUG
```

## Validation

```bash
/share/data/drive_3/conda_envs/track3-v2/bin/python -m pytest -q tests/spatial
/share/data/drive_3/conda_envs/track3-v2/bin/python \
  -m spatial_harness.prepare_data --verify-only \
  --lmudata /share/data/drive_3/track3-v2/LMUData
bash -n spatial_harness/run_eval.sh
```