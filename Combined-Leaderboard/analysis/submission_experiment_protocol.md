# Minimal experiment protocol for final submission

## Canonical ranking freeze

The ranking input is the immutable
`evaluation/results/ms-vista-ranking-final-evidence-v4-v15` tree. All model
variants must use the same pinned Qwen3-8B v4 evidence contract. The extractor
receives the question, answer contract, response metadata, and candidate
response. It receives no image and no ground truth.

Persistent extractor schema failures are fail-closed as unresolved. Committed
answers outside the task answer domain are retained as invalid commitments.
Neither outcome may be repaired with the reference answer.

Reference verification:

```bash
BUNDLE=evaluation/results/ms-vista-ranking-final-evidence-v4-v15

(cd "$BUNDLE" && shasum -a 256 -c SHA256SUMS)

.venv/bin/python -m evaluation.finalize_visual_results \
  --output-root "$BUNDLE" \
  --verify-only

PYTHONPATH="$PWD:$PWD/backend" \
  GROUND_TRUTHS_DIR="$PWD/Ground_truths" \
  .venv/bin/python scripts/import_canonical_visual_results.py \
  --result-root "$BUNDLE"
```

The importer is intentionally a dry run unless `--apply` is supplied.

Acceptance criteria:

* Every expected model and question is present exactly once.
* Every retained raw response matches the response hash in its audit row.
* The audit uses one extractor model, revision, and prompt contract.
* Ground truth and images are absent from the extractor process.
* There are no blocking extractor statuses.
* All canonical answers are valid answers, `__INVALID_FORMAT__`, or
  `UNRESOLVED`.

## Experiment 2: Qwen3.5 thinking replication

Purpose: estimate decoding-seed variance for the only controlled inference-mode
comparison.

Run both thinking modes with two additional base seeds, recommended `11` and
`29`. Keep every other field identical to the existing run:

* Same Qwen3.5-9B repository and pinned revision.
* Unquantized BF16 checkpoint and BF16 KV cache.
* Same original image bytes and question order.
* Same prompts, temperatures, top-p values, stop behavior, and token limits.
* Separate result roots for every mode and seed.
* Final v4 evidence audit with the same extractor contract.

Primary endpoint: the interaction
`(thinking minus nonthinking perception) minus (thinking minus nonthinking cognition)`.

Secondary endpoints: benchmark deltas and task deltas. Apply false-discovery
correction separately within each benchmark. Report the distribution across
seeds and do not select the best seed.

Acceptance criteria:

* The interaction has the same sign in all three seeds.
* The seed-aggregated interval excludes zero for a headline claim.
* The result remains after excluding length-capped responses as a sensitivity
  analysis.

The current shell runner exposes Qwen3.5 with thinking disabled only. Add an
explicit, provenance-recorded thinking-mode option before launching this
replication; do not rely on a model default that is absent from the run config.

## Optional experiments

1. Add one more same-family size pair before claiming a general scaling law.
2. Evaluate humans on a stratified subset of the exact pinned items before
   making an inferential human-gap claim.
3. Add more independent model families before testing specific cross-task
   mechanisms again. With nine family-level units and 56 comparisons, the
   current study is underpowered for that question.

## Analysis rerun

After constructing a refreshed ranking bundle, rerun:

```bash
.venv/bin/python analysis/combined_visual_v14.py \
  --bundle /path/to/refreshed-ranking-bundle \
  --output analysis/results/combined_visual_final

.venv/bin/python analysis/submission_readiness_v14.py \
  --bundle /path/to/refreshed-ranking-bundle \
  --exploratory-output analysis/results/combined_visual_final \
  --output analysis/results/submission_readiness_final
```

The submission should use only the refreshed outputs and should archive the
analysis seed, source hashes, extractor contract hash, and model run manifests.
