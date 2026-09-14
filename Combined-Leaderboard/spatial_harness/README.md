# Spatial Reasoning Harness

This is the only supported MS-VISTA Track 3 harness. It prepares the 13
paper-aligned spatial datasets, runs all six frozen conditions, judges final
answers with the pinned Qwen judge, and creates the single ZIP accepted by the
leaderboard.

The harness does not redistribute benchmark images or questions. It downloads
the upstream datasets into a user-controlled work directory and records their
immutable hashes in `track3_data_manifest.json`.

## Requirements

- Linux with Conda and Git.
- An OpenAI-compatible endpoint serving the evaluated vision-language model.
- An OpenAI-compatible endpoint serving
  `Qwen/Qwen3-30B-A3B-Instruct-2507` at revision
  `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- A Hugging Face token with access to every gated dataset used by the release.

By default, environments, datasets, contracts, and outputs live under
`${XDG_DATA_HOME:-$HOME/.local/share}/ms-vista-track3`. Set `TRACK3_ROOT` to
place them on a larger volume.

## Install

```bash
cd spatial_harness
TRACK3_ROOT=/path/to/ms-vista-track3 ./install_track3_env.sh
```

The installer creates an isolated Conda prefix from the checked-in lock file
and checks out the pinned VLMEvalKit commit. Existing checkouts must be clean
and at the exact required commit.

## Prepare Data

```bash
export TRACK3_ROOT=/path/to/ms-vista-track3
export HF_TOKEN=<read-token>

"$TRACK3_ROOT/conda-env/bin/python" -m spatial_harness.prepare_data \
  --lmudata "$TRACK3_ROOT/LMUData" \
  --cache "$TRACK3_ROOT/cache" \
  --hf-token "$HF_TOKEN"
```

Re-run with `--verify-only` before every official evaluation. Any changed
dataset count or hash is a release-contract change and must not be accepted
silently.

## Evaluate

```bash
TRACK3_ROOT=/path/to/ms-vista-track3 \
VLM_API_KEY=<model-endpoint-key> \
JUDGE_API_KEY=<judge-endpoint-key> \
./spatial_harness/run_eval.sh \
  "Organization/Model-Name" \
  "<full-model-commit-sha>" \
  "http://model-host:8000/v1,http://model-host:8001/v1" \
  "http://judge-host:8100/v1"
```

The command verifies the data, rebuilds the public contract, executes all six
conditions, judges every completed response, and packages:

```text
<TRACK3_ROOT>/results/<model>/submission_package/
  track3_artifact_submission.zip
```

Upload that ZIP unchanged. It contains submitter-claimed per-sample credit,
aggregate integer counts, compact final answers, compressed original model
responses, and SHA-256 checksums. The leaderboard checks package integrity,
public sample coverage, and score arithmetic; it does not compare answers with
reference answers. Debug runs, partial dataset runs, edited packages, unjudged
rows, incompatible revisions, and provenance mismatches are rejected.
Explicit terminal context-limit failures use the versioned failure policy and
are scored incorrect without asking the judge to invent an answer.

## Release Contract

Administrators publish a contract with:

```bash
python -m spatial_harness.build_public_contract \
  --lmudata /path/to/LMUData \
  --output /path/to/public-contract \
  --private-ground-truth /secure/path/ground_truth.json \
  --benchmark-version <version>
```

Only the public `manifest.json`, `questions.jsonl`, and
`submission_template.jsonl` belong on the leaderboard server. The private
ground-truth artifact is an administrator audit artifact and must remain
outside source control and deployment images.
