# External Track 3 v2 review

Review and conversion date: 2026-09-15. The three models are packaged for a
separate self-reported cohort, not the installed paper-aligned release.

Input: `track3_submissions_v2.zip`, SHA-256
`f324880b4e621eb982b09115ccdfc57393f1af9e3a3997c9ca31ed69910ba37f`.
The original archive remains private and is not part of the repository.
Its embedded validator was read, not executed. No GPU or model call was used.

Models: GPT-5 (`gpt-5_2025-08-07`), GPT-4o (`gpt-4o_2024-11-20`),
Gemma-4-31B-it (`google/gemma-4-31B-it`). All declare `spatial-cot-eval-v2`
and Qwen3-30B-A3B-Instruct-2507 as the judge. That is submitter-declared
provenance, not proof of the judge/revision that ran.

## Checks

All three models contain six conditions, 13 datasets, unique prediction and
verdict keys, matching pair metadata, and no ground-truth fields in prediction
rows. All per-dataset scores, macro-average rows, and main-condition deltas
match aggregation of the **submitted** verdicts within 0.02 percentage points.
This checks arithmetic only, not independent semantic correctness.

They cannot be imported into the installed `paper-aligned-v5-2026-07` release:

| Condition (each prompt mode) | Supplied rows | Required live rows | Missing exact public IDs | Unknown IDs |
| --- | ---: | ---: | ---: | ---: |
| Main | 15,370 | 13,951 | 3,709 | 5,128 |
| No-image | 15,370 | 13,539 | 3,968 | 5,799 |
| No-image-plus | 14,672 | 12,923 | 3,665 | 5,414 |

Each model has 90,824 rows, versus 80,826 in the live release. Exact comparison
qualifies each supplied index with its dataset name; it never guesses a mapping
from numerical indices to release IDs. The 3DSRBench and MindCube ID schemes
differ. SAT-Real and SpatialBench also differ in condition coverage. Main and
no-image each have 327 overlapping RealWorldQA IDs with a different answer type.
This is a benchmark-contract mismatch, not merely ZIP naming.

Gemma additionally has 243 null model outputs: 130 main-CoT, 112 no-image-plus
direct, and one no-image-plus CoT. GPT-5 has three empty-string outputs. None
receives positive submitted credit. The canonical evidence retains these exact
null/empty values, marks `output_status=missing_source_output`, and uses the
administrative `MISSING_SOURCE_OUTPUT` final-answer marker with zero credit.
The marker is not a model response; the failure cause is unknown.

## Comparison boundary

Obtain exports for the installed public release, with exact public IDs, groups,
answer types and conditions, using its canonical artifact harness. Mapping IDs
alone does not resolve differing sample scopes/types. The experiment owner must
confirm whether existing outputs can faithfully support that export or whether
some conditions need rerunning. Do not assign the current manifest hash to a
different experiment or silently change reported scores.

The requested addition instead uses a separate, frozen submitted sample catalog:
`spatial-cot-eval-v2-submitted-2026-08-12`. Its manifest SHA-256 is
`4230f7d68d778d885189f1ebb7d5342bf7e1525000c7c19d701073f05650e0bf`.
All three models have identical sample IDs, conditions, group IDs, and answer
types. Cohort selection is bound to this digest, not a display label. There is
no all-cohorts ranking. InternVL remains in the default paper-aligned cohort.

## Canonical packaging

The converter uses the existing `ms-vista-track3-artifact-package/v1` package, score
aggregation, checksums, and gzip evidence implementation. It does not fabricate
an official dataset, prompt, judge, or model-revision attestation. The separate
catalog declares `official_protocol_attested=false` and
`provenance_status=submitter_declared_unattested`.

Public packages exclude source contact/team metadata, answer keys, and options.
Only model ID, run date, decoding configuration, judge name, and kit version are
copied from source metadata. MCQ final answers retain the submitted judge's
extracted label. VQA final answers retain the actual model output, not the
judge's correctness verdict. Original output and submitted judge output are
retained separately. Source score arithmetic is reproduced from claimed credit;
this is not fresh answer extraction or independent correctness verification.

| Model | Evidence rows | Missing outputs | Package SHA-256 |
| --- | ---: | ---: | --- |
| GPT-5 (2025-08-07) | 90,824 | 3 | `5d649b29222df7c1f730e0715cfd12631fd75ebdb35ccdd36087c900c4491d8e` |
| GPT-4o (2024-11-20) | 90,824 | 0 | `729caa3f1040146d5283cb4f09e6f28c20f51f1aad91f94910b55435f2e8b9ad` |
| Gemma-4-31B-it | 90,824 | 243 | `8d34c137cf038236d14f4a7db7244631a8f069520279b25a903f37d770451505` |

Normal public uploads remain pinned to the installed official contract. Only
the trusted administrative importer accepts `--submitted-cohort` with an
explicit reviewed contract directory. The database retains that exact contract
for later evidence revalidation. Public upload size/resource limits are not
increased for this import.

```sh
python3 scripts/prepare_track3_v2.py /private/track3_submissions_v2.zip \
  --output /private/reviewed-v2
python3 scripts/import_spatial_results.py --submitted-cohort \
  --contract-dir /private/reviewed-v2/contract \
  --package-root /private/reviewed-v2
# Apply only after review, a validated backup, and the deployment maintenance lock.
# Add --apply; import one --package at a time on memory-constrained machines.
```

The private rehearsal preserved all 47 prior submission fingerprints and added
three submissions, producing 50 submissions, 475,175 answer rows, 20 artifacts,
and two frozen contracts. Retrying every package produced no duplicates. All
answer/artifact hashes and cache/database score fingerprints validated. No
authentication records were copied into this rehearsal.

The API accepts `GET /api/leaderboard/spatial?cohort=<manifest-sha256>` and returns
the selected cohort plus available cohorts. Omitting the filter preserves the
paper-aligned default; unknown cohorts return HTTP 400. The frontend applies the
same selection to rankings and integrated comparison, clears stale results on
failure, and ignores obsolete in-flight responses.

Reproduce the review (exit 2 means blocked from canonical repackaging):

```sh
python3 scripts/review_track3_v2.py /private/track3_submissions_v2.zip
```
