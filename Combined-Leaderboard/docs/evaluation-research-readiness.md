# Evaluation research readiness

## Current verdict

The evaluation and postprocessing implementation is mechanically verified, but
the locally recovered answers are **not yet approved for research promotion**.
They remain in the review-only staging tree until the blinded human audit passes.

This distinction is intentional:

- The canonical evaluated-model outputs are immutable and hash verified.
- Local answer extraction cannot access benchmark images or ground truth.
- Private deterministic scoring runs only after extracted answers are locked.
- Ground-truth correctness never decides whether an extraction is accepted.
- Independent human review of the actual malformed-response population is still
  required before those recoveries can affect published leaderboard scores.

## Evidence checked

The current extraction run contains 5,137 candidate responses across 12 models
and two benchmark tracks. It recovered 1,714 answers and left 3,423 responses as
`__INVALID_FORMAT__`. The staging verifier covered all candidate rows and all 24
model-track outputs without modifying the canonical source tree.

The extractor passed its 256-row parseable-response validation with 243 faithful
resolutions, zero incorrect accepted resolutions, and 13 abstentions. This proves
the basic parser and model contract, but it does not by itself prove performance
on malformed responses. That distribution gap is why the human audit is a hard
release gate.

The private scoring audit covered 63,588 model responses. The staged recovery
changed 1,839 answers, producing 490 newly correct answers, 45 stricter-protocol
regressions, and 1,304 answers that remained incorrect. These scores are an
outcome audit only. They must not be used to rescue, reject, or tune individual
extractions.

## Blinded release gate

Run:

```bash
.venv/bin/python evaluation/human_audit_local_extractor.py prepare
```

The current package contains:

- 350 probability-sampled accepted recoveries.
- A census of all 80 recoveries flagged for truncation, long output, uncertainty
  language, prior legacy extraction, or a nonstandard extractor finish.
- Coverage of all 32 model, benchmark-track, and task strata.
- 441 unique review items after overlap between those groups.

Two independent reviewers judge only whether each extracted answer faithfully
represents a single final commitment in the raw response. They receive no image,
model identity, score, or ground truth. Modified review evidence fails its content
hash check.

Research promotion requires all of the following:

1. Every selected item is independently reviewed.
2. All disagreements and `unclear` labels are adjudicated.
3. Cohen's kappa is at least 0.80.
4. No selected or high-risk item is confirmed as unsupported or ambiguous.
5. The probability sample has zero false accepts. For 350 reviewed rows, this
   gives a one-sided 95% upper false-accept bound of 0.852%.
6. Canonical, staging, extraction-report, question, and raw-response hashes still
   match before assessment.

If the gate fails, version the extraction policy, regenerate staging from the
original canonical outputs, and draw a fresh audit sample. Never repair sampled
answers manually or expose the extractor to the answer key.

## Permitted use now

The canonical strict-parser results are suitable for internal analysis with their
documented `__INVALID_FORMAT__` treatment. The locally recovered staging scores
are suitable for method development and audit only. They should not be cited as
final benchmark results until `assessment.json` reports
`approved_for_research_promotion: true`.
