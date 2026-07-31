# Prespecified Qwen3.5 Thinking Ablation

Frozen before any condition in this experiment is scored.

Analysis ID: `ms-vista-qwen35-thinking-factorial-v1`

## Research question

The leaderboard contains an unexpectedly weak Mind's Eye result for
Qwen3.5-9B with internal thinking disabled. This experiment separates two
mechanisms that were previously coupled:

1. Whether the benchmark prompt explicitly asks for reasoning.
2. Whether the checkpoint's chat template enables internal thinking.

The result is a paired 2 by 2 factorial design. It tests whether the weak
result is attributable to the external prompt, the internal thinking mode, or
an interaction between them.

## Fixed model and inference contract

- Target model: `Qwen/Qwen3.5-9B`
- Target revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Track: all 799 Mind's Eye questions
- Weights: original unquantized BF16 checkpoint tensors
- Context: 32,768 tokens
- Completion budget: remaining context, with no artificial output cap
- Decoding: temperature 0, top-p 1, seed 0
- Image preprocessing: original benchmark image bytes
- Answer extraction: pinned gold-blind `Qwen/Qwen3-8B` revision
  `b968826d9c46dd6066d109eabc6255188de91218`

## Conditions

| Condition | Benchmark prompt | Internal thinking |
| --- | --- | --- |
| `direct_disabled` | Direct answer | Disabled |
| `direct_enabled` | Direct answer | Enabled |
| `cot_disabled` | Reasoning prompt | Disabled |
| `cot_enabled` | Reasoning prompt | Enabled |

Every condition uses the same question IDs, images, checkpoint, extraction
contract, and decoding seed.

## Confirmatory tests

Five paired contrasts form one family:

1. Internal-thinking effect with the direct prompt.
2. Internal-thinking effect with the reasoning prompt.
3. Reasoning-prompt effect with internal thinking disabled.
4. Reasoning-prompt effect with internal thinking enabled.
5. Prompt-by-thinking interaction:
   `(cot_enabled - direct_enabled) - (cot_disabled - direct_disabled)`.

Accuracy uses the production Mind's Eye macro: equal weight for each cognitive
capability, then equal weight for items within capability. Confidence intervals
use 10,000 paired, capability-stratified bootstrap replicates. Two-sided
p-values use 10,000 paired sign-randomization replicates under the same macro
weights. One Holm correction covers all five tests at alpha 0.05.

## Quality gates

Every condition must contain exactly the same 799 unique question IDs. Raw
responses must be nonempty and free of inference errors. Both `stop` and
`length` finish reasons are retained as model outcomes. The fixed extractor
must resolve or explicitly mark every answer unresolved, and its model,
revision, method, and source-response hash must match the contract.

Unresolved and length-capped answers count as incorrect. Infrastructure,
coverage, extractor, or provenance failures must be rerun and cannot be
converted into model errors.

## Interpretation limits

This is a within-checkpoint protocol ablation, not a comparison of model
families or training methods. A significant interaction identifies sensitivity
to how reasoning is elicited; it does not establish why the checkpoint learned
that behavior. Task-level effects are exploratory and receive a separate Holm
correction within each confirmatory contrast.
