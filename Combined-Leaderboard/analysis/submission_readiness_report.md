# MS-VISTA Combined Visual Evaluation: Submission Readiness

## Decision

The canonical v15 result set is ready for the production leaderboard. It is
also ready to support a focused paper contribution, provided the claims below
are separated into ready, replication-required, descriptive, and unsupported
categories.

The result tree uses one pinned gold-blind v4 extraction contract across all
74,186 responses. Canonical verification, production scoring, raw-response
hash checks, and the checksum inventory all pass.

## Evidence Base

| Property | Value |
| --- | ---: |
| Model variants | 14 |
| Model families | 9 |
| Do You See Me items per variant | 4,500 |
| Mind's Eye items per variant | 799 |
| Total responses | 74,186 |
| Shared item bootstraps | 10,000 |
| Split-half replications | 5,000 |
| Controlled-pair bootstraps | 20,000 |
| Cross-task permutations | 100,000 |

Every score is recomputed with the backend production `TaskScorer`. The
expanded benchmark set is not identical to either source paper's test set, so
paper human values must not be used as matched inferential baselines.

## Findings Ready for Submission

### 1. Shared visual competence coexists with layer specialization

Perception and cognition correlate at Pearson `r = 0.751`, with a 95% interval
of `[0.498, 0.906]`, and Spearman `rho = 0.736`. Pearson remains between
`0.725` and `0.802` across all one-variant-per-family cohorts.

VPCI is highly stable to item sampling: median split-half Spearman is `0.982`,
rank one agrees in every replicate, and median top-three overlap is complete.
The first two task-profile components explain `33.90%` and `23.14%` of
variance, supporting both shared competence and specialization.

Suggested claim:

> Combined evaluation reveals a stable shared visual competence structure,
> while layer-specific rank reversals show that perception and cognition remain
> empirically distinct.

### 2. The frontier is robust to aggregation choice

Geometric and harmonic VPCI preserve the top six and differ from arithmetic
VPCI only by one adjacent middle-tier swap (`rho = 0.996`). The bottleneck
ranking has `rho = 0.982`. Qwen3.6-27B and GLM-4.6V-Flash form the two-model
Pareto frontier.

Suggested claim:

> The combined frontier and broad ordering are not artifacts of one averaging
> rule, although layer scores remain necessary for interpreting specialists.

### 3. Complexity exposes a persistent perception hard tail

Mean perception accuracy falls from `37.51%` on easy items to `17.88%` on hard
items. The `19.63` point gap remains `17.23` points after removing form
discrimination. Removing all objective floor strata preserves the top VPCI
model and yields rank correlation `0.996`.

Suggested claim:

> Increasing visual complexity produces a hard-tail deficit that remains after
> removing low-information form-discrimination strata.

### 4. Cognition bottlenecks survive chance normalization

Chance normalization changes the cognition ordering by at most one rank
(`rho = 0.996`) and keeps Qwen3.6-27B first. Dynamic reasoning, mental
rotation, and spatial visualization remain below chance at the cohort mean.

Suggested claim:

> The cognition bottleneck is not explained by unequal option counts; several
> transformation tasks remain at or below chance across the evaluated cohort.

## Strong Candidate Requiring Independent Confirmation

### Perception supports task-level model routing

Out-of-fold task routing improves perception macro accuracy by `11.02` points
on average and is positive in all 100 repeated splits. It does not improve
cognition, where the mean change is `-0.51` points.

The repeated folds measure split sensitivity on the same item population, not
independent experimental replication. Confirm this result on a fresh pinned
item generation before using it as a headline contribution.

Provisional claim:

> Perception errors are sufficiently task-structured to support held-out
> model routing, while cognition performance remains dominated by the best
> global model.

## Finding Requiring a New GPU Run

### Thinking-enabled Qwen3.5 is confounded by output exhaustion

Thinking enabled is 4.03 perception points and 9.86 cognition points below the
same revision with thinking disabled. The interaction is statistically clear,
but the thinking run reaches its token limit on `31.4%` of perception items and
`55.2%` of cognition items.

This does not isolate the causal effect of reasoning. Run at least two
additional seeds per mode with a reserved final-answer budget before presenting
the comparison as capability evidence.

Defensible current statement:

> Under a shared finite output budget, unrestricted reasoning can reduce final
> answer commitment and therefore lower measured benchmark performance.

## Descriptive Case Study

Gemma 3 27B improves over Gemma 3 12B by 2.07 perception points and 4.49
cognition points. Both intervals exclude zero, but the cross-layer interaction
does not. With only one same-family size pair, this is not a general scaling
law.

## Claim to Reject

Do not claim that a specific perception skill enables a specific cognition
skill. None of the 56 cross-layer task associations survives false-discovery
correction after controlling for competence from the other tasks, including
after averaging variants within family.

## Required Disclosure

The strict v4 protocol produces 50,960 evidence-supported commitments, 15,353
explicit invalid-format commitments, and 7,873 unresolved, truncated, or
unsupported responses. These outcomes are scored as wrong.

Consequently, reported scores measure visual performance together with the
ability to produce an auditable final commitment. This is appropriate for an
automatic public leaderboard, but it is not a pure estimate of latent model
knowledge. The large unresolved share in the thinking-enabled run must be
reported wherever that comparison appears.

## Remaining Experiments

1. Repeat Qwen3.5 thinking enabled and disabled with at least two additional
   seeds and a protected final-answer budget.
2. Confirm task routing on a fresh independently generated item set.
3. Evaluate a second same-family size pair before making scaling claims.
4. Collect matched human responses only if the paper will make inferential
   human-versus-model claims.
5. Add the spatial-reasoning track when its evaluation and evidence pipeline is
   complete.

## Recommended Contribution

The strongest current paper contribution is a layered evaluation result:

1. A stable shared competence structure spans perception and cognition.
2. Layer specialization remains substantial enough that one score is
   insufficient by itself.
3. The combined frontier is robust to item sampling and common aggregation
   choices.
4. Complexity, shared blind spots, and near-chance transformation tasks remain
   unresolved at the current model frontier.
5. Perception errors show exploitable model complementarity, pending
   independent item-set confirmation.

Exact outputs are in `analysis/results/combined_visual_v15`,
`analysis/results/submission_readiness_v15`, and
`analysis/results/research_experiments/complementarity`.
