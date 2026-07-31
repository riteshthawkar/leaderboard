#!/usr/bin/env python3
"""Score a downloaded GPU analysis queue and synthesize corrected findings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.combined_visual_v14 import score_from_correctness  # noqa: E402
from analysis.research.score_condition_matrix import (  # noqa: E402
    benchmark_weights,
    bootstrap_delta,
    paired_bootstrap_delta,
    score_conditions,
)
from evaluation.research.common import describe_path, write_json  # noqa: E402


MODEL_NAMES = {
    "qwen3-vl-8b": "Qwen3-VL-8B-Instruct",
    "internvl35-8b": "InternVL3.5-8B",
}
PRESPECIFIED_MODELS = ("qwen3-vl-8b", "internvl35-8b")
ANALYSIS_PLAN = (
    PROJECT_ROOT / "evaluation/research/PRESPECIFIED_ANALYSIS_PLAN.md"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(
        sorted(range(count), key=lambda item: p_values[item])
    ):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def require_quality_gates(model_root: Path) -> None:
    summaries = sorted(model_root.glob("**/quality/summary.json"))
    if len(summaries) != 11:
        raise ValueError(
            f"{model_root}: expected 11 quality summaries, found {len(summaries)}"
        )
    failed = [
        path
        for path in summaries
        if not read_json(path).get("quality_gate_passed")
    ]
    if failed:
        raise ValueError(
            f"{model_root}: quality gates failed for "
            + ", ".join(str(path.parent.parent.relative_to(model_root)) for path in failed)
        )


def condition_paths(
    model_root: Path,
    prefix: str,
    conditions: list[str],
    prompt_mode: str,
) -> list[tuple[str, Path]]:
    return [
        (
            condition,
            model_root
            / prefix
            / condition
            / prompt_mode
            / "submission.jsonl",
        )
        for condition in conditions
    ]


def find_comparison(summary: dict[str, Any], condition: str) -> dict[str, Any]:
    for row in summary["comparisons"]:
        if row["condition"] == condition:
            return row
    raise ValueError(f"Comparison for {condition} is missing")


def add_condition_endpoint(
    rows: list[dict[str, Any]],
    *,
    model_slug: str,
    family: str,
    endpoint: str,
    comparison: dict[str, Any],
) -> None:
    rows.append(
        {
            "model_slug": model_slug,
            "model": MODEL_NAMES.get(model_slug, model_slug),
            "comparison_type": "within_model",
            "contrast": "condition minus baseline",
            "family": family,
            "endpoint": endpoint,
            "effect": float(comparison["macro_delta"]),
            "interval_low": float(
                comparison["stratified_bootstrap_95_interval"][0]
            ),
            "interval_high": float(
                comparison["stratified_bootstrap_95_interval"][1]
            ),
            "raw_p": float(comparison["paired_randomization_p"]),
        }
    )


def read_csv_indexed(path: Path, key: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = str(row.get(key) or "").strip()
            if not identifier:
                raise ValueError(f"{path}: row is missing {key}")
            if identifier in rows:
                raise ValueError(f"{path}: duplicate {key} {identifier}")
            rows[identifier] = row
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def metadata_from_item_rows(
    *, track: str, item_ids: list[str], rows: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    metadata: list[dict[str, str]] = []
    for question_id in item_ids:
        subgroup = rows[question_id]["subgroup"]
        if track == "do_you_see_me":
            dimension, separator, capability = subgroup.partition(":")
            if not separator:
                raise ValueError(
                    f"{question_id}: invalid perception subgroup {subgroup!r}"
                )
            metadata.append(
                {"dimension": dimension, "capability": capability}
            )
        else:
            metadata.append({"capability": subgroup})
    return metadata


def sign_flip_p(
    contrast: np.ndarray,
    weights: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> float:
    observed = abs(float(np.dot(weights, contrast)))
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    batch_size = 500
    while completed < replicates:
        count = min(batch_size, replicates - completed)
        signs = rng.integers(0, 2, size=(count, len(contrast))) * 2 - 1
        permuted = np.abs(signs @ (weights * contrast))
        extreme += int(np.sum(permuted >= observed - 1e-12))
        completed += count
    return (extreme + 1) / (replicates + 1)


def interaction_statistics(
    *,
    contrast: np.ndarray,
    track: str | None,
    metadata: list[dict[str, str]],
    bootstrap_replicates: int,
    randomization_replicates: int,
    seed: int,
) -> dict[str, Any]:
    if track is None:
        effect = float(contrast.mean())
        interval = paired_bootstrap_delta(
            contrast,
            np.zeros_like(contrast),
            replicates=bootstrap_replicates,
            seed=seed,
        )
    else:
        effect = score_from_correctness(track, contrast, metadata)
        interval = bootstrap_delta(
            track=track,
            candidate=contrast,
            baseline=np.zeros_like(contrast),
            metadata=metadata,
            replicates=bootstrap_replicates,
            seed=seed,
        )
    weights = (
        np.full(len(metadata), 1.0 / len(metadata), dtype=float)
        if track is None
        else benchmark_weights(track, metadata)
    )
    return {
        "effect": effect,
        "interval_low": float(interval[0]),
        "interval_high": float(interval[1]),
        "raw_p": sign_flip_p(
            contrast,
            weights,
            replicates=randomization_replicates,
            seed=seed + 1,
        ),
    }


def condition_interaction(
    *,
    first_path: Path,
    second_path: Path,
    first_model: str,
    second_model: str,
    track: str,
    baseline: str,
    condition: str,
    family: str,
    endpoint: str,
    bootstrap_replicates: int,
    randomization_replicates: int,
    seed: int,
) -> dict[str, Any]:
    first = read_csv_indexed(first_path, "question_id")
    second = read_csv_indexed(second_path, "question_id")
    if set(first) != set(second):
        raise ValueError(
            f"Cross-model interaction is not paired: {first_path} vs {second_path}"
        )
    item_ids = sorted(first)
    metadata = metadata_from_item_rows(
        track=track, item_ids=item_ids, rows=first
    )
    for question_id in item_ids:
        if first[question_id]["subgroup"] != second[question_id]["subgroup"]:
            raise ValueError(
                f"{question_id}: cross-model subgroup metadata differs"
            )
    contrast = np.asarray(
        [
            (
                int(first[question_id][condition])
                - int(first[question_id][baseline])
            )
            - (
                int(second[question_id][condition])
                - int(second[question_id][baseline])
            )
            for question_id in item_ids
        ],
        dtype=float,
    )
    return {
        "model_slug": f"{first_model}__vs__{second_model}",
        "model": (
            f"{MODEL_NAMES.get(first_model, first_model)} minus "
            f"{MODEL_NAMES.get(second_model, second_model)}"
        ),
        "comparison_type": "cross_model_interaction",
        "contrast": "difference of condition-minus-baseline effects",
        "family": family,
        "endpoint": endpoint,
        **interaction_statistics(
            contrast=contrast,
            track=track,
            metadata=metadata,
            bootstrap_replicates=bootstrap_replicates,
            randomization_replicates=randomization_replicates,
            seed=seed,
        ),
    }


def causal_interaction(
    *,
    root: Path,
    first_model: str,
    second_model: str,
    bootstrap_replicates: int,
    randomization_replicates: int,
    seed: int,
) -> dict[str, Any]:
    metric = "full_triplet_success"

    def load(model: str, prompt: str) -> dict[str, dict[str, str]]:
        return read_csv_indexed(
            root
            / "runs"
            / model
            / "causal"
            / prompt
            / "score"
            / "pair_results.csv",
            "pair_id",
        )

    first_direct = load(first_model, "noncot")
    first_cot = load(first_model, "cot")
    second_direct = load(second_model, "noncot")
    second_cot = load(second_model, "cot")
    id_sets = [
        set(first_direct),
        set(first_cot),
        set(second_direct),
        set(second_cot),
    ]
    if len({frozenset(values) for values in id_sets}) != 1:
        raise ValueError("Cross-model causal interaction is not paired")
    pair_ids = sorted(first_direct)
    contrast = np.asarray(
        [
            (
                int(first_cot[pair_id][metric])
                - int(first_direct[pair_id][metric])
            )
            - (
                int(second_cot[pair_id][metric])
                - int(second_direct[pair_id][metric])
            )
            for pair_id in pair_ids
        ],
        dtype=float,
    )
    metadata = [{"capability": "causal_triplet"} for _ in pair_ids]
    return {
        "model_slug": f"{first_model}__vs__{second_model}",
        "model": (
            f"{MODEL_NAMES.get(first_model, first_model)} minus "
            f"{MODEL_NAMES.get(second_model, second_model)}"
        ),
        "comparison_type": "cross_model_interaction",
        "contrast": "difference of CoT-minus-direct effects",
        "family": "prompt_protocol",
        "endpoint": "cot_minus_direct_full_triplet_success",
        **interaction_statistics(
            contrast=contrast,
            track=None,
            metadata=metadata,
            bootstrap_replicates=bootstrap_replicates,
            randomization_replicates=randomization_replicates,
            seed=seed,
        ),
    }


def add_cross_model_interactions(
    rows: list[dict[str, Any]],
    *,
    root: Path,
    model_slugs: list[str],
    bootstrap_replicates: int,
    randomization_replicates: int,
    seed: int,
) -> None:
    if len(model_slugs) != 2:
        return
    first, second = model_slugs
    specifications = [
        (
            "fidelity_do_you_see_me",
            "do_you_see_me",
            "native",
            condition,
            "visual_fidelity",
            f"do_you_see_me_{condition}",
        )
        for condition in ("downsample_50", "downsample_25")
    ] + [
        (
            "fidelity_minds_eye",
            "minds_eye",
            "native",
            condition,
            "visual_fidelity",
            f"minds_eye_{condition}",
        )
        for condition in ("downsample_50", "downsample_25")
    ] + [
        (
            "abstraction",
            "minds_eye",
            "baseline",
            condition,
            "abstraction_cue",
            condition,
        )
        for condition in ("oracle_abstraction", "mismatched_abstraction")
    ]
    for index, (
        score_directory,
        track,
        baseline,
        condition,
        family,
        endpoint,
    ) in enumerate(specifications):
        rows.append(
            condition_interaction(
                first_path=root
                / "runs"
                / first
                / "local_scores"
                / score_directory
                / "item_results.csv",
                second_path=root
                / "runs"
                / second
                / "local_scores"
                / score_directory
                / "item_results.csv",
                first_model=first,
                second_model=second,
                track=track,
                baseline=baseline,
                condition=condition,
                family=family,
                endpoint=endpoint,
                bootstrap_replicates=bootstrap_replicates,
                randomization_replicates=randomization_replicates,
                seed=seed + index * 101,
            )
        )
    rows.append(
        causal_interaction(
            root=root,
            first_model=first,
            second_model=second,
            bootstrap_replicates=bootstrap_replicates,
            randomization_replicates=randomization_replicates,
            seed=seed + len(specifications) * 101,
        )
    )


def score_model(root: Path, model_slug: str, bootstrap: int, seed: int):
    model_root = root / "runs" / model_slug
    require_quality_gates(model_root)
    score_root = model_root / "local_scores"
    dysm = score_conditions(
        track="do_you_see_me",
        condition_paths=condition_paths(
            model_root,
            "fidelity/do_you_see_me",
            ["native", "downsample_50", "downsample_25"],
            "noncot",
        ),
        baseline_name="native",
        output=score_root / "fidelity_do_you_see_me",
        bootstrap_replicates=bootstrap,
        seed=seed,
    )
    minds_eye = score_conditions(
        track="minds_eye",
        condition_paths=condition_paths(
            model_root,
            "fidelity/minds_eye",
            ["native", "downsample_50", "downsample_25"],
            "cot",
        ),
        baseline_name="native",
        output=score_root / "fidelity_minds_eye",
        bootstrap_replicates=bootstrap,
        seed=seed,
    )
    abstraction = score_conditions(
        track="minds_eye",
        condition_paths=condition_paths(
            model_root,
            "abstraction",
            ["baseline", "oracle_abstraction", "mismatched_abstraction"],
            "cot",
        ),
        baseline_name="baseline",
        output=score_root / "abstraction",
        bootstrap_replicates=bootstrap,
        seed=seed,
    )
    causal = read_json(model_root / "causal/prompt_comparison/summary.json")
    return dysm, minds_eye, abstraction, causal


def synthesize(
    *,
    root: Path,
    model_slugs: list[str],
    output: Path,
    bootstrap: int,
    seed: int,
    confirmatory: bool,
) -> dict[str, Any]:
    primary: list[dict[str, Any]] = []
    for model_slug in model_slugs:
        dysm, minds_eye, abstraction, causal = score_model(
            root, model_slug, bootstrap, seed
        )
        for condition in ("downsample_50", "downsample_25"):
            add_condition_endpoint(
                primary,
                model_slug=model_slug,
                family="visual_fidelity",
                endpoint=f"do_you_see_me_{condition}",
                comparison=find_comparison(dysm, condition),
            )
            add_condition_endpoint(
                primary,
                model_slug=model_slug,
                family="visual_fidelity",
                endpoint=f"minds_eye_{condition}",
                comparison=find_comparison(minds_eye, condition),
            )
        for condition in ("oracle_abstraction", "mismatched_abstraction"):
            add_condition_endpoint(
                primary,
                model_slug=model_slug,
                family="abstraction_cue",
                endpoint=condition,
                comparison=find_comparison(abstraction, condition),
            )
        causal_row = next(
            row
            for row in causal["comparisons"]
            if row["metric"] == "full_triplet_success"
        )
        primary.append(
            {
                "model_slug": model_slug,
                "model": MODEL_NAMES.get(model_slug, model_slug),
                "comparison_type": "within_model",
                "contrast": "CoT minus direct",
                "family": "prompt_protocol",
                "endpoint": "cot_minus_direct_full_triplet_success",
                "effect": float(causal_row["delta"]),
                "interval_low": float(causal_row["bootstrap_95_interval"][0]),
                "interval_high": float(causal_row["bootstrap_95_interval"][1]),
                "raw_p": float(causal_row["mcnemar_exact_p"]),
            }
        )

    add_cross_model_interactions(
        primary,
        root=root,
        model_slugs=model_slugs,
        bootstrap_replicates=bootstrap,
        randomization_replicates=bootstrap,
        seed=seed + 10_000,
    )
    adjusted = holm_adjust([row["raw_p"] for row in primary])
    for row, adjusted_p in zip(primary, adjusted, strict=True):
        row["holm_adjusted_p"] = adjusted_p
        row["interval_excludes_zero"] = bool(
            row["interval_low"] > 0 or row["interval_high"] < 0
        )
        row["confirmatory"] = bool(
            adjusted_p < 0.05 and row["interval_excludes_zero"]
        )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "primary_endpoints.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(primary[0]))
        writer.writeheader()
        writer.writerows(primary)

    confirmed = [row for row in primary if row["confirmatory"]]
    summary = {
        "schema_version": "ms-vista-controlled-analysis-summary-v1",
        "analysis_id": "ms-vista-controlled-analysis-v2",
        "analysis_scope": "confirmatory" if confirmatory else "exploratory_follow_up",
        "analysis_plan": describe_path(ANALYSIS_PLAN),
        "models": model_slugs,
        "bootstrap_replicates": bootstrap,
        "randomization_replicates": bootstrap,
        "seed": seed,
        "multiplicity_correction": (
            "Holm across all prespecified within-model and cross-model "
            "primary-endpoint tests"
        ),
        "primary_endpoints": primary,
        "confirmed_findings": confirmed,
    }
    write_json(output / "summary.json", summary)

    lines = [
        "# Controlled analysis findings",
        "",
        (
            "Only effects whose paired 95% interval excludes zero and whose "
            "global Holm-adjusted p-value is below 0.05 are listed as confirmed."
        ),
        "",
    ]
    if confirmed:
        for row in confirmed:
            lines.append(
                f"- {row['model']}, {row['endpoint']}: "
                f"{100 * row['effect']:+.2f} points "
                f"(95% interval {100 * row['interval_low']:+.2f} to "
                f"{100 * row['interval_high']:+.2f}; "
                f"Holm p={row['holm_adjusted_p']:.4g})."
            )
    else:
        lines.append(
            "- No prespecified primary effect passed both confirmation criteria."
        )
    lines.extend(
        [
            "",
            (
                "Subgroup tables are exploratory and use Holm correction within "
                "each condition comparison. They should not replace the primary endpoints."
            ),
            "",
        ]
    )
    (output / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--models", default="qwen3-vl-8b,internvl35-8b"
    )
    parser.add_argument(
        "--exploratory-model-set",
        action="store_true",
        help=(
            "Allow a model list outside the frozen confirmatory plan. Results "
            "must then be labeled exploratory."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_slugs = [item.strip() for item in args.models.split(",") if item.strip()]
    if not model_slugs:
        raise ValueError("--models cannot be empty")
    if (
        tuple(model_slugs) != PRESPECIFIED_MODELS
        and not args.exploratory_model_set
    ):
        raise ValueError(
            "The confirmatory scorer requires the frozen model order "
            f"{','.join(PRESPECIFIED_MODELS)}. Pass --exploratory-model-set "
            "for a separately labeled follow-up."
        )
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    root = args.root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else root / "local_scoring"
    )
    summary = synthesize(
        root=root,
        model_slugs=model_slugs,
        output=output,
        bootstrap=args.bootstrap_replicates,
        seed=args.seed,
        confirmatory=not args.exploratory_model_set,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
