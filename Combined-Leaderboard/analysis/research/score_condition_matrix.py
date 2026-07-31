#!/usr/bin/env python3
"""Score matched benchmark condition outputs with paired uncertainty tests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_ROOT.parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
for candidate in (PROJECT_ROOT, ANALYSIS_ROOT, BACKEND_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from combined_visual_v14 import (  # noqa: E402
    exact_mcnemar_p,
    score_from_correctness,
    write_json,
)
from evaluation.research.common import read_jsonl  # noqa: E402
from scoring.task_scorer import TaskScorer  # noqa: E402


def parse_condition(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "Conditions must use NAME=/path/to/submission.jsonl"
        )
    return name.strip(), Path(raw_path).expanduser()


def answer_from_row(row: dict[str, Any]) -> Any:
    for field in ("answer", "extracted_answer", "prediction", "output"):
        if field in row:
            return row[field]
    return ""


def load_condition(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for row in read_jsonl(path):
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"{path}: row is missing question_id")
        if question_id in values:
            raise ValueError(f"{path}: duplicate question_id {question_id}")
        values[question_id] = answer_from_row(row)
    return values


def bootstrap_delta(
    *,
    track: str,
    candidate: np.ndarray,
    baseline: np.ndarray,
    metadata: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        if track == "do_you_see_me":
            label = f"{item['dimension']}:{item['capability']}"
        else:
            label = str(item["capability"])
        grouped[label].append(index)
    rng = np.random.default_rng(seed)
    deltas = np.zeros(replicates, dtype=float)
    for replicate in range(replicates):
        sampled: list[int] = []
        for indices in grouped.values():
            sampled.extend(rng.choice(indices, len(indices), replace=True).tolist())
        selected = np.asarray(sampled, dtype=np.int32)
        sampled_metadata = [metadata[int(index)] for index in selected]
        deltas[replicate] = score_from_correctness(
            track, candidate[selected], sampled_metadata
        ) - score_from_correctness(track, baseline[selected], sampled_metadata)
    return [
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    ]


def paired_bootstrap_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        indices = rng.integers(0, len(candidate), len(candidate))
        values[replicate] = float(
            candidate[indices].mean() - baseline[indices].mean()
        )
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


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


def benchmark_weights(
    track: str, metadata: list[dict[str, Any]]
) -> np.ndarray:
    weights = np.zeros(len(metadata), dtype=float)
    if track == "do_you_see_me":
        dimensions = sorted({str(item["dimension"]) for item in metadata})
        for dimension in dimensions:
            capabilities = sorted(
                {
                    str(item["capability"])
                    for item in metadata
                    if str(item["dimension"]) == dimension
                }
            )
            for capability in capabilities:
                indices = [
                    index
                    for index, item in enumerate(metadata)
                    if str(item["dimension"]) == dimension
                    and str(item["capability"]) == capability
                ]
                weights[indices] = (
                    1.0 / len(dimensions) / len(capabilities) / len(indices)
                )
    else:
        capabilities = sorted({str(item["capability"]) for item in metadata})
        for capability in capabilities:
            indices = [
                index
                for index, item in enumerate(metadata)
                if str(item["capability"]) == capability
            ]
            weights[indices] = 1.0 / len(capabilities) / len(indices)
    if not np.isclose(weights.sum(), 1.0):
        raise RuntimeError(f"Benchmark weights sum to {weights.sum():.12f}")
    return weights


def paired_randomization_p(
    *,
    track: str,
    candidate: np.ndarray,
    baseline: np.ndarray,
    metadata: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> float:
    contrast = candidate.astype(float) - baseline.astype(float)
    weights = benchmark_weights(track, metadata)
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


def subgroup_label(track: str, item: dict[str, Any]) -> str:
    if track == "do_you_see_me":
        return f"{item['dimension']}:{item['capability']}"
    return str(item["capability"])


def score_conditions(
    *,
    track: str,
    condition_paths: list[tuple[str, Path]],
    baseline_name: str,
    output: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    conditions = {name: load_condition(path.resolve()) for name, path in condition_paths}
    if baseline_name not in conditions:
        raise ValueError(f"Baseline condition '{baseline_name}' was not supplied")
    id_sets = {name: set(values) for name, values in conditions.items()}
    expected_ids = id_sets[baseline_name]
    for name, identifiers in id_sets.items():
        if identifiers != expected_ids:
            missing = sorted(expected_ids - identifiers)
            extra = sorted(identifiers - expected_ids)
            raise ValueError(
                f"Condition {name} is not paired with baseline: "
                f"{len(missing)} missing, {len(extra)} extra"
            )
    scorer = TaskScorer(track)
    unknown = sorted(expected_ids - set(scorer.ground_truth))
    if unknown:
        raise ValueError(
            f"{len(unknown)} question IDs are outside {track}, including {unknown[:5]}"
        )
    item_ids = sorted(expected_ids)
    metadata = [scorer.ground_truth[question_id] for question_id in item_ids]
    correctness: dict[str, np.ndarray] = {}
    for name, predictions in conditions.items():
        correctness[name] = np.asarray(
            [
                int(
                    scorer._grade_condition(
                        question_id,
                        predictions[question_id],
                        scorer.primary_condition,
                    )
                )
                for question_id in item_ids
            ],
            dtype=np.uint8,
        )

    baseline = correctness[baseline_name]
    condition_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        grouped[subgroup_label(track, item)].append(index)
    subgroup_condition_rows: list[dict[str, Any]] = []
    subgroup_comparison_rows: list[dict[str, Any]] = []
    for name, values in correctness.items():
        condition_rows.append(
            {
                "condition": name,
                "item_count": len(item_ids),
                "micro_accuracy": float(values.mean()),
                "benchmark_macro": score_from_correctness(track, values, metadata),
                "invalid_or_wrong_count": int(np.sum(values == 0)),
            }
        )
        for subgroup, indices in sorted(grouped.items()):
            selected = np.asarray(indices, dtype=np.int32)
            subgroup_condition_rows.append(
                {
                    "condition": name,
                    "subgroup": subgroup,
                    "item_count": len(indices),
                    "accuracy": float(values[selected].mean()),
                }
            )
        if name == baseline_name:
            continue
        candidate_only = int(np.sum((values == 1) & (baseline == 0)))
        baseline_only = int(np.sum((values == 0) & (baseline == 1)))
        candidate_macro = score_from_correctness(track, values, metadata)
        baseline_macro = score_from_correctness(track, baseline, metadata)
        comparison_rows.append(
            {
                "condition": name,
                "baseline": baseline_name,
                "macro_delta": candidate_macro - baseline_macro,
                "micro_delta": float(values.mean() - baseline.mean()),
                "candidate_only_correct": candidate_only,
                "baseline_only_correct": baseline_only,
                "mcnemar_exact_p": exact_mcnemar_p(
                    candidate_only, baseline_only
                ),
                "paired_randomization_p": paired_randomization_p(
                    track=track,
                    candidate=values,
                    baseline=baseline,
                    metadata=metadata,
                    replicates=bootstrap_replicates,
                    seed=seed + len(comparison_rows),
                ),
                "stratified_bootstrap_95_interval": bootstrap_delta(
                    track=track,
                    candidate=values,
                    baseline=baseline,
                    metadata=metadata,
                    replicates=bootstrap_replicates,
                    seed=seed,
                ),
            }
        )
        subgroup_rows_for_condition: list[dict[str, Any]] = []
        for subgroup_index, (subgroup, indices) in enumerate(sorted(grouped.items())):
            selected = np.asarray(indices, dtype=np.int32)
            candidate_group = values[selected]
            baseline_group = baseline[selected]
            candidate_only_group = int(
                np.sum((candidate_group == 1) & (baseline_group == 0))
            )
            baseline_only_group = int(
                np.sum((candidate_group == 0) & (baseline_group == 1))
            )
            subgroup_rows_for_condition.append(
                {
                    "condition": name,
                    "baseline": baseline_name,
                    "subgroup": subgroup,
                    "item_count": len(indices),
                    "candidate_accuracy": float(candidate_group.mean()),
                    "baseline_accuracy": float(baseline_group.mean()),
                    "accuracy_delta": float(
                        candidate_group.mean() - baseline_group.mean()
                    ),
                    "candidate_only_correct": candidate_only_group,
                    "baseline_only_correct": baseline_only_group,
                    "mcnemar_exact_p": exact_mcnemar_p(
                        candidate_only_group, baseline_only_group
                    ),
                    "paired_bootstrap_95_interval": paired_bootstrap_delta(
                        candidate_group,
                        baseline_group,
                        replicates=bootstrap_replicates,
                        seed=seed + subgroup_index,
                    ),
                }
            )
        adjusted = holm_adjust(
            [row["mcnemar_exact_p"] for row in subgroup_rows_for_condition]
        )
        for row, adjusted_p in zip(
            subgroup_rows_for_condition, adjusted, strict=True
        ):
            row["holm_adjusted_p"] = adjusted_p
        subgroup_comparison_rows.extend(subgroup_rows_for_condition)

    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("condition_scores.csv", condition_rows),
        ("paired_comparisons.csv", comparison_rows),
        ("subgroup_condition_scores.csv", subgroup_condition_rows),
        ("subgroup_paired_comparisons.csv", subgroup_comparison_rows),
    ):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    item_rows = [
        {
            "question_id": question_id,
            "subgroup": subgroup_label(track, metadata[index]),
            **{
                condition: int(values[index])
                for condition, values in correctness.items()
            },
        }
        for index, question_id in enumerate(item_ids)
    ]
    with (output / "item_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(item_rows[0]))
        writer.writeheader()
        writer.writerows(item_rows)
    summary = {
        "schema_version": "ms-vista-condition-matrix-v1",
        "track": track,
        "baseline": baseline_name,
        "item_count": len(item_ids),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "conditions": condition_rows,
        "comparisons": comparison_rows,
        "subgroup_conditions": subgroup_condition_rows,
        "subgroup_comparisons": subgroup_comparison_rows,
        "subgroup_multiplicity_correction": (
            "Holm within each condition-versus-baseline comparison"
        ),
        "primary_test": (
            "Paired label-swap randomization under the production benchmark "
            "macro weights"
        ),
        "submissions": {
            name: str(path.resolve()) for name, path in condition_paths
        },
    }
    write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track", choices=("do_you_see_me", "minds_eye"), required=True
    )
    parser.add_argument(
        "--condition",
        action="append",
        type=parse_condition,
        required=True,
        help="Repeat NAME=/path/to/submission.jsonl for every matched condition",
    )
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.condition) < 2:
        raise ValueError("At least two --condition values are required")
    names = [name for name, _path in args.condition]
    if len(names) != len(set(names)):
        raise ValueError("Condition names must be unique")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    summary = score_conditions(
        track=args.track,
        condition_paths=args.condition,
        baseline_name=args.baseline,
        output=args.output.resolve(),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
