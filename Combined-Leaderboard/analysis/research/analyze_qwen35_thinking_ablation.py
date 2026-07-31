#!/usr/bin/env python3
"""Score the prespecified Qwen3.5 prompt-by-thinking factorial experiment."""

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
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for candidate in (PROJECT_ROOT, BACKEND_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from analysis.combined_visual_v14 import score_from_correctness  # noqa: E402
from analysis.research.score_condition_matrix import (  # noqa: E402
    benchmark_weights,
    holm_adjust,
    load_condition,
)
from evaluation.research.common import describe_path, write_json  # noqa: E402
from scoring.task_scorer import TaskScorer  # noqa: E402


CONDITIONS = (
    "direct_disabled",
    "direct_enabled",
    "cot_disabled",
    "cot_enabled",
)
PLAN = (
    PROJECT_ROOT
    / "evaluation/research/PRESPECIFIED_QWEN35_THINKING_PLAN.md"
)
TARGET_MODEL = "Qwen/Qwen3.5-9B"
TARGET_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ENDPOINT_SPECS = (
    (
        "thinking_effect_direct_prompt",
        {"direct_enabled": 1.0, "direct_disabled": -1.0},
    ),
    (
        "thinking_effect_reasoning_prompt",
        {"cot_enabled": 1.0, "cot_disabled": -1.0},
    ),
    (
        "reasoning_prompt_effect_thinking_disabled",
        {"cot_disabled": 1.0, "direct_disabled": -1.0},
    ),
    (
        "reasoning_prompt_effect_thinking_enabled",
        {"cot_enabled": 1.0, "direct_enabled": -1.0},
    ),
    (
        "prompt_by_thinking_interaction",
        {
            "cot_enabled": 1.0,
            "direct_enabled": -1.0,
            "cot_disabled": -1.0,
            "direct_disabled": 1.0,
        },
    ),
)


def _macro(values: np.ndarray, metadata: list[dict[str, Any]]) -> float:
    return score_from_correctness("minds_eye", values, metadata)


def _weighted_effect(
    arrays: dict[str, np.ndarray],
    coefficients: dict[str, float],
    metadata: list[dict[str, Any]],
) -> float:
    return float(
        sum(
            coefficient * _macro(arrays[condition], metadata)
            for condition, coefficient in coefficients.items()
        )
    )


def _item_contrast(
    arrays: dict[str, np.ndarray], coefficients: dict[str, float]
) -> np.ndarray:
    first = next(iter(arrays.values()))
    contrast = np.zeros(len(first), dtype=float)
    for condition, coefficient in coefficients.items():
        contrast += coefficient * arrays[condition].astype(float)
    return contrast


def _stratified_bootstrap(
    arrays: dict[str, np.ndarray],
    coefficients: dict[str, float],
    metadata: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        grouped[str(item["capability"])].append(index)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled: list[int] = []
        for indices in grouped.values():
            sampled.extend(
                rng.choice(indices, len(indices), replace=True).tolist()
            )
        selected = np.asarray(sampled, dtype=np.int32)
        selected_metadata = [metadata[int(index)] for index in selected]
        values[replicate] = _weighted_effect(
            {
                condition: condition_values[selected]
                for condition, condition_values in arrays.items()
            },
            coefficients,
            selected_metadata,
        )
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def _sign_randomization_p(
    contrast: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> float:
    weights = benchmark_weights("minds_eye", metadata)
    observed = abs(float(np.dot(weights, contrast)))
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < replicates:
        count = min(500, replicates - completed)
        signs = rng.integers(0, 2, size=(count, len(contrast))) * 2 - 1
        permuted = np.abs(signs @ (weights * contrast))
        extreme += int(np.sum(permuted >= observed - 1e-12))
        completed += count
    return (extreme + 1) / (replicates + 1)


def factorial_statistics(
    arrays: dict[str, np.ndarray],
    metadata: list[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    randomization_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    missing = sorted(set(CONDITIONS) - set(arrays))
    if missing:
        raise ValueError(f"Missing factorial conditions: {', '.join(missing)}")
    lengths = {len(values) for values in arrays.values()}
    if lengths != {len(metadata)}:
        raise ValueError("Factorial arrays and metadata are not paired")

    rows: list[dict[str, Any]] = []
    for index, (endpoint, coefficients) in enumerate(ENDPOINT_SPECS):
        contrast = _item_contrast(arrays, coefficients)
        interval = _stratified_bootstrap(
            arrays,
            coefficients,
            metadata,
            replicates=bootstrap_replicates,
            seed=seed + index,
        )
        rows.append(
            {
                "endpoint": endpoint,
                "effect": _weighted_effect(arrays, coefficients, metadata),
                "interval_low": interval[0],
                "interval_high": interval[1],
                "raw_p": _sign_randomization_p(
                    contrast,
                    metadata,
                    replicates=randomization_replicates,
                    seed=seed + 100 + index,
                ),
            }
        )
    adjusted = holm_adjust([float(row["raw_p"]) for row in rows])
    for row, adjusted_p in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = adjusted_p
        row["confirmatory"] = bool(
            row["interval_low"] * row["interval_high"] > 0
            and adjusted_p < 0.05
        )
    return rows


def _require_quality_gates(root: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        path = root / "conditions" / condition / "quality/summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing quality summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary.get("quality_gate_passed"):
            raise ValueError(f"Quality gate failed for {condition}: {path}")
        summaries[condition] = summary
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quality_status_by_id(root: Path, condition: str) -> dict[str, str]:
    path = root / "conditions" / condition / "quality/item_quality.csv"
    statuses: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            question_id = str(row.get("question_id") or "").strip()
            if not question_id or question_id in statuses:
                raise ValueError(f"Invalid quality item table: {path}")
            statuses[question_id] = str(row.get("extractor_status") or "")
    return statuses


def analyze(
    root: Path,
    output: Path,
    *,
    bootstrap_replicates: int = 10_000,
    randomization_replicates: int = 10_000,
    seed: int = 20260723,
) -> dict[str, Any]:
    root = root.resolve()
    quality = _require_quality_gates(root)
    submissions = {
        condition: load_condition(
            root / "conditions" / condition / "submission.jsonl"
        )
        for condition in CONDITIONS
    }
    expected_ids = set(submissions[CONDITIONS[0]])
    for condition, rows in submissions.items():
        if set(rows) != expected_ids:
            raise ValueError(
                f"{condition} is not paired with {CONDITIONS[0]}"
            )

    scorer = TaskScorer("minds_eye")
    unknown = sorted(expected_ids - set(scorer.ground_truth))
    if unknown:
        raise ValueError(
            f"{len(unknown)} IDs are outside Mind's Eye, including {unknown[:5]}"
        )
    item_ids = sorted(expected_ids)
    metadata = [scorer.ground_truth[question_id] for question_id in item_ids]
    arrays = {
        condition: np.asarray(
            [
                int(
                    scorer._grade_condition(
                        question_id,
                        rows[question_id],
                        scorer.primary_condition,
                    )
                )
                for question_id in item_ids
            ],
            dtype=np.uint8,
        )
        for condition, rows in submissions.items()
    }
    quality_statuses = {
        condition: _quality_status_by_id(root, condition)
        for condition in CONDITIONS
    }
    for condition, statuses in quality_statuses.items():
        if set(statuses) != expected_ids:
            raise ValueError(
                f"{condition} quality rows are not paired with submissions"
            )
    endpoints = factorial_statistics(
        arrays,
        metadata,
        bootstrap_replicates=bootstrap_replicates,
        randomization_replicates=randomization_replicates,
        seed=seed,
    )

    condition_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        resolved = np.asarray(
            [
                quality_statuses[condition][question_id] == "resolved"
                for question_id in item_ids
            ],
            dtype=bool,
        )
        token_summary = quality[condition].get("completion_tokens") or {}
        finish_reasons = quality[condition].get("finish_reason_counts") or {}
        condition_rows.append(
            {
                "condition": condition,
                "item_count": len(item_ids),
                "benchmark_macro": _macro(arrays[condition], metadata),
                "micro_accuracy": float(arrays[condition].mean()),
                "resolved_rate": float(resolved.mean()),
                "micro_accuracy_among_resolved": (
                    float(arrays[condition][resolved].mean())
                    if np.any(resolved)
                    else 0.0
                ),
                "length_stop_rate": float(
                    int(finish_reasons.get("length", 0)) / len(item_ids)
                ),
                "completion_tokens_p50": token_summary.get("p50", ""),
                "completion_tokens_p95": token_summary.get("p95", ""),
                "completion_tokens_p99": token_summary.get("p99", ""),
                "completion_tokens_max": token_summary.get("max", ""),
                "incorrect_or_unresolved": int(np.sum(arrays[condition] == 0)),
            }
        )
    task_rows: list[dict[str, Any]] = []
    task_contrast_rows: list[dict[str, Any]] = []
    capabilities = sorted({str(item["capability"]) for item in metadata})
    for capability in capabilities:
        indices = np.asarray(
            [
                index
                for index, item in enumerate(metadata)
                if str(item["capability"]) == capability
            ],
            dtype=np.int32,
        )
        for condition in CONDITIONS:
            task_rows.append(
                {
                    "capability": capability,
                    "condition": condition,
                    "item_count": len(indices),
                    "accuracy": float(arrays[condition][indices].mean()),
                }
            )
        selected_metadata = [metadata[int(index)] for index in indices]
        selected_arrays = {
            condition: values[indices] for condition, values in arrays.items()
        }
        capability_contrasts: list[dict[str, Any]] = []
        for endpoint_index, (endpoint, coefficients) in enumerate(ENDPOINT_SPECS):
            contrast = _item_contrast(selected_arrays, coefficients)
            interval = _stratified_bootstrap(
                selected_arrays,
                coefficients,
                selected_metadata,
                replicates=bootstrap_replicates,
                seed=seed + 1_000 + endpoint_index,
            )
            capability_contrasts.append(
                {
                    "capability": capability,
                    "endpoint": endpoint,
                    "item_count": len(indices),
                    "effect": _weighted_effect(
                        selected_arrays, coefficients, selected_metadata
                    ),
                    "interval_low": interval[0],
                    "interval_high": interval[1],
                    "raw_p": _sign_randomization_p(
                        contrast,
                        selected_metadata,
                        replicates=randomization_replicates,
                        seed=seed + 2_000 + endpoint_index,
                    ),
                }
            )
        task_contrast_rows.extend(capability_contrasts)
    for endpoint, _coefficients in ENDPOINT_SPECS:
        selected_rows = [
            row for row in task_contrast_rows if row["endpoint"] == endpoint
        ]
        adjusted = holm_adjust([float(row["raw_p"]) for row in selected_rows])
        for row, adjusted_p in zip(selected_rows, adjusted, strict=True):
            row["holm_adjusted_p_within_endpoint"] = adjusted_p
    item_rows = [
        {
            "question_id": question_id,
            "capability": metadata[index]["capability"],
            **{
                condition: int(arrays[condition][index])
                for condition in CONDITIONS
            },
        }
        for index, question_id in enumerate(item_ids)
    ]

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "condition_scores.csv", condition_rows)
    _write_csv(output / "confirmatory_endpoints.csv", endpoints)
    _write_csv(output / "task_condition_scores.csv", task_rows)
    _write_csv(output / "exploratory_task_contrasts.csv", task_contrast_rows)
    _write_csv(output / "item_results.csv", item_rows)
    summary = {
        "schema_version": "ms-vista-qwen35-thinking-factorial-v1",
        "status": "complete",
        "target_model": TARGET_MODEL,
        "target_revision": TARGET_REVISION,
        "item_count": len(item_ids),
        "conditions": condition_rows,
        "confirmatory_test_count": len(endpoints),
        "endpoints": endpoints,
        "multiplicity_correction": "Holm across five prespecified tests",
        "bootstrap_replicates": bootstrap_replicates,
        "randomization_replicates": randomization_replicates,
        "seed": seed,
        "quality": quality,
        "analysis_plan": describe_path(PLAN),
        "submissions": {
            condition: describe_path(
                root / "conditions" / condition / "submission.jsonl"
            )
            for condition in CONDITIONS
        },
    }
    write_json(output / "summary.json", summary)

    report = [
        "# Qwen3.5 Prompt by Thinking Ablation",
        "",
        "All four paired conditions passed the frozen quality gate.",
        "",
        "## Condition scores",
        "",
        "| Condition | Mind's Eye macro | Resolved | Length stops |",
        "| --- | ---: | ---: | ---: |",
    ]
    report.extend(
        (
            f"| {row['condition']} | {100 * row['benchmark_macro']:.2f}% | "
            f"{100 * row['resolved_rate']:.2f}% | "
            f"{100 * row['length_stop_rate']:.2f}% |"
        )
        for row in condition_rows
    )
    report.extend(
        [
            "",
            "## Confirmatory contrasts",
            "",
            "| Endpoint | Effect | 95% CI | Holm p | Confirmatory |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    report.extend(
        (
            f"| {row['endpoint']} | {100 * row['effect']:+.2f} pp | "
            f"[{100 * row['interval_low']:+.2f}, "
            f"{100 * row['interval_high']:+.2f}] | "
            f"{row['holm_adjusted_p']:.4g} | "
            f"{'yes' if row['confirmatory'] else 'no'} |"
        )
        for row in endpoints
    )
    report.extend(
        [
            "",
            "Task-level differences are exploratory. Unresolved and "
            "length-capped model outcomes are retained as incorrect.",
        ]
    )
    (output / "REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--randomization-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.root / "analysis"
    summary = analyze(
        args.root,
        output,
        bootstrap_replicates=args.bootstrap_replicates,
        randomization_replicates=args.randomization_replicates,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
