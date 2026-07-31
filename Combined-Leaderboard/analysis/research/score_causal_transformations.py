#!/usr/bin/env python3
"""Score paired causal-sensitivity and nuisance-invariance outputs."""

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

from evaluation.research.common import read_jsonl, write_json  # noqa: E402


METRIC_FIELDS = (
    "base_correct",
    "causal_correct",
    "nuisance_correct",
    "causal_answer_flip",
    "nuisance_answer_invariant",
    "causal_pair_success",
    "nuisance_pair_success",
    "full_triplet_success",
)


def canonical_letter(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    return ""


def load_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for row in read_jsonl(path):
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"{path}: submission row is missing question_id")
        if question_id in predictions:
            raise ValueError(f"{path}: duplicate question_id {question_id}")
        predictions[question_id] = canonical_letter(
            row.get("answer")
            or row.get("extracted_answer")
            or row.get("prediction")
            or row.get("output")
        )
    return predictions


def percentile_interval(values: np.ndarray) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def bootstrap_metrics(
    rows: list[dict[str, Any]], replicates: int, seed: int
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    fields = (
        "base_correct",
        "causal_correct",
        "nuisance_correct",
        "causal_answer_flip",
        "nuisance_answer_invariant",
        "causal_pair_success",
        "nuisance_pair_success",
        "full_triplet_success",
    )
    samples = {field: np.zeros(replicates, dtype=float) for field in fields}
    for replicate in range(replicates):
        indices = rng.integers(0, len(rows), len(rows))
        for field in fields:
            samples[field][replicate] = np.mean(
                [rows[int(index)][field] for index in indices]
            )
    return {field: percentile_interval(values) for field, values in samples.items()}


def score(
    *,
    ground_truth_path: Path,
    pairs_path: Path,
    submission_path: Path,
    output: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    truth_rows = read_jsonl(ground_truth_path)
    truth = {
        str(row["question_id"]): canonical_letter(row["answer"]) for row in truth_rows
    }
    predictions = load_predictions(submission_path)
    expected = set(truth)
    received = set(predictions)
    if missing := sorted(expected - received):
        raise ValueError(
            f"Submission is missing {len(missing)} question(s), including {missing[:5]}"
        )
    if unknown := sorted(received - expected):
        raise ValueError(
            f"Submission has {len(unknown)} unknown question(s), including {unknown[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for pair in read_jsonl(pairs_path):
        identifiers = {
            variant: str(pair[f"{variant}_question_id"])
            for variant in ("base", "causal", "nuisance")
        }
        answers = {variant: predictions[value] for variant, value in identifiers.items()}
        correct = {
            variant: bool(answers[variant] and answers[variant] == truth[identifier])
            for variant, identifier in identifiers.items()
        }
        row = {
            "pair_id": pair["pair_id"],
            "base_operation": pair.get("base_operation", ""),
            "causal_operation": pair.get("causal_operation", ""),
            "base_answer": answers["base"],
            "causal_answer": answers["causal"],
            "nuisance_answer": answers["nuisance"],
            "base_correct": int(correct["base"]),
            "causal_correct": int(correct["causal"]),
            "nuisance_correct": int(correct["nuisance"]),
            "causal_answer_flip": int(
                bool(answers["base"])
                and bool(answers["causal"])
                and answers["base"] != answers["causal"]
            ),
            "nuisance_answer_invariant": int(
                bool(answers["base"])
                and answers["base"] == answers["nuisance"]
            ),
            "causal_pair_success": int(correct["base"] and correct["causal"]),
            "nuisance_pair_success": int(
                correct["base"]
                and correct["nuisance"]
                and answers["base"] == answers["nuisance"]
            ),
            "full_triplet_success": int(
                correct["base"]
                and correct["causal"]
                and correct["nuisance"]
                and answers["base"] == answers["nuisance"]
            ),
        }
        rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "pair_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    intervals = bootstrap_metrics(rows, bootstrap_replicates, seed)
    operation_rows: list[dict[str, Any]] = []
    for operation_role in ("base_operation", "causal_operation"):
        for operation in sorted({str(row[operation_role]) for row in rows}):
            selected = [row for row in rows if row[operation_role] == operation]
            operation_rows.append(
                {
                    "operation_role": operation_role,
                    "operation": operation,
                    "pair_count": len(selected),
                    **{
                        metric: float(
                            np.mean([row[metric] for row in selected])
                        )
                        for metric in METRIC_FIELDS
                    },
                }
            )
    with (output / "operation_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(operation_rows[0]))
        writer.writeheader()
        writer.writerows(operation_rows)
    summary = {
        "schema_version": "ms-vista-causal-transform-score-v1",
        "submission": str(submission_path.resolve()),
        "pair_count": len(rows),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "metrics": {
            field: {
                "estimate": float(np.mean([row[field] for row in rows])),
                "bootstrap_95_interval": intervals[field],
            }
            for field in METRIC_FIELDS
        },
        "operation_breakdown": operation_rows,
    }
    write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    summary = score(
        ground_truth_path=args.ground_truth.resolve(),
        pairs_path=args.pairs.resolve(),
        submission_path=args.submission.resolve(),
        output=args.output.resolve(),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
