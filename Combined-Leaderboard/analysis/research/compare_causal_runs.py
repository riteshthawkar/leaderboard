#!/usr/bin/env python3
"""Compare two paired causal-transformation runs with corrected significance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_ROOT.parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
for candidate in (PROJECT_ROOT, ANALYSIS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluation.research.common import write_json  # noqa: E402


METRICS = (
    "base_correct",
    "causal_correct",
    "nuisance_correct",
    "causal_answer_flip",
    "nuisance_answer_invariant",
    "causal_pair_success",
    "nuisance_pair_success",
    "full_triplet_success",
)


def exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    tail = min(discordant_a, discordant_b)
    probability = 0.0
    for value in range(tail + 1):
        probability += math.exp(
            math.lgamma(total + 1)
            - math.lgamma(value + 1)
            - math.lgamma(total - value + 1)
            - total * math.log(2.0)
        )
    return min(1.0, 2.0 * probability)


def parse_run(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Runs must use NAME=/path/to/pair_results.csv")
    return name.strip(), Path(raw_path).expanduser()


def load_rows(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pair_id = str(row.get("pair_id") or "").strip()
            if not pair_id:
                raise ValueError(f"{path}: row is missing pair_id")
            if pair_id in rows:
                raise ValueError(f"{path}: duplicate pair_id {pair_id}")
            rows[pair_id] = {metric: int(row[metric]) for metric in METRICS}
    if not rows:
        raise ValueError(f"{path}: no pair rows")
    return rows


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


def bootstrap_interval(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    count = len(candidate)
    values = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        indices = rng.integers(0, count, count)
        values[replicate] = float(
            candidate[indices].mean() - baseline[indices].mean()
        )
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def compare(
    *,
    baseline_name: str,
    baseline_path: Path,
    candidate_name: str,
    candidate_path: Path,
    output: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    baseline_rows = load_rows(baseline_path)
    candidate_rows = load_rows(candidate_path)
    if set(baseline_rows) != set(candidate_rows):
        missing = sorted(set(baseline_rows) - set(candidate_rows))
        extra = sorted(set(candidate_rows) - set(baseline_rows))
        raise ValueError(
            f"Runs are not paired: {len(missing)} missing and {len(extra)} extra pairs"
        )
    pair_ids = sorted(baseline_rows)
    comparisons: list[dict[str, Any]] = []
    p_values: list[float] = []
    for metric_index, metric in enumerate(METRICS):
        baseline = np.asarray(
            [baseline_rows[pair_id][metric] for pair_id in pair_ids],
            dtype=np.uint8,
        )
        candidate = np.asarray(
            [candidate_rows[pair_id][metric] for pair_id in pair_ids],
            dtype=np.uint8,
        )
        candidate_only = int(np.sum((candidate == 1) & (baseline == 0)))
        baseline_only = int(np.sum((candidate == 0) & (baseline == 1)))
        p_value = exact_mcnemar_p(candidate_only, baseline_only)
        p_values.append(p_value)
        comparisons.append(
            {
                "metric": metric,
                "baseline_estimate": float(baseline.mean()),
                "candidate_estimate": float(candidate.mean()),
                "delta": float(candidate.mean() - baseline.mean()),
                "bootstrap_95_interval": bootstrap_interval(
                    candidate,
                    baseline,
                    replicates=bootstrap_replicates,
                    seed=seed + metric_index,
                ),
                "candidate_only_success": candidate_only,
                "baseline_only_success": baseline_only,
                "mcnemar_exact_p": p_value,
            }
        )
    adjusted = holm_adjust(p_values)
    for row, adjusted_p in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_p"] = adjusted_p

    summary = {
        "schema_version": "ms-vista-causal-run-comparison-v1",
        "baseline": baseline_name,
        "candidate": candidate_name,
        "pair_count": len(pair_ids),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "multiplicity_correction": "Holm across the eight prespecified metrics",
        "comparisons": comparisons,
        "inputs": {
            baseline_name: str(baseline_path.resolve()),
            candidate_name: str(candidate_path.resolve()),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    with (output / "comparisons.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=parse_run, required=True)
    parser.add_argument("--candidate", type=parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    baseline_name, baseline_path = args.baseline
    candidate_name, candidate_path = args.candidate
    if baseline_name == candidate_name:
        raise ValueError("Baseline and candidate names must differ")
    summary = compare(
        baseline_name=baseline_name,
        baseline_path=baseline_path.resolve(),
        candidate_name=candidate_name,
        candidate_path=candidate_path.resolve(),
        output=args.output.resolve(),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
