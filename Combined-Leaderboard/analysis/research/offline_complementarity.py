#!/usr/bin/env python3
"""Measure model complementarity and leakage-free task routing on v4 outputs."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_ROOT.parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
for candidate in (PROJECT_ROOT, ANALYSIS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from combined_visual_v14 import (  # noqa: E402
    DEFAULT_BUNDLE,
    exact_mcnemar_p,
    score_from_correctness,
    write_csv,
    write_json,
)
from submission_readiness_v14 import load_cohort  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "analysis/results/research_experiments/complementarity"
TRACKS = ("do_you_see_me", "minds_eye")
TRACK_SCORE_FIELDS = {
    "do_you_see_me": "do_you_see_me_macro",
    "minds_eye": "minds_eye_macro",
}


def task_label(track: str, item: dict[str, Any]) -> str:
    if track == "do_you_see_me":
        return f"{item['dimension']}:{item['capability']}"
    return str(item["capability"])


def grouped_item_indices(
    track: str, metadata: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[task_label(track, item)].append(index)
    return {
        label: np.asarray(indices, dtype=np.int32)
        for label, indices in sorted(groups.items())
    }


def pairwise_complementarity(cohort: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track in TRACKS:
        matrix = cohort["correctness"][track].astype(bool)
        metadata = cohort["metadata"][track]
        score_field = TRACK_SCORE_FIELDS[track]
        for first, second in itertools.combinations(range(matrix.shape[0]), 2):
            first_correct = matrix[first]
            second_correct = matrix[second]
            oracle = first_correct | second_correct
            first_only = first_correct & ~second_correct
            second_only = second_correct & ~first_correct
            either_wrong = ~first_correct | ~second_correct
            both_wrong = ~first_correct & ~second_correct
            first_score = float(cohort["ranking"][first][score_field])
            second_score = float(cohort["ranking"][second][score_field])
            oracle_score = score_from_correctness(track, oracle, metadata)
            rows.append(
                {
                    "track": track,
                    "model_a": cohort["display_names"][first],
                    "model_b": cohort["display_names"][second],
                    "model_a_macro": first_score,
                    "model_b_macro": second_score,
                    "best_member_macro": max(first_score, second_score),
                    "oracle_union_macro": oracle_score,
                    "oracle_gain": oracle_score - max(first_score, second_score),
                    "disagreement_rate": float(
                        np.mean(first_correct != second_correct)
                    ),
                    "model_a_unique_correct_rate": float(np.mean(first_only)),
                    "model_b_unique_correct_rate": float(np.mean(second_only)),
                    "both_wrong_rate": float(np.mean(both_wrong)),
                    "error_jaccard": (
                        float(np.sum(both_wrong) / np.sum(either_wrong))
                        if np.any(either_wrong)
                        else 1.0
                    ),
                    "mcnemar_exact_p": exact_mcnemar_p(
                        int(np.sum(first_only)), int(np.sum(second_only))
                    ),
                    "item_count": int(first_correct.size),
                }
            )
    return sorted(rows, key=lambda row: (row["track"], -row["oracle_gain"]))


def assign_folds(
    groups: dict[str, np.ndarray],
    fold_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    item_count = max(int(indices.max()) for indices in groups.values()) + 1
    assignments = np.full(item_count, -1, dtype=np.int16)
    for indices in groups.values():
        shuffled = rng.permutation(indices)
        assignments[shuffled] = np.arange(shuffled.size) % fold_count
    if np.any(assignments < 0):
        raise RuntimeError("Fold assignment omitted one or more items")
    return assignments


def choose_best_model(
    track: str,
    matrix: np.ndarray,
    metadata: list[dict[str, Any]],
    indices: np.ndarray,
) -> int:
    selected_metadata = [metadata[int(index)] for index in indices]
    scores = np.asarray(
        [
            score_from_correctness(
                track,
                matrix[model_index, indices],
                selected_metadata,
            )
            for model_index in range(matrix.shape[0])
        ]
    )
    return int(np.flatnonzero(scores == scores.max())[0])


def run_task_routing(
    cohort: dict[str, Any],
    *,
    repeats: int,
    fold_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repeat_rows: list[dict[str, Any]] = []
    selection_counts: dict[tuple[str, str, str], int] = Counter()
    for track_offset, track in enumerate(TRACKS):
        matrix = cohort["correctness"][track].astype(np.uint8)
        metadata = cohort["metadata"][track]
        groups = grouped_item_indices(track, metadata)
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + track_offset * 1_000_003 + repeat)
            fold_by_item = assign_folds(groups, fold_count, rng)
            routed = np.zeros(matrix.shape[1], dtype=np.uint8)
            global_baseline = np.zeros(matrix.shape[1], dtype=np.uint8)
            for fold in range(fold_count):
                test = np.flatnonzero(fold_by_item == fold)
                train = np.flatnonzero(fold_by_item != fold)
                global_model = choose_best_model(track, matrix, metadata, train)
                global_baseline[test] = matrix[global_model, test]
                for task, task_indices in groups.items():
                    task_test = task_indices[fold_by_item[task_indices] == fold]
                    task_train = task_indices[fold_by_item[task_indices] != fold]
                    training_accuracy = matrix[:, task_train].mean(axis=1)
                    selected_model = int(
                        np.flatnonzero(training_accuracy == training_accuracy.max())[0]
                    )
                    routed[task_test] = matrix[selected_model, task_test]
                    selection_counts[
                        (track, task, cohort["display_names"][selected_model])
                    ] += 1
            routed_macro = score_from_correctness(track, routed, metadata)
            global_macro = score_from_correctness(track, global_baseline, metadata)
            repeat_rows.append(
                {
                    "track": track,
                    "repeat": repeat + 1,
                    "fold_count": fold_count,
                    "task_routed_macro": routed_macro,
                    "global_selected_macro": global_macro,
                    "delta": routed_macro - global_macro,
                }
            )

    selection_rows = [
        {
            "track": track,
            "task": task,
            "model": model,
            "selection_count": count,
            "selection_rate": count / (repeats * fold_count),
        }
        for (track, task, model), count in sorted(selection_counts.items())
    ]
    return repeat_rows, selection_rows


def oracle_summary(cohort: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track in TRACKS:
        matrix = cohort["correctness"][track].astype(bool)
        metadata = cohort["metadata"][track]
        oracle = np.any(matrix, axis=0)
        best_score = max(
            float(model[TRACK_SCORE_FIELDS[track]]) for model in cohort["ranking"]
        )
        rows.append(
            {
                "track": track,
                "model_count": matrix.shape[0],
                "item_count": matrix.shape[1],
                "all_model_oracle_macro": score_from_correctness(
                    track, oracle, metadata
                ),
                "best_single_model_macro": best_score,
                "oracle_gain": score_from_correctness(track, oracle, metadata)
                - best_score,
                "universally_missed_items": int(np.sum(~oracle)),
                "universally_missed_rate": float(np.mean(~oracle)),
            }
        )
    return rows


def distribution_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "p2_5": float(np.percentile(array, 2.5)),
        "median": float(np.median(array)),
        "p97_5": float(np.percentile(array, 97.5)),
        "maximum": float(array.max()),
        "positive_fraction": float(np.mean(array > 0)),
    }


def write_report(
    path: Path,
    *,
    pairwise: list[dict[str, Any]],
    routing: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    repeats: int,
    folds: int,
) -> None:
    lines = [
        "# Model Complementarity and Task Routing",
        "",
        "This analysis uses canonical v4 evidence-audited item-level outputs and the "
        "production leaderboard scorer. Task routing is evaluated out of fold: the model for "
        "each task is selected only from training folds and scored on held-out items.",
        "",
        "Repeated folds measure split sensitivity. They are not treated as "
        "independent experimental replications.",
        "",
        "## All-model oracle",
        "",
        "| Track | Best model | Oracle | Gain | Universally missed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in oracle:
        lines.append(
            f"| {row['track']} | {row['best_single_model_macro']:.3f} | "
            f"{row['all_model_oracle_macro']:.3f} | {row['oracle_gain']:.3f} | "
            f"{row['universally_missed_items']} ({row['universally_missed_rate']:.1%}) |"
        )
    lines.extend(["", "## Best pairwise oracle gains", ""])
    for track in TRACKS:
        lines.append(f"### {track}")
        lines.append("")
        lines.append("| Model A | Model B | Best member | Oracle | Gain |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for row in [candidate for candidate in pairwise if candidate["track"] == track][
            :5
        ]:
            lines.append(
                f"| {row['model_a']} | {row['model_b']} | "
                f"{row['best_member_macro']:.3f} | "
                f"{row['oracle_union_macro']:.3f} | {row['oracle_gain']:.3f} |"
            )
        lines.append("")
        absolute = sorted(
            [candidate for candidate in pairwise if candidate["track"] == track],
            key=lambda row: -row["oracle_union_macro"],
        )[:5]
        lines.append("Highest absolute pair-oracle scores:")
        lines.append("")
        lines.append("| Model A | Model B | Pair oracle | Gain |")
        lines.append("| --- | --- | ---: | ---: |")
        for row in absolute:
            lines.append(
                f"| {row['model_a']} | {row['model_b']} | "
                f"{row['oracle_union_macro']:.3f} | {row['oracle_gain']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Out-of-fold task routing",
            "",
            f"Protocol: {folds}-fold stratification within every task, repeated "
            f"{repeats} times with fixed recorded seeds.",
            "",
            "| Track | Routed macro | Global baseline | Delta | Positive splits |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for track in TRACKS:
        selected = [row for row in routing if row["track"] == track]
        routed_summary = distribution_summary(
            [row["task_routed_macro"] for row in selected]
        )
        global_summary = distribution_summary(
            [row["global_selected_macro"] for row in selected]
        )
        delta_summary = distribution_summary([row["delta"] for row in selected])
        lines.append(
            f"| {track} | {routed_summary['mean']:.3f} | "
            f"{global_summary['mean']:.3f} | {delta_summary['mean']:+.3f} "
            f"[{delta_summary['p2_5']:+.3f}, {delta_summary['p97_5']:+.3f}] | "
            f"{delta_summary['positive_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "The oracle results quantify available complementarity, not a deployable "
            "system. The out-of-fold router is deployable in principle, but it uses "
            "benchmark task labels and therefore tests task-level specialization "
            "rather than item-level routing.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    cohort = load_cohort(args.bundle.resolve())
    pairwise = pairwise_complementarity(cohort)
    routing, selections = run_task_routing(
        cohort,
        repeats=args.repeats,
        fold_count=args.folds,
        seed=args.seed,
    )
    oracle = oracle_summary(cohort)
    output = args.output.resolve()
    write_csv(output / "pairwise_complementarity.csv", pairwise, list(pairwise[0]))
    write_csv(output / "routing_repeats.csv", routing, list(routing[0]))
    write_csv(output / "routing_selection_frequency.csv", selections, list(selections[0]))
    write_csv(output / "all_model_oracle.csv", oracle, list(oracle[0]))
    summary = {
        "schema_version": "ms-vista-complementarity-v1",
        "bundle": str(args.bundle.resolve()),
        "parameters": {
            "repeats": args.repeats,
            "folds": args.folds,
            "seed": args.seed,
        },
        "oracle": oracle,
        "routing": {
            track: {
                metric: distribution_summary(
                    [
                        row[metric]
                        for row in routing
                        if row["track"] == track
                    ]
                )
                for metric in (
                    "task_routed_macro",
                    "global_selected_macro",
                    "delta",
                )
            }
            for track in TRACKS
        },
        "best_pairs": {
            track: [
                row
                for row in pairwise
                if row["track"] == track
            ][:10]
            for track in TRACKS
        },
        "highest_absolute_pair_oracles": {
            track: sorted(
                [row for row in pairwise if row["track"] == track],
                key=lambda row: -row["oracle_union_macro"],
            )[:10]
            for track in TRACKS
        },
    }
    write_json(output / "summary.json", summary)
    write_report(
        output / "REPORT.md",
        pairwise=pairwise,
        routing=routing,
        oracle=oracle,
        repeats=args.repeats,
        folds=args.folds,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
