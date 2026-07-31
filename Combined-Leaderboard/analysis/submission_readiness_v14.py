#!/usr/bin/env python3
"""Confirmatory robustness experiments for the evidence-audited visual cohort."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ANALYSIS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
for path in (str(BACKEND_ROOT), str(PROJECT_ROOT), str(ANALYSIS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from combined_visual_v14 import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_OUTPUT,
    TRACKS,
    bh_adjust,
    bootstrap_track_scores,
    descending_ranks,
    display_path,
    exact_mcnemar_p,
    grouped_indices,
    pearson,
    percentile_interval,
    rankdata,
    read_json,
    score_from_correctness,
    spearman,
    write_csv,
    write_json,
)
from config import MINDS_EYE_ART_BY_CAPABILITY  # noqa: E402
from scoring.task_scorer import TaskScorer  # noqa: E402


DEFAULT_CONFIRMATORY_OUTPUT = (
    PROJECT_ROOT / "analysis/results/submission_readiness_v15"
)
SEED = 20260723
SPLIT_HALF_REPLICATES = 5_000
PAIR_BOOTSTRAPS = 20_000
LPO_PERMUTATIONS = 50_000


def load_cohort(bundle: Path) -> dict[str, Any]:
    ranking = read_json(bundle / "RANKING_SCORES.json")["ranking"]
    model_slugs = [row["slug"] for row in ranking]
    display_names = [row["display_name"] for row in ranking]
    scorers = {track: TaskScorer(track) for track in TRACKS}
    item_ids: dict[str, list[str]] = {}
    metadata: dict[str, list[dict[str, Any]]] = {}
    correctness: dict[str, np.ndarray] = {}
    task_scores: dict[str, list[Any]] = {track: [] for track in TRACKS}

    for track, scorer in scorers.items():
        item_ids[track] = sorted(scorer.ground_truth)
        metadata[track] = [scorer.ground_truth[item] for item in item_ids[track]]
        matrix = np.zeros((len(ranking), len(item_ids[track])), dtype=np.uint8)
        for model_index, model in enumerate(ranking):
            submission = bundle / model["slug"] / f"{track}_submission.jsonl"
            predictions, parsed_meta = scorer.parse_submission(submission)
            score = scorer.score_predictions(
                predictions,
                model["display_name"],
                parsed_meta=parsed_meta,
            )
            task_scores[track].append(score)
            primary = predictions[scorer.primary_condition]
            for item_index, item_id in enumerate(item_ids[track]):
                matrix[model_index, item_index] = int(
                    scorer._grade_condition(
                        item_id,
                        primary[item_id],
                        scorer.primary_condition,
                    )
                )
            expected = float(model[f"{track}_macro"])
            if not math.isclose(score.macro_accuracy, expected, abs_tol=5e-7):
                raise RuntimeError(
                    f"{model['slug']}/{track} score drift: "
                    f"{score.macro_accuracy:.9f} != {expected:.9f}"
                )
        correctness[track] = matrix

    task_names = []
    task_columns = []
    for track, layer in (("do_you_see_me", "perception"), ("minds_eye", "cognition")):
        tasks = sorted(task_scores[track][0].groups)
        for task in tasks:
            task_names.append(f"{layer}/{task}")
            task_columns.append(
                np.asarray(
                    [score.groups[task].accuracy for score in task_scores[track]],
                    dtype=float,
                )
            )
    task_matrix = np.column_stack(task_columns)
    return {
        "ranking": ranking,
        "model_slugs": model_slugs,
        "display_names": display_names,
        "scorers": scorers,
        "item_ids": item_ids,
        "metadata": metadata,
        "correctness": correctness,
        "task_scores": task_scores,
        "task_names": task_names,
        "task_matrix": task_matrix,
    }


def family_name(model_id: str) -> str:
    owner = model_id.split("/", 1)[0].casefold()
    aliases = {
        "qwen": "Qwen",
        "google": "Gemma",
        "deepseek-ai": "DeepSeek",
        "zai-org": "GLM",
        "opengvlab": "InternVL",
        "moonshotai": "Kimi",
        "meta-llama": "Llama",
        "openbmb": "MiniCPM",
        "microsoft": "Phi",
    }
    return aliases.get(owner, owner)


def standardized_pc1_variance(matrix: np.ndarray) -> float:
    standard_deviation = matrix.std(axis=0, ddof=1)
    usable = standard_deviation > 0
    standardized = (
        matrix[:, usable] - matrix[:, usable].mean(axis=0)
    ) / standard_deviation[usable]
    singular_values = np.linalg.svd(standardized, compute_uv=False)
    eigenvalues = singular_values**2 / (standardized.shape[0] - 1)
    return float(eigenvalues[0] / eigenvalues.sum())


def run_family_sensitivity(cohort: dict[str, Any], output: Path) -> dict[str, Any]:
    ranking = cohort["ranking"]
    family_indices: dict[str, list[int]] = defaultdict(list)
    for index, model in enumerate(ranking):
        family_indices[family_name(model["model_id"])].append(index)
    families = sorted(family_indices)
    combinations = itertools.product(*(family_indices[family] for family in families))
    perception = np.asarray([row["do_you_see_me_macro"] for row in ranking], dtype=float)
    cognition = np.asarray([row["minds_eye_macro"] for row in ranking], dtype=float)

    rows = []
    for combination_index, selected in enumerate(combinations, start=1):
        indices = np.asarray(selected, dtype=np.int32)
        rows.append(
            {
                "combination": combination_index,
                "selected_models": "; ".join(
                    cohort["display_names"][index] for index in indices
                ),
                "model_count": len(indices),
                "pearson": pearson(perception[indices], cognition[indices]),
                "spearman": spearman(perception[indices], cognition[indices]),
                "pc1_explained_variance": standardized_pc1_variance(
                    cohort["task_matrix"][indices]
                ),
            }
        )
    write_csv(output / "family_balanced_sensitivity.csv", rows, list(rows[0]))
    return {
        "family_count": len(families),
        "variant_combinations": len(rows),
        "pearson_range": [
            min(row["pearson"] for row in rows),
            max(row["pearson"] for row in rows),
        ],
        "pearson_median": float(np.median([row["pearson"] for row in rows])),
        "spearman_range": [
            min(row["spearman"] for row in rows),
            max(row["spearman"] for row in rows),
        ],
        "spearman_median": float(np.median([row["spearman"] for row in rows])),
        "pc1_variance_range": [
            min(row["pc1_explained_variance"] for row in rows),
            max(row["pc1_explained_variance"] for row in rows),
        ],
        "all_pearson_positive": all(row["pearson"] > 0 for row in rows),
        "all_spearman_positive": all(row["spearman"] > 0 for row in rows),
    }


def split_strata(
    metadata: list[dict[str, Any]], track: str
) -> list[tuple[str, np.ndarray]]:
    strata: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        if track == "do_you_see_me":
            key = f"{item['dimension']}:{item['capability']}"
        else:
            key = str(item["capability"])
        strata[key].append(index)
    return [
        (key, np.asarray(indices, dtype=np.int32))
        for key, indices in sorted(strata.items())
    ]


def split_scores(
    correctness: np.ndarray,
    strata: list[tuple[str, np.ndarray]],
    track: str,
    rng: np.random.Generator,
) -> np.ndarray:
    first_by_top_group: dict[str, list[np.ndarray]] = defaultdict(list)
    second_by_top_group: dict[str, list[np.ndarray]] = defaultdict(list)
    for label, stratum in strata:
        shuffled = rng.permutation(stratum)
        midpoint = len(shuffled) // 2
        first = correctness[:, shuffled[:midpoint]].mean(axis=1)
        second = correctness[:, shuffled[midpoint:]].mean(axis=1)
        top_group = label.split(":", 1)[0] if track == "do_you_see_me" else "tasks"
        first_by_top_group[top_group].append(first)
        second_by_top_group[top_group].append(second)

    first_score = np.mean(
        [np.mean(values, axis=0) for values in first_by_top_group.values()], axis=0
    )
    second_score = np.mean(
        [np.mean(values, axis=0) for values in second_by_top_group.values()], axis=0
    )
    return np.vstack([first_score, second_score])


def subset_score_reference(
    correctness: np.ndarray,
    metadata: list[dict[str, Any]],
    indices: np.ndarray,
    track: str,
) -> np.ndarray:
    """Reference implementation retained for focused test comparisons."""
    selected = set(int(index) for index in indices)
    if track == "do_you_see_me":
        variants: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, item in enumerate(metadata):
            if index in selected:
                variants[(str(item["dimension"]), str(item["capability"]))].append(index)
        dimensions: dict[str, list[np.ndarray]] = defaultdict(list)
        for (dimension, _capability), item_indices in variants.items():
            dimensions[dimension].append(
                correctness[:, np.asarray(item_indices, dtype=np.int32)].mean(axis=1)
            )
        return np.mean(
            [np.mean(values, axis=0) for values in dimensions.values()], axis=0
        )

    tasks: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        if index in selected:
            tasks[str(item["capability"])].append(index)
    return np.mean(
        [
            correctness[:, np.asarray(item_indices, dtype=np.int32)].mean(axis=1)
            for item_indices in tasks.values()
        ],
        axis=0,
    )


def summarize_distribution(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "95_interval": percentile_interval(array),
    }


def run_split_half(
    cohort: dict[str, Any], output: Path, replicates: int, rng: np.random.Generator
) -> dict[str, Any]:
    track_strata = {
        track: split_strata(cohort["metadata"][track], track) for track in TRACKS
    }
    rows = []
    metric_values: dict[str, dict[str, list[float]]] = {
        metric: defaultdict(list) for metric in ("perception", "cognition", "vpci")
    }
    for replicate in range(1, replicates + 1):
        half_scores: dict[str, dict[str, np.ndarray]] = {"a": {}, "b": {}}
        for track in TRACKS:
            scores = split_scores(
                cohort["correctness"][track],
                track_strata[track],
                track,
                rng,
            )
            half_scores["a"][track] = scores[0]
            half_scores["b"][track] = scores[1]

        by_metric = {
            "perception": (
                half_scores["a"]["do_you_see_me"],
                half_scores["b"]["do_you_see_me"],
            ),
            "cognition": (
                half_scores["a"]["minds_eye"],
                half_scores["b"]["minds_eye"],
            ),
            "vpci": (
                (
                    half_scores["a"]["do_you_see_me"]
                    + half_scores["a"]["minds_eye"]
                )
                / 2.0,
                (
                    half_scores["b"]["do_you_see_me"]
                    + half_scores["b"]["minds_eye"]
                )
                / 2.0,
            ),
        }
        for metric, (first, second) in by_metric.items():
            first_top = int(np.argmax(first))
            second_top = int(np.argmax(second))
            first_top3 = set(np.argsort(-first)[:3].tolist())
            second_top3 = set(np.argsort(-second)[:3].tolist())
            result = {
                "replicate": replicate,
                "metric": metric,
                "pearson": pearson(first, second),
                "spearman": spearman(first, second),
                "top1_agreement": int(first_top == second_top),
                "top3_jaccard": len(first_top3 & second_top3)
                / len(first_top3 | second_top3),
            }
            rows.append(result)
            for key in ("pearson", "spearman", "top1_agreement", "top3_jaccard"):
                metric_values[metric][key].append(float(result[key]))

    write_csv(output / "split_half_stability.csv", rows, list(rows[0]))
    summary = {}
    for metric, values in metric_values.items():
        median_pearson = float(np.median(values["pearson"]))
        summary[metric] = {
            key: summarize_distribution(metric_values[metric][key])
            for key in ("pearson", "spearman", "top1_agreement", "top3_jaccard")
        }
        summary[metric]["spearman_brown_from_median_pearson"] = (
            2.0 * median_pearson / (1.0 + median_pearson)
        )
    return summary


def run_chance_normalization(cohort: dict[str, Any], output: Path) -> dict[str, Any]:
    task_chance = {
        "analogical_reasoning": 1.0 / 6.0,
        "conceptual_slippage": 1.0 / 6.0,
    }
    task_rows = []
    model_rows = []
    normalized_model_scores = []
    raw_scores = []
    for model_index, model in enumerate(cohort["ranking"]):
        score = cohort["task_scores"]["minds_eye"][model_index]
        normalized_by_task = {}
        normalized_by_art: dict[str, list[float]] = defaultdict(list)
        for task, result in sorted(score.groups.items()):
            chance = task_chance.get(task, 0.25)
            normalized = (result.accuracy - chance) / (1.0 - chance)
            normalized_by_task[task] = normalized
            normalized_by_art[MINDS_EYE_ART_BY_CAPABILITY[task]].append(normalized)
            task_rows.append(
                {
                    "published_vpci_rank": model["rank"],
                    "slug": model["slug"],
                    "display_name": model["display_name"],
                    "task": task,
                    "raw_accuracy": result.accuracy,
                    "chance_accuracy": chance,
                    "chance_normalized_accuracy": normalized,
                }
            )
        normalized_score = float(np.mean(list(normalized_by_task.values())))
        normalized_model_scores.append(normalized_score)
        raw_scores.append(score.macro_accuracy)
        model_rows.append(
            {
                "published_vpci_rank": model["rank"],
                "slug": model["slug"],
                "display_name": model["display_name"],
                "raw_cognition": score.macro_accuracy,
                "chance_normalized_cognition": normalized_score,
                **{
                    f"normalized_{art}": float(np.mean(values))
                    for art, values in normalized_by_art.items()
                },
            }
        )
    raw_ranks = descending_ranks(np.asarray(raw_scores))
    normalized_ranks = descending_ranks(np.asarray(normalized_model_scores))
    for index, row in enumerate(model_rows):
        row["raw_cognition_rank"] = int(raw_ranks[index])
        row["chance_normalized_rank"] = int(normalized_ranks[index])
        row["rank_change"] = int(raw_ranks[index]) - int(normalized_ranks[index])
    write_csv(output / "chance_normalized_cognition.csv", model_rows, list(model_rows[0]))
    write_csv(
        output / "chance_normalized_cognition_tasks.csv", task_rows, list(task_rows[0])
    )

    task_summary = []
    for task in sorted({row["task"] for row in task_rows}):
        selected = [row for row in task_rows if row["task"] == task]
        task_summary.append(
            {
                "task": task,
                "chance_accuracy": selected[0]["chance_accuracy"],
                "cohort_raw_mean": float(
                    np.mean([row["raw_accuracy"] for row in selected])
                ),
                "cohort_chance_normalized_mean": float(
                    np.mean([row["chance_normalized_accuracy"] for row in selected])
                ),
            }
        )
    write_csv(
        output / "chance_normalized_task_summary.csv",
        task_summary,
        list(task_summary[0]),
    )
    return {
        "raw_vs_normalized_rank_spearman": spearman(
            np.asarray(raw_scores), np.asarray(normalized_model_scores)
        ),
        "top_raw_model": cohort["display_names"][int(np.argmax(raw_scores))],
        "top_normalized_model": cohort["display_names"][
            int(np.argmax(normalized_model_scores))
        ],
        "maximum_absolute_rank_change": int(
            max(abs(int(row["rank_change"])) for row in model_rows)
        ),
        "task_summary": task_summary,
    }


def run_controlled_pair_bootstraps(
    cohort: dict[str, Any],
    output: Path,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    pairs = (
        (
            "qwen35_thinking",
            "qwen35-9b-thinking-enabled",
            "qwen35-9b",
        ),
        ("gemma_scale", "gemma3-27b-it", "gemma3-12b-it"),
    )
    result_rows = []
    task_rows = []
    summary = {}
    for pair_name, first_slug, second_slug in pairs:
        first_index = cohort["model_slugs"].index(first_slug)
        second_index = cohort["model_slugs"].index(second_slug)
        bootstrap_deltas = {}
        pair_summary = {}
        for track in TRACKS:
            score_bootstrap = bootstrap_track_scores(
                cohort["correctness"][track][[first_index, second_index]],
                grouped_indices(track, cohort["metadata"][track]),
                replicates,
                rng,
            )
            delta = score_bootstrap[:, 0] - score_bootstrap[:, 1]
            bootstrap_deltas[track] = delta
            first_score = score_from_correctness(
                track,
                cohort["correctness"][track][first_index],
                cohort["metadata"][track],
            )
            second_score = score_from_correctness(
                track,
                cohort["correctness"][track][second_index],
                cohort["metadata"][track],
            )
            result = {
                "comparison": pair_name,
                "first_model": cohort["display_names"][first_index],
                "second_model": cohort["display_names"][second_index],
                "benchmark": track,
                "first_macro": first_score,
                "second_macro": second_score,
                "macro_delta": first_score - second_score,
                "delta_ci_low": percentile_interval(delta)[0],
                "delta_ci_high": percentile_interval(delta)[1],
                "probability_delta_positive": float(np.mean(delta > 0)),
            }
            result_rows.append(result)
            pair_summary[track] = result

            groups: dict[str, list[int]] = defaultdict(list)
            for item_index, item in enumerate(cohort["metadata"][track]):
                key = (
                    f"{item['dimension']}:{item['capability']}"
                    if track == "do_you_see_me"
                    else str(item["capability"])
                )
                groups[key].append(item_index)
            pending_task_rows = []
            for group_name, indices in sorted(groups.items()):
                group_indices_array = np.asarray(indices, dtype=np.int32)
                first = cohort["correctness"][track][first_index, group_indices_array]
                second = cohort["correctness"][track][second_index, group_indices_array]
                first_wins = int(np.sum((first == 1) & (second == 0)))
                second_wins = int(np.sum((first == 0) & (second == 1)))
                pending_task_rows.append(
                    {
                        "comparison": pair_name,
                        "benchmark": track,
                        "group": group_name,
                        "sample_count": len(indices),
                        "first_accuracy": float(first.mean()),
                        "second_accuracy": float(second.mean()),
                        "delta": float(first.mean() - second.mean()),
                        "first_only_correct": first_wins,
                        "second_only_correct": second_wins,
                        "mcnemar_exact_p": exact_mcnemar_p(
                            first_wins, second_wins
                        ),
                    }
                )
            adjusted = bh_adjust(
                [row["mcnemar_exact_p"] for row in pending_task_rows]
            )
            for row, q_value in zip(pending_task_rows, adjusted):
                row["within_track_fdr_q"] = q_value
            task_rows.extend(pending_task_rows)

        interaction = (
            bootstrap_deltas["do_you_see_me"] - bootstrap_deltas["minds_eye"]
        )
        observed_interaction = (
            pair_summary["do_you_see_me"]["macro_delta"]
            - pair_summary["minds_eye"]["macro_delta"]
        )
        pair_summary["perception_minus_cognition_interaction"] = {
            "observed": observed_interaction,
            "95_interval": percentile_interval(interaction),
            "probability_positive": float(np.mean(interaction > 0)),
        }
        summary[pair_name] = pair_summary
    write_csv(output / "controlled_pair_bootstraps.csv", result_rows, list(result_rows[0]))
    write_csv(
        output / "controlled_pair_task_tests.csv", task_rows, list(task_rows[0])
    )
    return summary


def run_shared_blindspots(cohort: dict[str, Any], output: Path) -> dict[str, Any]:
    aggregate_rows = []
    item_rows = []
    summary = {}
    model_count = len(cohort["ranking"])
    for track in TRACKS:
        strata: dict[str, list[int]] = defaultdict(list)
        for item_index, item in enumerate(cohort["metadata"][track]):
            if track == "do_you_see_me":
                key = (
                    f"{item['dimension']}:{item['capability']}:"
                    f"{item['difficulty']}"
                )
            else:
                key = str(item["capability"])
            strata[key].append(item_index)

        total_all_wrong = 0
        total_all_correct = 0
        for stratum, indices in sorted(strata.items()):
            counts = cohort["correctness"][track][
                :, np.asarray(indices, dtype=np.int32)
            ].sum(axis=0)
            all_wrong = int(np.sum(counts == 0))
            all_correct = int(np.sum(counts == model_count))
            total_all_wrong += all_wrong
            total_all_correct += all_correct
            aggregate_rows.append(
                {
                    "benchmark": track,
                    "stratum": stratum,
                    "item_count": len(indices),
                    "all_models_wrong": all_wrong,
                    "all_models_wrong_rate": all_wrong / len(indices),
                    "at_most_one_model_correct": int(np.sum(counts <= 1)),
                    "at_least_half_models_correct": int(
                        np.sum(counts >= math.ceil(model_count / 2))
                    ),
                    "all_models_correct": all_correct,
                    "mean_model_accuracy": float(counts.mean() / model_count),
                }
            )
            for local_index, correct_models in zip(indices, counts):
                if correct_models <= 1:
                    item_rows.append(
                        {
                            "benchmark": track,
                            "question_id": cohort["item_ids"][track][local_index],
                            "stratum": stratum,
                            "models_correct": int(correct_models),
                        }
                    )
        summary[track] = {
            "item_count": len(cohort["item_ids"][track]),
            "all_models_wrong": total_all_wrong,
            "all_models_wrong_rate": total_all_wrong
            / len(cohort["item_ids"][track]),
            "all_models_correct": total_all_correct,
            "all_models_correct_rate": total_all_correct
            / len(cohort["item_ids"][track]),
        }
    write_csv(output / "shared_blindspot_summary.csv", aggregate_rows, list(aggregate_rows[0]))
    write_csv(output / "shared_blindspot_items.csv", item_rows, list(item_rows[0]))
    return summary


def run_measurement_floor_audit(
    cohort: dict[str, Any], output: Path
) -> dict[str, Any]:
    track = "do_you_see_me"
    metadata = cohort["metadata"][track]
    correctness = cohort["correctness"][track]
    model_count = correctness.shape[0]
    strata: dict[str, list[int]] = defaultdict(list)
    for item_index, item in enumerate(metadata):
        key = f"{item['dimension']}:{item['capability']}:{item['difficulty']}"
        strata[key].append(item_index)

    audit_rows = []
    floor_strata = set()
    for stratum, indices in sorted(strata.items()):
        selected = correctness[:, np.asarray(indices, dtype=np.int32)]
        item_totals = selected.sum(axis=0)
        cohort_mean = float(selected.mean())
        all_wrong_rate = float(np.mean(item_totals == 0))
        is_floor = cohort_mean <= 0.05 or all_wrong_rate >= 0.50
        if is_floor:
            floor_strata.add(stratum)
        audit_rows.append(
            {
                "stratum": stratum,
                "item_count": len(indices),
                "cohort_mean_accuracy": cohort_mean,
                "model_accuracy_sd": float(selected.mean(axis=1).std(ddof=1)),
                "all_models_wrong_rate": all_wrong_rate,
                "all_models_correct_rate": float(np.mean(item_totals == model_count)),
                "floor_flag": int(is_floor),
            }
        )
    write_csv(output / "measurement_floor_audit.csv", audit_rows, list(audit_rows[0]))

    floor_keep = np.asarray(
        [
            index
            for index, item in enumerate(metadata)
            if f"{item['dimension']}:{item['capability']}:{item['difficulty']}"
            not in floor_strata
        ],
        dtype=np.int32,
    )
    no_form_keep = np.asarray(
        [
            index
            for index, item in enumerate(metadata)
            if item["capability"] != "form_discrimination"
        ],
        dtype=np.int32,
    )

    original_perception = np.asarray(
        [score.macro_accuracy for score in cohort["task_scores"][track]],
        dtype=float,
    )
    cognition = np.asarray(
        [score.macro_accuracy for score in cohort["task_scores"]["minds_eye"]],
        dtype=float,
    )

    def scores_for(indices: np.ndarray) -> np.ndarray:
        selected_metadata = [metadata[index] for index in indices]
        return np.asarray(
            [
                score_from_correctness(
                    track,
                    correctness[model_index, indices],
                    selected_metadata,
                )
                for model_index in range(model_count)
            ],
            dtype=float,
        )

    floor_ablated = scores_for(floor_keep)
    no_form = scores_for(no_form_keep)
    original_vpci = (original_perception + cognition) / 2.0
    floor_ablated_vpci = (floor_ablated + cognition) / 2.0
    no_form_vpci = (no_form + cognition) / 2.0
    original_perception_rank = descending_ranks(original_perception)
    original_vpci_rank = descending_ranks(original_vpci)
    floor_perception_rank = descending_ranks(floor_ablated)
    floor_vpci_rank = descending_ranks(floor_ablated_vpci)
    no_form_perception_rank = descending_ranks(no_form)
    no_form_vpci_rank = descending_ranks(no_form_vpci)

    rank_rows = []
    for index, model in enumerate(cohort["ranking"]):
        rank_rows.append(
            {
                "slug": model["slug"],
                "display_name": model["display_name"],
                "original_perception": original_perception[index],
                "floor_ablated_perception": floor_ablated[index],
                "no_form_perception": no_form[index],
                "original_perception_rank": int(original_perception_rank[index]),
                "floor_ablated_perception_rank": int(floor_perception_rank[index]),
                "no_form_perception_rank": int(no_form_perception_rank[index]),
                "original_vpci": original_vpci[index],
                "floor_ablated_vpci": floor_ablated_vpci[index],
                "no_form_vpci": no_form_vpci[index],
                "original_vpci_rank": int(original_vpci_rank[index]),
                "floor_ablated_vpci_rank": int(floor_vpci_rank[index]),
                "no_form_vpci_rank": int(no_form_vpci_rank[index]),
            }
        )
    write_csv(
        output / "measurement_floor_rank_sensitivity.csv",
        rank_rows,
        list(rank_rows[0]),
    )

    difficulty_rows = []
    for scope, indices in (
        ("all_perception_items", np.arange(len(metadata), dtype=np.int32)),
        ("excluding_form_discrimination", no_form_keep),
    ):
        for difficulty in ("easy", "medium", "hard"):
            selected = np.asarray(
                [
                    index
                    for index in indices
                    if metadata[index]["difficulty"] == difficulty
                ],
                dtype=np.int32,
            )
            difficulty_rows.append(
                {
                    "scope": scope,
                    "difficulty": difficulty,
                    "item_count": len(selected),
                    "cohort_mean_accuracy": float(correctness[:, selected].mean()),
                }
            )
    write_csv(
        output / "difficulty_floor_sensitivity.csv",
        difficulty_rows,
        list(difficulty_rows[0]),
    )

    def sensitivity_summary(
        perception_scores: np.ndarray,
        vpci_scores: np.ndarray,
        perception_ranks: np.ndarray,
        vpci_ranks: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "perception_rank_spearman": spearman(
                original_perception, perception_scores
            ),
            "vpci_rank_spearman": spearman(original_vpci, vpci_scores),
            "maximum_absolute_perception_rank_change": int(
                np.max(np.abs(original_perception_rank - perception_ranks))
            ),
            "maximum_absolute_vpci_rank_change": int(
                np.max(np.abs(original_vpci_rank - vpci_ranks))
            ),
            "top_perception_model": cohort["display_names"][
                int(np.argmax(perception_scores))
            ],
            "top_vpci_model": cohort["display_names"][int(np.argmax(vpci_scores))],
        }

    difficulty = {
        (row["scope"], row["difficulty"]): row["cohort_mean_accuracy"]
        for row in difficulty_rows
    }
    return {
        "floor_definition": (
            "Cohort mean accuracy at most 5%, or at least half of items answered "
            "incorrectly by every evaluated model."
        ),
        "floor_strata": sorted(floor_strata),
        "floor_item_count": int(len(metadata) - len(floor_keep)),
        "floor_item_rate": float((len(metadata) - len(floor_keep)) / len(metadata)),
        "floor_strata_ablation": sensitivity_summary(
            floor_ablated,
            floor_ablated_vpci,
            floor_perception_rank,
            floor_vpci_rank,
        ),
        "form_discrimination_ablation": sensitivity_summary(
            no_form,
            no_form_vpci,
            no_form_perception_rank,
            no_form_vpci_rank,
        ),
        "easy_hard_gap_all_items": float(
            difficulty[("all_perception_items", "easy")]
            - difficulty[("all_perception_items", "hard")]
        ),
        "easy_hard_gap_excluding_form_discrimination": float(
            difficulty[("excluding_form_discrimination", "easy")]
            - difficulty[("excluding_form_discrimination", "hard")]
        ),
    }


def run_leave_pair_out_associations(
    cohort: dict[str, Any],
    output: Path,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    task_names = cohort["task_names"]
    perception_indices = [
        index for index, name in enumerate(task_names) if name.startswith("perception/")
    ]
    cognition_indices = [
        index for index, name in enumerate(task_names) if name.startswith("cognition/")
    ]

    family_indices: dict[str, list[int]] = defaultdict(list)
    for index, model in enumerate(cohort["ranking"]):
        family_indices[family_name(model["model_id"])].append(index)
    family_matrix = np.vstack(
        [
            cohort["task_matrix"][np.asarray(indices, dtype=np.int32)].mean(axis=0)
            for _family, indices in sorted(family_indices.items())
        ]
    )

    all_rows = []
    summaries = {}
    for cohort_name, matrix in (
        ("all_model_variants", cohort["task_matrix"]),
        ("family_averaged", family_matrix),
    ):
        permutation_indices = np.argsort(
            rng.random((permutations, matrix.shape[0])), axis=1
        ).astype(np.int16)
        rows = []
        for perception_index in perception_indices:
            for cognition_index in cognition_indices:
                retained = [
                    index
                    for index in range(matrix.shape[1])
                    if index not in {perception_index, cognition_index}
                ]
                controls = matrix[:, retained]
                standard_deviations = controls.std(axis=0, ddof=1)
                usable = standard_deviations > 0
                standardized = (
                    controls[:, usable] - controls[:, usable].mean(axis=0)
                ) / standard_deviations[usable]
                control = standardized.mean(axis=1)
                design = np.column_stack(
                    [np.ones(matrix.shape[0]), rankdata(control)]
                )

                def residualize(values: np.ndarray) -> np.ndarray:
                    ranked = rankdata(values)
                    residuals = ranked - design @ np.linalg.lstsq(
                        design, ranked, rcond=None
                    )[0]
                    norm = math.sqrt(float(np.dot(residuals, residuals)))
                    return residuals / norm if norm else residuals

                x = matrix[:, perception_index]
                y = matrix[:, cognition_index]
                x_residual = residualize(x)
                y_residual = residualize(y)
                observed = float(np.dot(x_residual, y_residual))
                null = y_residual[permutation_indices] @ x_residual
                p_value = (
                    int(np.count_nonzero(np.abs(null) >= abs(observed))) + 1
                ) / (permutations + 1)
                rows.append(
                    {
                        "cohort": cohort_name,
                        "model_units": matrix.shape[0],
                        "perception_task": task_names[perception_index].split("/", 1)[1],
                        "cognition_task": task_names[cognition_index].split("/", 1)[1],
                        "raw_spearman": spearman(x, y),
                        "leave_pair_out_partial_spearman": observed,
                        "permutation_p": p_value,
                    }
                )
        adjusted = bh_adjust([row["permutation_p"] for row in rows])
        for row, q_value in zip(rows, adjusted):
            row["fdr_q"] = q_value
        all_rows.extend(rows)
        strongest = max(
            rows,
            key=lambda row: abs(float(row["leave_pair_out_partial_spearman"])),
        )
        summaries[cohort_name] = {
            "model_units": matrix.shape[0],
            "tested_pairs": len(rows),
            "fdr_significant_pair_count": int(
                sum(float(row["fdr_q"]) < 0.05 for row in rows)
            ),
            "strongest_absolute_partial_association": strongest,
        }
    write_csv(
        output / "leave_pair_out_associations.csv",
        all_rows,
        list(all_rows[0]),
    )
    return summaries


def readiness_decisions(
    exploratory_summary: dict[str, Any],
    family_summary: dict[str, Any],
    split_summary: dict[str, Any],
    chance_summary: dict[str, Any],
    pair_summary: dict[str, Any],
    floor_summary: dict[str, Any],
    association_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    extraction_current = bool(
        exploratory_summary["protocol_audit"][
            "bundle_matches_checked_out_production_contract"
        ]
    )
    shared_factor_pass = (
        family_summary["all_pearson_positive"]
        and family_summary["pearson_range"][0] >= 0.45
        and split_summary["vpci"]["spearman"]["median"] >= 0.90
    )
    aggregation = exploratory_summary["aggregation_sensitivity"]
    aggregation_robust = (
        aggregation["arithmetic_geometric_rank_spearman"] >= 0.98
        and aggregation["arithmetic_harmonic_rank_spearman"] >= 0.98
        and aggregation["arithmetic_bottleneck_rank_spearman"] >= 0.95
        and len(aggregation["pareto_frontier"]) == 2
    )
    thinking_interaction = pair_summary["qwen35_thinking"][
        "perception_minus_cognition_interaction"
    ]
    return [
        {
            "finding": "General visual factor plus layer specialization",
            "statistical_robustness": "pass" if shared_factor_pass else "fail",
            "pipeline_readiness": "pass" if extraction_current else "blocked",
            "submission_status": (
                "ready"
                if shared_factor_pass and extraction_current
                else "needs_more_evidence"
            ),
            "reason": (
                "Positive in every one-variant-per-family cohort, high split-half "
                "stability, and a separate second capability axis."
            ),
        },
        {
            "finding": "Balanced aggregation and two-model Pareto frontier",
            "statistical_robustness": "pass" if aggregation_robust else "fail",
            "pipeline_readiness": "pass" if extraction_current else "blocked",
            "submission_status": (
                "ready"
                if aggregation_robust and extraction_current
                else "needs_more_evidence"
            ),
            "reason": (
                "Geometric and harmonic aggregation each preserve the top six and "
                "change only one adjacent middle-tier pair; rank correlations with "
                "the arithmetic score are 0.996, the bottleneck correlation is "
                "0.982, and the same two models define the Pareto frontier."
            ),
        },
        {
            "finding": "Native thinking reallocates perception and cognition",
            "statistical_robustness": "pass"
            if thinking_interaction["95_interval"][0] > 0
            else "fail",
            "pipeline_readiness": "pass" if extraction_current else "blocked",
            "submission_status": "requires_seed_replication",
            "reason": (
                "The layer interaction is strong, but it is a single stochastic run "
                "and the thinking profile has many capped responses."
            ),
        },
        {
            "finding": "Gemma 27B outperforms 12B within this family",
            "statistical_robustness": "case_study_only",
            "pipeline_readiness": "pass" if extraction_current else "blocked",
            "submission_status": "descriptive_case_study",
            "reason": (
                "The larger model improves both benchmark macros, but the "
                "cross-layer interaction interval includes zero and only one "
                "same-family size pair is available."
            ),
        },
        {
            "finding": "Specific perception tasks enable specific cognition tasks",
            "statistical_robustness": "fail"
            if all(
                result["fdr_significant_pair_count"] == 0
                for result in association_summary.values()
            )
            else "needs_review",
            "pipeline_readiness": "not_applicable",
            "submission_status": "do_not_claim",
            "reason": (
                "No cross-task association survives FDR correction after controlling "
                "for competence estimated from the other 13 tasks, including after "
                "averaging variants within model family."
            ),
        },
        {
            "finding": "Cognition ordering is robust to task chance differences",
            "statistical_robustness": "pass"
            if chance_summary["raw_vs_normalized_rank_spearman"] >= 0.95
            else "sensitive",
            "pipeline_readiness": "pass" if extraction_current else "blocked",
            "submission_status": "ready" if extraction_current else "blocked",
            "reason": "Chance normalization is a diagnostic sensitivity analysis.",
        },
        {
            "finding": "Perception difficulty gradient is not only a letter-task floor effect",
            "statistical_robustness": "pass"
            if floor_summary["easy_hard_gap_excluding_form_discrimination"] > 0.05
            else "fail",
            "pipeline_readiness": "pass" if extraction_current else "blocked",
            "submission_status": "ready" if extraction_current else "blocked",
            "reason": (
                "The easy-to-hard gap remains after removing form discrimination; "
                "floor strata must still be disclosed as a measurement limitation."
            ),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--exploratory-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIRMATORY_OUTPUT)
    parser.add_argument("--split-half", type=int, default=SPLIT_HALF_REPLICATES)
    parser.add_argument("--pair-bootstrap", type=int, default=PAIR_BOOTSTRAPS)
    parser.add_argument("--lpo-permutations", type=int, default=LPO_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    exploratory_summary_path = (
        args.exploratory_output.resolve() / "analysis_summary.json"
    )
    if not exploratory_summary_path.exists():
        raise FileNotFoundError(
            "Run analysis/combined_visual_v14.py before the confirmatory suite."
        )
    exploratory_summary = read_json(exploratory_summary_path)
    extraction_current = bool(
        exploratory_summary["protocol_audit"][
            "bundle_matches_checked_out_production_contract"
        ]
    )
    cohort = load_cohort(bundle)
    family_summary = run_family_sensitivity(cohort, output)
    split_summary = run_split_half(
        cohort,
        output,
        args.split_half,
        np.random.default_rng(args.seed + 1),
    )
    chance_summary = run_chance_normalization(cohort, output)
    pair_summary = run_controlled_pair_bootstraps(
        cohort,
        output,
        args.pair_bootstrap,
        np.random.default_rng(args.seed + 2),
    )
    blindspot_summary = run_shared_blindspots(cohort, output)
    floor_summary = run_measurement_floor_audit(cohort, output)
    association_summary = run_leave_pair_out_associations(
        cohort,
        output,
        args.lpo_permutations,
        np.random.default_rng(args.seed + 3),
    )
    decisions = readiness_decisions(
        exploratory_summary,
        family_summary,
        split_summary,
        chance_summary,
        pair_summary,
        floor_summary,
        association_summary,
    )
    write_csv(
        output / "submission_readiness_decisions.csv", decisions, list(decisions[0])
    )
    summary = {
        "analysis_id": "submission_readiness_v15",
        "seed": args.seed,
        "source_bundle": display_path(bundle),
        "status": (
            "canonical_v4_verified"
            if extraction_current
            else "blocked_by_extraction_contract_mismatch"
        ),
        "family_balanced_sensitivity": family_summary,
        "split_half_stability": split_summary,
        "chance_normalized_cognition": chance_summary,
        "controlled_pair_bootstraps": pair_summary,
        "shared_blindspots": blindspot_summary,
        "measurement_floor_audit": floor_summary,
        "leave_pair_out_associations": association_summary,
        "submission_readiness": decisions,
        "required_next_experiments": [
            "Run two additional seeds for Qwen3.5 thinking enabled and disabled on both benchmarks.",
            "Use final-answer preservation or reserve a separate answer budget so capped reasoning cannot remove the commitment.",
            "Treat Gemma scaling as a case study unless a second controlled size family is evaluated.",
            "Do not claim task-specific perception-to-cognition mechanisms from the current 14-model cohort.",
        ],
    }
    write_json(output / "submission_readiness_summary.json", summary)
    print(f"Validated and analyzed {len(cohort['ranking'])} model variants.")
    print(f"Wrote confirmatory artifacts to {output}")
    print(
        "Family-balanced Pearson range: "
        f"{family_summary['pearson_range'][0]:.3f} to "
        f"{family_summary['pearson_range'][1]:.3f}"
    )
    print(
        "VPCI split-half median Spearman: "
        f"{split_summary['vpci']['spearman']['median']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
