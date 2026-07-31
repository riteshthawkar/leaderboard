#!/usr/bin/env python3
"""Reproducible combined analysis for the canonical 14-model visual cohort.

The script intentionally calls the production ``TaskScorer`` so the research
tables cannot drift from leaderboard grading. It reads canonical artifacts and
writes derived CSV/JSON files only; source submissions and ground truth are
never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import MINDS_EYE_ART_BY_CAPABILITY  # noqa: E402
from evaluation.extract_canonical_answers import (  # noqa: E402
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    extractor_contract_sha256,
)
from scoring.task_scorer import TaskScorer  # noqa: E402
from visual_answer_contract import PRODUCTION_EXTRACTION_METHOD  # noqa: E402


DEFAULT_BUNDLE = (
    PROJECT_ROOT / "evaluation/results/ms-vista-ranking-final-evidence-v4-v15"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis/results/combined_visual_v15"
TRACKS = ("do_you_see_me", "minds_eye")
UNRESOLVED = "UNRESOLVED"
BOOTSTRAP_REPLICATES = 10_000
CORRELATION_BOOTSTRAPS = 50_000
PERMUTATIONS = 100_000
SEED = 20260722


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ascending, with deterministic tie handling."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0 + 1.0
        cursor = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(np.asarray(x)), rankdata(np.asarray(y)))


def exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    """Two-sided exact McNemar p-value using a stable binomial tail."""
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    tail = min(discordant_a, discordant_b)
    log_two = math.log(2.0)
    probability = 0.0
    for k in range(tail + 1):
        log_probability = (
            math.lgamma(total + 1)
            - math.lgamma(k + 1)
            - math.lgamma(total - k + 1)
            - total * log_two
        )
        probability += math.exp(log_probability)
    return min(1.0, 2.0 * probability)


def bh_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    count = len(values)
    for position in range(count - 1, -1, -1):
        index = order[position]
        candidate = values[index] * count / (position + 1)
        running = min(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def descending_ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(-np.asarray(values, dtype=float))


def score_from_correctness(
    track: str,
    correctness: np.ndarray,
    metadata: list[dict[str, Any]],
) -> float:
    if track == "do_you_see_me":
        dimension_values: dict[str, list[float]] = defaultdict(list)
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, info in enumerate(metadata):
            groups[(str(info["dimension"]), str(info["capability"]))].append(index)
        for (dimension, _capability), indices in groups.items():
            dimension_values[dimension].append(float(correctness[indices].mean()))
        return float(np.mean([np.mean(values) for values in dimension_values.values()]))

    groups: dict[str, list[int]] = defaultdict(list)
    for index, info in enumerate(metadata):
        groups[str(info["capability"])].append(index)
    return float(np.mean([correctness[indices].mean() for indices in groups.values()]))


def grouped_indices(
    track: str, metadata: list[dict[str, Any]]
) -> dict[str, list[np.ndarray]]:
    if track == "do_you_see_me":
        variants: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, info in enumerate(metadata):
            variants[(str(info["dimension"]), str(info["capability"]))].append(index)
        dimensions: dict[str, list[np.ndarray]] = defaultdict(list)
        for (dimension, capability), indices in sorted(variants.items()):
            del capability
            dimensions[dimension].append(np.asarray(indices, dtype=np.int32))
        return dict(dimensions)

    tasks: dict[str, list[int]] = defaultdict(list)
    for index, info in enumerate(metadata):
        tasks[str(info["capability"])].append(index)
    return {"tasks": [np.asarray(v, dtype=np.int32) for _, v in sorted(tasks.items())]}


def bootstrap_track_scores(
    correctness: np.ndarray,
    groups: dict[str, list[np.ndarray]],
    replicates: int,
    rng: np.random.Generator,
    batch_size: int = 250,
) -> np.ndarray:
    """Shared stratified item bootstrap, preserving the ranking aggregation."""
    model_count = correctness.shape[0]
    output = np.zeros((replicates, model_count), dtype=np.float64)
    top_groups = list(groups.values())
    for top_group in top_groups:
        top_scores = np.zeros_like(output)
        for indices in top_group:
            values = correctness[:, indices]
            for start in range(0, replicates, batch_size):
                stop = min(start + batch_size, replicates)
                sampled = rng.integers(
                    0,
                    values.shape[1],
                    size=(stop - start, values.shape[1]),
                    dtype=np.int32,
                )
                top_scores[start:stop] += values[:, sampled].mean(axis=2).T
        output += top_scores / len(top_group)
    return output / len(top_groups)


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    estimates: list[float] = []
    n = len(x)
    for _ in range(replicates):
        indices = rng.integers(0, n, size=n)
        estimate = pearson(x[indices], y[indices])
        if math.isfinite(estimate):
            estimates.append(estimate)
    return tuple(float(v) for v in np.percentile(estimates, [2.5, 97.5]))


def permutation_p_values(
    perception: np.ndarray,
    cognition: np.ndarray,
    control: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    model_count = perception.shape[0]
    permutation_indices = np.argsort(
        rng.random((permutations, model_count)), axis=1
    ).astype(np.int16)
    control_rank = rankdata(control)
    design = np.column_stack([np.ones(model_count), control_rank])

    def residualize(values: np.ndarray) -> np.ndarray:
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        residuals = values - design @ coefficients
        norm = math.sqrt(float(np.dot(residuals, residuals)))
        return residuals / norm if norm else residuals

    rows: list[dict[str, Any]] = []
    for perception_index in range(perception.shape[1]):
        x = rankdata(perception[:, perception_index])
        x = (x - x.mean()) / math.sqrt(float(np.dot(x - x.mean(), x - x.mean())))
        partial_x = residualize(rankdata(perception[:, perception_index]))
        for cognition_index in range(cognition.shape[1]):
            y = rankdata(cognition[:, cognition_index])
            y = y - y.mean()
            y = y / math.sqrt(float(np.dot(y, y)))
            observed = float(np.dot(x, y))
            null = y[permutation_indices] @ x
            p_value = (int(np.count_nonzero(np.abs(null) >= abs(observed))) + 1) / (
                permutations + 1
            )
            partial_y = residualize(rankdata(cognition[:, cognition_index]))
            partial_observed = float(np.dot(partial_x, partial_y))
            partial_null = partial_y[permutation_indices] @ partial_x
            partial_p = (
                int(np.count_nonzero(np.abs(partial_null) >= abs(partial_observed))) + 1
            ) / (permutations + 1)
            rows.append(
                {
                    "perception_index": perception_index,
                    "cognition_index": cognition_index,
                    "spearman_r": observed,
                    "permutation_p": p_value,
                    "partial_spearman_controlling_vpci": partial_observed,
                    "partial_permutation_p": partial_p,
                }
            )
    adjusted = bh_adjust([row["permutation_p"] for row in rows])
    partial_adjusted = bh_adjust([row["partial_permutation_p"] for row in rows])
    for row, q_value, partial_q in zip(rows, adjusted, partial_adjusted):
        row["fdr_q"] = q_value
        row["partial_fdr_q"] = partial_q
    return rows


def pca(task_matrix: np.ndarray, task_names: list[str]) -> tuple[list[dict], list[dict]]:
    standardized = (task_matrix - task_matrix.mean(axis=0)) / task_matrix.std(
        axis=0, ddof=1
    )
    u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    eigenvalues = singular_values**2 / (standardized.shape[0] - 1)
    explained = eigenvalues / eigenvalues.sum()

    # Stable orientation: PC1 points toward general competence; PC2 toward
    # mental rotation, the clearest procedural-spatial loading in this cohort.
    if vt[0].mean() < 0:
        vt[0] *= -1
        u[:, 0] *= -1
    mental_rotation_index = task_names.index("cognition/mental_rotation")
    if vt[1, mental_rotation_index] < 0:
        vt[1] *= -1
        u[:, 1] *= -1

    scores = u * singular_values
    loading_rows = []
    for component in range(vt.shape[0]):
        for task_index, task_name in enumerate(task_names):
            loading_rows.append(
                {
                    "component": component + 1,
                    "explained_variance": float(explained[component]),
                    "task": task_name,
                    "loading": float(vt[component, task_index]),
                }
            )
    score_rows = [
        {
            "model_index": model_index,
            **{
                f"pc{component + 1}": float(scores[model_index, component])
                for component in range(scores.shape[1])
            },
        }
        for model_index in range(scores.shape[0])
    ]
    return loading_rows, score_rows


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--correlation-bootstrap", type=int, default=CORRELATION_BOOTSTRAPS
    )
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ranking_document = read_json(bundle / "RANKING_SCORES.json")
    bundle_manifest = read_json(bundle / "BUNDLE_MANIFEST.json")
    index_document = read_json(bundle / "index.json")
    ranking = ranking_document["ranking"]
    model_slugs = [row["slug"] for row in ranking]
    display_names = [row["display_name"] for row in ranking]
    model_count = len(model_slugs)

    snapshot_path = (
        PROJECT_ROOT / "frontend/src/data/snapshot/leaderboard-visual-cognition.json"
    )
    snapshot_meta: dict[str, dict[str, Any]] = {}
    if snapshot_path.exists():
        snapshot = read_json(snapshot_path)
        snapshot_meta = {
            row["model_name"]: row.get("model_meta", {})
            for row in snapshot.get("leaderboard", [])
        }

    manifest_by_slug = {row["slug"]: row for row in bundle_manifest["models"]}
    index_by_slug = {row["slug"]: row for row in index_document["models"]}
    scorers = {track: TaskScorer(track) for track in TRACKS}
    metadata: dict[str, list[dict[str, Any]]] = {}
    ids: dict[str, list[str]] = {}
    correctness: dict[str, np.ndarray] = {}
    unresolved: dict[str, np.ndarray] = {}
    task_scores: dict[str, list[Any]] = {track: [] for track in TRACKS}
    predictions_by_track: dict[str, list[dict[str, str]]] = {
        track: [] for track in TRACKS
    }
    validation_errors: list[str] = []

    for track, scorer in scorers.items():
        ids[track] = sorted(scorer.ground_truth)
        metadata[track] = [scorer.ground_truth[sid] for sid in ids[track]]
        track_correctness = np.zeros((model_count, len(ids[track])), dtype=np.uint8)
        track_unresolved = np.zeros_like(track_correctness)
        for model_index, row in enumerate(ranking):
            slug = row["slug"]
            submission = bundle / slug / f"{track}_submission.jsonl"
            expected_hash = manifest_by_slug[slug]["tracks"][track]["submission_sha256"]
            actual_hash = sha256(submission)
            if expected_hash != actual_hash:
                validation_errors.append(
                    f"{slug}/{track}: SHA-256 mismatch ({actual_hash} != {expected_hash})"
                )
            parsed, parsed_meta = scorer.parse_submission(submission)
            primary = parsed[scorer.primary_condition]
            predictions_by_track[track].append(primary)
            score = scorer.score_predictions(
                parsed,
                row["display_name"],
                parsed_meta=parsed_meta,
            )
            task_scores[track].append(score)
            expected_score = float(row[f"{track}_macro"])
            if not math.isclose(score.macro_accuracy, expected_score, abs_tol=5e-7):
                validation_errors.append(
                    f"{slug}/{track}: macro score {score.macro_accuracy:.9f} "
                    f"does not match ranking {expected_score:.9f}"
                )
            for item_index, sample_id in enumerate(ids[track]):
                answer = primary[sample_id]
                track_unresolved[model_index, item_index] = int(answer == UNRESOLVED)
                track_correctness[model_index, item_index] = int(
                    scorer._grade_condition(sample_id, answer, scorer.primary_condition)
                )
        correctness[track] = track_correctness
        unresolved[track] = track_unresolved

    expected_rows = model_count * sum(len(ids[track]) for track in TRACKS)
    observed_rows = sum(matrix.size for matrix in correctness.values())
    observed_unresolved = sum(int(matrix.sum()) for matrix in unresolved.values())
    if observed_rows != expected_rows:
        validation_errors.append(
            f"Observed {observed_rows} responses; expected {expected_rows}."
        )
    if observed_rows != int(bundle_manifest["coverage"]["total_rows"]):
        validation_errors.append("Response count differs from BUNDLE_MANIFEST.json.")
    if observed_unresolved != int(
        bundle_manifest["coverage"]["unresolved_answer_count"]
    ):
        validation_errors.append("UNRESOLVED count differs from BUNDLE_MANIFEST.json.")
    if validation_errors:
        raise RuntimeError("Artifact validation failed:\n- " + "\n- ".join(validation_errors))

    # Aggregate model table through the same production score objects.
    cohort_rows: list[dict[str, Any]] = []
    for model_index, row in enumerate(ranking):
        perception_score = task_scores["do_you_see_me"][model_index]
        cognition_score = task_scores["minds_eye"][model_index]
        perception = float(perception_score.macro_accuracy)
        cognition = float(cognition_score.macro_accuracy)
        model_meta = snapshot_meta.get(row["display_name"], {})
        cohort_rows.append(
            {
                "rank": row["rank"],
                "slug": row["slug"],
                "display_name": row["display_name"],
                "model_id": row["model_id"],
                "organization": model_meta.get("organization")
                or model_meta.get("org")
                or "",
                "parameter_count": model_meta.get("parameter_count", ""),
                "reasoning_profile": row["reasoning_profile"],
                "perception_macro": perception,
                "cognition_macro": cognition,
                "vpci": (perception + cognition) / 2.0,
                "perception_minus_cognition": perception - cognition,
                "perception_rank": int(descending_ranks(
                    np.asarray([r["do_you_see_me_macro"] for r in ranking])
                )[model_index]),
                "cognition_rank": int(descending_ranks(
                    np.asarray([r["minds_eye_macro"] for r in ranking])
                )[model_index]),
                "perception_2d": perception_score.analysis["dimension"]["2D"].accuracy,
                "perception_3d": perception_score.analysis["dimension"]["3D"].accuracy,
                "perception_easy": perception_score.analysis["difficulty"]["easy"].accuracy,
                "perception_medium": perception_score.analysis["difficulty"]["medium"].accuracy,
                "perception_hard": perception_score.analysis["difficulty"]["hard"].accuracy,
                "perception_easy_hard_drop": (
                    perception_score.analysis["difficulty"]["easy"].accuracy
                    - perception_score.analysis["difficulty"]["hard"].accuracy
                ),
                "perception_unresolved": int(unresolved["do_you_see_me"][model_index].sum()),
                "cognition_unresolved": int(unresolved["minds_eye"][model_index].sum()),
            }
        )

    write_csv(
        output / "cohort_scores.csv",
        cohort_rows,
        list(cohort_rows[0]),
    )

    # Long task-profile table and ART summaries.
    task_profile_rows: list[dict[str, Any]] = []
    art_rows: list[dict[str, Any]] = []
    perception_tasks = sorted(task_scores["do_you_see_me"][0].groups)
    cognition_tasks = sorted(task_scores["minds_eye"][0].groups)
    task_names = [f"perception/{name}" for name in perception_tasks] + [
        f"cognition/{name}" for name in cognition_tasks
    ]
    task_matrix = np.zeros((model_count, len(task_names)), dtype=float)
    for model_index, row in enumerate(ranking):
        column = 0
        for track, layer, tasks in (
            ("do_you_see_me", "perception", perception_tasks),
            ("minds_eye", "cognition", cognition_tasks),
        ):
            for task in tasks:
                result = task_scores[track][model_index].groups[task]
                task_matrix[model_index, column] = result.accuracy
                task_profile_rows.append(
                    {
                        "rank": row["rank"],
                        "slug": row["slug"],
                        "display_name": row["display_name"],
                        "benchmark": track,
                        "layer": layer,
                        "task": task,
                        "correct": result.correct_samples,
                        "total": result.total_samples,
                        "accuracy": result.accuracy,
                    }
                )
                column += 1

        cognition_group_scores = task_scores["minds_eye"][model_index].groups
        art_values: dict[str, list[float]] = defaultdict(list)
        for capability, result in cognition_group_scores.items():
            art_values[MINDS_EYE_ART_BY_CAPABILITY[capability]].append(result.accuracy)
        art_rows.append(
            {
                "rank": row["rank"],
                "slug": row["slug"],
                "display_name": row["display_name"],
                **{axis: float(np.mean(values)) for axis, values in art_values.items()},
            }
        )

    write_csv(output / "task_profiles.csv", task_profile_rows, list(task_profile_rows[0]))
    write_csv(output / "art_profiles.csv", art_rows, list(art_rows[0]))

    task_summary_rows: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(task_names):
        values = task_matrix[:, task_index]
        winner_indices = np.flatnonzero(values == values.max())
        task_summary_rows.append(
            {
                "task": task_name,
                "cohort_mean": float(values.mean()),
                "cohort_sd": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "winner": "; ".join(display_names[index] for index in winner_indices),
                "winner_accuracy": float(values.max()),
            }
        )
    task_summary_rows.sort(key=lambda row: row["cohort_mean"], reverse=True)
    write_csv(output / "task_summary.csv", task_summary_rows, list(task_summary_rows[0]))

    # Descriptive specialization residuals: task performance above or below a
    # linear expectation from overall VPCI. These are exploratory, not causal.
    vpci_for_residuals = np.asarray([float(row["vpci"]) for row in ranking])
    residual_design = np.column_stack(
        [np.ones(model_count), vpci_for_residuals]
    )
    specialization_rows: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(task_names):
        values = task_matrix[:, task_index]
        expected = residual_design @ np.linalg.lstsq(
            residual_design, values, rcond=None
        )[0]
        for model_index, value in enumerate(values):
            specialization_rows.append(
                {
                    "rank": ranking[model_index]["rank"],
                    "slug": model_slugs[model_index],
                    "display_name": display_names[model_index],
                    "task": task_name,
                    "accuracy": float(value),
                    "expected_from_vpci": float(expected[model_index]),
                    "specialization_residual": float(
                        value - expected[model_index]
                    ),
                }
            )
    specialization_rows.sort(
        key=lambda row: row["specialization_residual"], reverse=True
    )
    write_csv(
        output / "specialization_residuals.csv",
        specialization_rows,
        list(specialization_rows[0]),
    )

    loading_rows, pca_score_rows = pca(task_matrix, task_names)
    for row in pca_score_rows:
        model_index = row.pop("model_index")
        row.update(
            {
                "rank": ranking[model_index]["rank"],
                "slug": model_slugs[model_index],
                "display_name": display_names[model_index],
            }
        )
    write_csv(output / "pca_loadings.csv", loading_rows, list(loading_rows[0]))
    write_csv(output / "pca_model_scores.csv", pca_score_rows, list(pca_score_rows[0]))

    # Cross-layer task associations with a shared permutation null and BH FDR.
    perception_matrix = task_matrix[:, : len(perception_tasks)]
    cognition_matrix = task_matrix[:, len(perception_tasks) :]
    association_rows = permutation_p_values(
        perception_matrix,
        cognition_matrix,
        np.asarray([row["vpci"] for row in ranking], dtype=float),
        args.permutations,
        np.random.default_rng(args.seed + 1),
    )
    for row in association_rows:
        row["perception_task"] = perception_tasks[row["perception_index"]]
        row["cognition_task"] = cognition_tasks[row["cognition_index"]]
    for row in association_rows:
        row.pop("perception_index")
        row.pop("cognition_index")
    association_rows.sort(key=lambda row: (row["fdr_q"], -abs(row["spearman_r"])))
    write_csv(
        output / "cross_task_associations.csv",
        association_rows,
        [
            "perception_task",
            "cognition_task",
            "spearman_r",
            "permutation_p",
            "fdr_q",
            "partial_spearman_controlling_vpci",
            "partial_permutation_p",
            "partial_fdr_q",
        ],
    )

    # Item-stratified bootstrap of leaderboard scores and ranks.
    bootstrap_rng = np.random.default_rng(args.seed + 2)
    perception_bootstrap = bootstrap_track_scores(
        correctness["do_you_see_me"],
        grouped_indices("do_you_see_me", metadata["do_you_see_me"]),
        args.bootstrap,
        bootstrap_rng,
    )
    cognition_bootstrap = bootstrap_track_scores(
        correctness["minds_eye"],
        grouped_indices("minds_eye", metadata["minds_eye"]),
        args.bootstrap,
        bootstrap_rng,
    )
    vpci_bootstrap = (perception_bootstrap + cognition_bootstrap) / 2.0
    rank_bootstrap = np.empty_like(vpci_bootstrap)
    for replicate in range(args.bootstrap):
        rank_bootstrap[replicate] = descending_ranks(vpci_bootstrap[replicate])
    rank_rows: list[dict[str, Any]] = []
    for model_index, row in enumerate(ranking):
        rank_rows.append(
            {
                "rank": row["rank"],
                "slug": row["slug"],
                "display_name": row["display_name"],
                "vpci": row["vpci"],
                "vpci_ci_low": percentile_interval(vpci_bootstrap[:, model_index])[0],
                "vpci_ci_high": percentile_interval(vpci_bootstrap[:, model_index])[1],
                "mean_bootstrap_rank": float(rank_bootstrap[:, model_index].mean()),
                "rank_ci_low": percentile_interval(rank_bootstrap[:, model_index])[0],
                "rank_ci_high": percentile_interval(rank_bootstrap[:, model_index])[1],
                "probability_rank_1": float(
                    np.mean(rank_bootstrap[:, model_index] == 1)
                ),
                "probability_top_3": float(
                    np.mean(rank_bootstrap[:, model_index] <= 3)
                ),
            }
        )
    write_csv(output / "rank_uncertainty.csv", rank_rows, list(rank_rows[0]))

    adjacent_rows: list[dict[str, Any]] = []
    for upper in range(model_count - 1):
        lower = upper + 1
        difference = vpci_bootstrap[:, upper] - vpci_bootstrap[:, lower]
        adjacent_rows.append(
            {
                "higher_rank": ranking[upper]["rank"],
                "higher_model": display_names[upper],
                "lower_rank": ranking[lower]["rank"],
                "lower_model": display_names[lower],
                "observed_difference": ranking[upper]["vpci"] - ranking[lower]["vpci"],
                "difference_ci_low": percentile_interval(difference)[0],
                "difference_ci_high": percentile_interval(difference)[1],
                "probability_higher_model_wins": float(np.mean(difference > 0)),
            }
        )
    write_csv(
        output / "adjacent_rank_comparisons.csv",
        adjacent_rows,
        list(adjacent_rows[0]),
    )

    # Paired comparisons that have a defensible within-family control.
    paired_rows: list[dict[str, Any]] = []
    controlled_pairs = (
        (
            "qwen35_thinking",
            "qwen35-9b-thinking-enabled",
            "qwen35-9b",
        ),
        ("gemma_scale", "gemma3-27b-it", "gemma3-12b-it"),
    )
    for comparison, first_slug, second_slug in controlled_pairs:
        first_index = model_slugs.index(first_slug)
        second_index = model_slugs.index(second_slug)
        for track in TRACKS:
            if track == "do_you_see_me":
                group_labels = [
                    f"{info['dimension']}:{info['capability']}" for info in metadata[track]
                ]
            else:
                group_labels = [str(info["capability"]) for info in metadata[track]]
            groups: dict[str, list[int]] = defaultdict(list)
            for item_index, label in enumerate(group_labels):
                groups[label].append(item_index)
            for group_name, indices_list in [("overall", list(range(len(group_labels))))] + sorted(
                groups.items()
            ):
                indices_array = np.asarray(indices_list, dtype=np.int32)
                first = correctness[track][first_index, indices_array]
                second = correctness[track][second_index, indices_array]
                first_wins = int(np.sum((first == 1) & (second == 0)))
                second_wins = int(np.sum((first == 0) & (second == 1)))
                paired_rows.append(
                    {
                        "comparison": comparison,
                        "first_model": display_names[first_index],
                        "second_model": display_names[second_index],
                        "benchmark": track,
                        "group": group_name,
                        "sample_count": len(indices_array),
                        "first_accuracy": float(first.mean()),
                        "second_accuracy": float(second.mean()),
                        "delta": float(first.mean() - second.mean()),
                        "first_only_correct": first_wins,
                        "second_only_correct": second_wins,
                        "mcnemar_exact_p": exact_mcnemar_p(first_wins, second_wins),
                    }
                )
    write_csv(output / "paired_comparisons.csv", paired_rows, list(paired_rows[0]))

    # Unresolved output audit and a deliberately extreme all-correct ceiling.
    unresolved_rows: list[dict[str, Any]] = []
    ceiling_rows: list[dict[str, Any]] = []
    for model_index, row in enumerate(ranking):
        current_track_scores: dict[str, float] = {}
        ceiling_track_scores: dict[str, float] = {}
        for track in TRACKS:
            current_track_scores[track] = score_from_correctness(
                track, correctness[track][model_index], metadata[track]
            )
            ceiling_correctness = correctness[track][model_index].copy()
            ceiling_correctness[unresolved[track][model_index].astype(bool)] = 1
            ceiling_track_scores[track] = score_from_correctness(
                track, ceiling_correctness, metadata[track]
            )
            counts: Counter[tuple[str, str, str]] = Counter()
            for item_index in np.flatnonzero(unresolved[track][model_index]):
                info = metadata[track][int(item_index)]
                counts[
                    (
                        str(info.get("dimension", "")),
                        str(info.get("capability", "")),
                        str(info.get("difficulty", "")),
                    )
                ] += 1
            for (dimension, capability, difficulty), count in counts.items():
                unresolved_rows.append(
                    {
                        "rank": row["rank"],
                        "slug": row["slug"],
                        "display_name": row["display_name"],
                        "benchmark": track,
                        "dimension": dimension,
                        "task": capability,
                        "difficulty": difficulty,
                        "unresolved_count": count,
                    }
                )
        current_vpci = np.mean(list(current_track_scores.values()))
        ceiling_vpci = np.mean(list(ceiling_track_scores.values()))
        ceiling_rows.append(
            {
                "rank": row["rank"],
                "slug": row["slug"],
                "display_name": row["display_name"],
                "current_perception": current_track_scores["do_you_see_me"],
                "all_unresolved_correct_perception": ceiling_track_scores[
                    "do_you_see_me"
                ],
                "current_cognition": current_track_scores["minds_eye"],
                "all_unresolved_correct_cognition": ceiling_track_scores["minds_eye"],
                "current_vpci": current_vpci,
                "all_unresolved_correct_vpci": ceiling_vpci,
                "maximum_vpci_uplift": ceiling_vpci - current_vpci,
            }
        )
    write_csv(
        output / "unresolved_by_model_task.csv",
        unresolved_rows,
        [
            "rank",
            "slug",
            "display_name",
            "benchmark",
            "dimension",
            "task",
            "difficulty",
            "unresolved_count",
        ],
    )
    write_csv(
        output / "unresolved_ceiling_sensitivity.csv",
        ceiling_rows,
        list(ceiling_rows[0]),
    )

    # Protocol audit. Validate the retained raw-response hashes and describe how
    # the shared v4 evidence contract resolved each model's responses.
    protocol_rows: list[dict[str, Any]] = []
    finish_reason_rows: list[dict[str, Any]] = []
    extraction_status_rows: list[dict[str, Any]] = []
    extraction_status_totals: Counter[str] = Counter()
    source_response_hash_mismatches = 0
    terminal_fallback_count = 0
    for model_index, row in enumerate(ranking):
        slug = row["slug"]
        for track in TRACKS:
            run_config = read_json(bundle / slug / f"{track}.run_config.json")
            generation = run_config["generation"][track]
            diagnostics_path = bundle / slug / f"{track}.diagnostics.jsonl"
            diagnostics = {
                record["question_id"]: record
                for record in (
                    json.loads(line)
                    for line in diagnostics_path.open(encoding="utf-8")
                    if line.strip()
                )
            }
            evidence_path = bundle / slug / f"{track}.evidence_extraction.jsonl"
            evidence = {
                record["question_id"]: record
                for record in (
                    json.loads(line)
                    for line in evidence_path.open(encoding="utf-8")
                    if line.strip()
                )
            }
            if set(diagnostics) != set(ids[track]) or set(evidence) != set(ids[track]):
                raise RuntimeError(f"Evidence coverage changed for {slug}/{track}.")
            status_counts: Counter[str] = Counter()
            finish_counts: Counter[str] = Counter()
            token_counts: list[int] = []
            finish_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
            status_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for item_index, sample_id in enumerate(ids[track]):
                record = diagnostics[sample_id]
                audit = evidence[sample_id]
                finish_reason = str(record.get("finish_reason") or "missing")
                finish_counts[finish_reason] += 1
                if record.get("completion_tokens") is not None:
                    token_counts.append(int(record["completion_tokens"]))
                status = str(record.get("extractor_status") or "")
                if status != str(audit.get("status") or ""):
                    raise RuntimeError(
                        f"Evidence status changed for {slug}/{track}/{sample_id}."
                    )
                status_counts[status] += 1
                extraction_status_totals[f"{track}/{status}"] += 1
                if audit.get("terminal_fallback_method"):
                    terminal_fallback_count += 1
                raw_hash = hashlib.sha256(
                    str(record.get("output") or "").encode("utf-8")
                ).hexdigest()
                if {
                    str(record.get("extractor_source_output_sha256") or ""),
                    str(audit.get("response_sha256") or ""),
                } != {raw_hash}:
                    source_response_hash_mismatches += 1
                finish_stats[finish_reason][0] += 1
                finish_stats[finish_reason][1] += int(
                    correctness[track][model_index, item_index]
                )
                finish_stats[finish_reason][2] += int(
                    unresolved[track][model_index, item_index]
                )
                status_stats[status][0] += 1
                status_stats[status][1] += int(
                    correctness[track][model_index, item_index]
                )
            for finish_reason, (count, correct, unresolved_count) in sorted(
                finish_stats.items()
            ):
                finish_reason_rows.append(
                    {
                        "rank": row["rank"],
                        "slug": slug,
                        "display_name": row["display_name"],
                        "benchmark": track,
                        "finish_reason": finish_reason,
                        "count": count,
                        "rate": count / len(ids[track]),
                        "canonical_accuracy": correct / count,
                        "canonical_unresolved_count": unresolved_count,
                    }
                )
            for status, (count, correct) in sorted(status_stats.items()):
                extraction_status_rows.append(
                    {
                        "rank": row["rank"],
                        "slug": slug,
                        "display_name": row["display_name"],
                        "benchmark": track,
                        "extractor_status": status,
                        "count": count,
                        "rate": count / len(ids[track]),
                        "canonical_accuracy": correct / count,
                    }
                )
            protocol_rows.append(
                {
                    "rank": row["rank"],
                    "slug": slug,
                    "display_name": row["display_name"],
                    "benchmark": track,
                    "reasoning_profile": row["reasoning_profile"],
                    "weight_loading": run_config.get("weight_loading"),
                    "compute_dtype": run_config.get("compute_dtype"),
                    "max_model_len": run_config.get("max_model_len"),
                    "max_tokens": generation.get("max_tokens"),
                    "temperature": generation.get("temperature"),
                    "top_p": generation.get("top_p"),
                    "base_seed": run_config.get("generation", {}).get("base_seed"),
                    "prompt_mode": generation.get("prompt_mode"),
                    "prompt_sha256": generation.get("prompt_sha256"),
                    "stop_count": finish_counts.get("stop", 0),
                    "length_count": finish_counts.get("length", 0),
                    "length_rate": finish_counts.get("length", 0) / len(ids[track]),
                    "median_completion_tokens": (
                        float(np.median(token_counts)) if token_counts else ""
                    ),
                    "max_completion_tokens": max(token_counts) if token_counts else "",
                    "committed_count": status_counts.get("committed", 0),
                    "invalid_format_committed_count": status_counts.get(
                        "invalid_format_committed", 0
                    ),
                    "unsupported_by_evidence_count": status_counts.get(
                        "unsupported_by_evidence", 0
                    ),
                    "unresolved_count": status_counts.get("unresolved", 0),
                    "unresolved_truncated_response_count": status_counts.get(
                        "unresolved_truncated_response", 0
                    ),
                    "canonical_unresolved_count": int(
                        unresolved[track][model_index].sum()
                    ),
                }
            )
    if source_response_hash_mismatches:
        raise RuntimeError(
            "The v4 evidence audit no longer matches retained model responses: "
            f"{source_response_hash_mismatches} hash mismatch(es)."
        )
    write_csv(output / "protocol_audit.csv", protocol_rows, list(protocol_rows[0]))
    write_csv(
        output / "finish_reason_performance.csv",
        finish_reason_rows,
        list(finish_reason_rows[0]),
    )
    write_csv(
        output / "extraction_status_performance.csv",
        extraction_status_rows,
        list(extraction_status_rows[0]),
    )

    perception_values = np.asarray([row["perception_macro"] for row in cohort_rows])
    cognition_values = np.asarray([row["cognition_macro"] for row in cohort_rows])
    vpci_values = np.asarray([row["vpci"] for row in cohort_rows])

    # Test whether conclusions depend on the chosen two-layer aggregation.
    aggregation_values = {
        "arithmetic_mean": vpci_values,
        "geometric_mean": np.sqrt(perception_values * cognition_values),
        "harmonic_mean": (
            2.0
            * perception_values
            * cognition_values
            / (perception_values + cognition_values)
        ),
        "bottleneck_minimum": np.minimum(perception_values, cognition_values),
    }
    aggregation_rows: list[dict[str, Any]] = []
    aggregation_ranks: dict[str, np.ndarray] = {}
    for metric, values in aggregation_values.items():
        metric_ranks = descending_ranks(values)
        aggregation_ranks[metric] = metric_ranks
        for model_index, value in enumerate(values):
            aggregation_rows.append(
                {
                    "metric": metric,
                    "rank": int(metric_ranks[model_index]),
                    "slug": model_slugs[model_index],
                    "display_name": display_names[model_index],
                    "score": float(value),
                }
            )
    write_csv(
        output / "aggregation_sensitivity.csv",
        aggregation_rows,
        list(aggregation_rows[0]),
    )

    weight_rows: list[dict[str, Any]] = []
    weight_grid = np.linspace(0.0, 1.0, 1001)
    weight_ranks = np.zeros((len(weight_grid), model_count), dtype=np.int16)
    top_indices = np.zeros(len(weight_grid), dtype=np.int16)
    for weight_index, perception_weight in enumerate(weight_grid):
        values = (
            perception_weight * perception_values
            + (1.0 - perception_weight) * cognition_values
        )
        ranks = descending_ranks(values).astype(np.int16)
        weight_ranks[weight_index] = ranks
        top_indices[weight_index] = int(np.argmax(values))
        if weight_index % 10 == 0:
            for model_index, value in enumerate(values):
                weight_rows.append(
                    {
                        "perception_weight": float(perception_weight),
                        "cognition_weight": float(1.0 - perception_weight),
                        "rank": int(ranks[model_index]),
                        "slug": model_slugs[model_index],
                        "display_name": display_names[model_index],
                        "weighted_score": float(value),
                    }
                )
    write_csv(
        output / "layer_weight_sensitivity.csv",
        weight_rows,
        list(weight_rows[0]),
    )

    top_weight_intervals: list[dict[str, Any]] = []
    interval_start = 0
    for index in range(1, len(weight_grid) + 1):
        changed = index == len(weight_grid) or top_indices[index] != top_indices[index - 1]
        if changed:
            model_index = int(top_indices[index - 1])
            top_weight_intervals.append(
                {
                    "display_name": display_names[model_index],
                    "perception_weight_start": float(weight_grid[interval_start]),
                    "perception_weight_end": float(weight_grid[index - 1]),
                }
            )
            interval_start = index

    pareto_indices = []
    for model_index in range(model_count):
        dominated = any(
            other_index != model_index
            and perception_values[other_index] >= perception_values[model_index]
            and cognition_values[other_index] >= cognition_values[model_index]
            and (
                perception_values[other_index] > perception_values[model_index]
                or cognition_values[other_index] > cognition_values[model_index]
            )
            for other_index in range(model_count)
        )
        if not dominated:
            pareto_indices.append(model_index)

    qwen_mask = np.asarray(
        [not row["model_id"].lower().startswith("qwen/") for row in cohort_rows]
    )
    organization_values: dict[str, list[int]] = defaultdict(list)
    for model_index, row in enumerate(cohort_rows):
        organization = row["organization"] or row["model_id"].split("/", 1)[0]
        organization_values[organization].append(model_index)
    organization_perception = np.asarray(
        [perception_values[indices].mean() for indices in organization_values.values()]
    )
    organization_cognition = np.asarray(
        [cognition_values[indices].mean() for indices in organization_values.values()]
    )
    leave_one_out = [
        pearson(np.delete(perception_values, index), np.delete(cognition_values, index))
        for index in range(model_count)
    ]
    correlation_ci = bootstrap_correlation(
        perception_values,
        cognition_values,
        args.correlation_bootstrap,
        np.random.default_rng(args.seed + 3),
    )

    pc1_variance = next(
        row["explained_variance"] for row in loading_rows if row["component"] == 1
    )
    pc2_variance = next(
        row["explained_variance"] for row in loading_rows if row["component"] == 2
    )
    significant_associations = [row for row in association_rows if row["fdr_q"] < 0.05]
    significant_partial_associations = [
        row for row in association_rows if row["partial_fdr_q"] < 0.05
    ]

    top_difference = vpci_bootstrap[:, 0] - vpci_bootstrap[:, 1]
    summary = {
        "analysis_id": "combined_visual_v15",
        "seed": args.seed,
        "source": {
            "bundle": display_path(bundle),
            "bundle_generated_at": bundle_manifest["generated_at"],
            "model_count": model_count,
            "responses_per_model": sum(len(ids[track]) for track in TRACKS),
            "do_you_see_me_items": len(ids["do_you_see_me"]),
            "minds_eye_items": len(ids["minds_eye"]),
            "total_responses": observed_rows,
            "unresolved_responses": observed_unresolved,
            "validation_passed": True,
        },
        "protocol_audit": {
            "bundle_extraction_method": bundle_manifest["extraction"]["method"],
            "bundle_extraction_contract_sha256": bundle_manifest["extraction"][
                "contract_sha256"
            ],
            "checked_out_production_extraction_method": PRODUCTION_EXTRACTION_METHOD,
            "checked_out_production_contract_sha256": (
                extractor_contract_sha256(
                    DEFAULT_EXTRACTOR_MODEL,
                    int(bundle_manifest["extraction"]["max_tokens"]),
                    DEFAULT_EXTRACTOR_REVISION,
                )
            ),
            "bundle_matches_checked_out_production_contract": (
                bundle_manifest["extraction"]["method"]
                == PRODUCTION_EXTRACTION_METHOD
                and bundle_manifest["extraction"]["contract_sha256"]
                == extractor_contract_sha256(
                    DEFAULT_EXTRACTOR_MODEL,
                    int(bundle_manifest["extraction"]["max_tokens"]),
                    DEFAULT_EXTRACTOR_REVISION,
                )
            ),
            "length_finished_response_count": sum(
                int(row["length_count"]) for row in protocol_rows
            ),
            "evidence_status_counts": dict(sorted(extraction_status_totals.items())),
            "source_response_sha256_mismatches": source_response_hash_mismatches,
            "terminal_fallback_count": terminal_fallback_count,
            "interpretation": (
                "Every retained response was processed with the checked-out gold-blind "
                "v4 evidence contract. Invalid commitments and unresolved responses "
                "remain explicit incorrect outcomes rather than being repaired."
            ),
        },
        "methods": {
            "vpci": "Arithmetic mean of benchmark macro scores",
            "do_you_see_me": "Mean task accuracy within 2D and 3D, then equal mean of dimensions",
            "minds_eye": "Unweighted mean of eight task accuracies",
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_unit": "Items, stratified by ranking aggregation groups and shared across models",
            "correlation_bootstrap_replicates": args.correlation_bootstrap,
            "cross_task_permutations": args.permutations,
            "multiple_testing": "Benjamini-Hochberg FDR across 56 cross-layer task pairs",
        },
        "cross_benchmark": {
            "pearson": pearson(perception_values, cognition_values),
            "pearson_model_bootstrap_95_ci": list(correlation_ci),
            "spearman": spearman(perception_values, cognition_values),
            "non_qwen_model_count": int(qwen_mask.sum()),
            "non_qwen_pearson": pearson(
                perception_values[qwen_mask], cognition_values[qwen_mask]
            ),
            "non_qwen_spearman": spearman(
                perception_values[qwen_mask], cognition_values[qwen_mask]
            ),
            "organization_count": len(organization_values),
            "organization_mean_pearson": pearson(
                organization_perception, organization_cognition
            ),
            "organization_mean_spearman": spearman(
                organization_perception, organization_cognition
            ),
            "leave_one_model_out_pearson_range": [
                min(leave_one_out),
                max(leave_one_out),
            ],
        },
        "capability_structure": {
            "pc1_explained_variance": pc1_variance,
            "pc2_explained_variance": pc2_variance,
            "fdr_significant_cross_task_pair_count": len(significant_associations),
            "fdr_significant_cross_task_pairs": significant_associations,
            "vpci_controlled_fdr_significant_pair_count": len(
                significant_partial_associations
            ),
            "vpci_controlled_fdr_significant_pairs": significant_partial_associations,
        },
        "ranking_uncertainty": {
            "top_model": display_names[0],
            "top_model_probability_rank_1": rank_rows[0]["probability_rank_1"],
            "top_vs_second_observed_difference": ranking[0]["vpci"]
            - ranking[1]["vpci"],
            "top_vs_second_bootstrap_95_ci": percentile_interval(top_difference),
            "top_vs_second_probability_positive": float(np.mean(top_difference > 0)),
        },
        "aggregation_sensitivity": {
            "pareto_frontier": [display_names[index] for index in pareto_indices],
            "top_model_weight_intervals": top_weight_intervals,
            "arithmetic_geometric_rank_spearman": spearman(
                aggregation_ranks["arithmetic_mean"],
                aggregation_ranks["geometric_mean"],
            ),
            "arithmetic_harmonic_rank_spearman": spearman(
                aggregation_ranks["arithmetic_mean"],
                aggregation_ranks["harmonic_mean"],
            ),
            "arithmetic_bottleneck_rank_spearman": spearman(
                aggregation_ranks["arithmetic_mean"],
                aggregation_ranks["bottleneck_minimum"],
            ),
            "arithmetic_geometric_identical_order": bool(
                np.array_equal(
                    aggregation_ranks["arithmetic_mean"],
                    aggregation_ranks["geometric_mean"],
                )
            ),
            "arithmetic_harmonic_identical_order": bool(
                np.array_equal(
                    aggregation_ranks["arithmetic_mean"],
                    aggregation_ranks["harmonic_mean"],
                )
            ),
            "model_rank_ranges_across_layer_weights": [
                {
                    "display_name": display_names[index],
                    "best_rank": int(weight_ranks[:, index].min()),
                    "worst_rank": int(weight_ranks[:, index].max()),
                }
                for index in range(model_count)
            ],
        },
        "difficulty": {
            "cohort_mean_easy": float(
                np.mean([row["perception_easy"] for row in cohort_rows])
            ),
            "cohort_mean_medium": float(
                np.mean([row["perception_medium"] for row in cohort_rows])
            ),
            "cohort_mean_hard": float(
                np.mean([row["perception_hard"] for row in cohort_rows])
            ),
            "vpci_vs_easy_hard_drop_pearson": pearson(
                vpci_values,
                np.asarray([row["perception_easy_hard_drop"] for row in cohort_rows]),
            ),
        },
        "paired_comparisons": {
            "qwen35_thinking": [
                row
                for row in paired_rows
                if row["comparison"] == "qwen35_thinking" and row["group"] == "overall"
            ],
            "gemma_scale": [
                row
                for row in paired_rows
                if row["comparison"] == "gemma_scale" and row["group"] == "overall"
            ],
        },
        "interpretation_limits": [
            "The model is the unit for cross-model correlations and regressions (n=14).",
            "Five Qwen variants and two Gemma variants make models non-independent by family.",
            "Item bootstrap intervals describe sensitivity to the generated item population, not repeated-training or repeated-decoding uncertainty.",
            "Each model has one evaluated run; stochastic generation variance is not estimated.",
            "The generated suite has 4,500 Do You See Me and 799 Mind's Eye items, not the exact paper test distributions.",
            "Paper human scores are contextual references and are not matched-item baselines for this cohort.",
            "Cross-task correlations are exploratory associations and do not establish causal cognitive mechanisms.",
        ],
    }
    write_json(output / "analysis_summary.json", summary)

    print(f"Validated {model_count} models and {observed_rows:,} responses.")
    print(f"Wrote analysis artifacts to {output}")
    print(
        "Cross-benchmark Pearson/Spearman: "
        f"{summary['cross_benchmark']['pearson']:.3f}/"
        f"{summary['cross_benchmark']['spearman']:.3f}"
    )
    print(
        "Top model bootstrap P(rank 1): "
        f"{summary['ranking_uncertainty']['top_model_probability_rank_1']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
