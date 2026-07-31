"""Run the prespecified paired analysis for completed Track-3 evaluations."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from spatial_harness.judge_track3 import correct
from spatial_harness.run_track3_vllm import DATASETS


SEED = 20260723
DIMENSIONS = {
    "BLINK": "2D",
    "CV-Bench-2D": "2D",
    "CV-Bench-3D": "3D",
    "MMVP": "2D",
    "RealWorldQA": "2D",
    "VStarBench": "2D",
    "MMSIBench_wo_circular": "3D",
    "3DSRBench": "3D",
    "VSR_MCQ": "2D",
    "SpatialBench": "2D",
    "MindCube": "3D",
    "OmniSpatial": "3D",
    "SAT-Real": "dynamic",
}
CONDITION_ORDER = (
    "main_noncot",
    "main_cot",
    "noimage_noncot",
    "noimage_cot",
    "noimgpp_noncot",
    "noimgpp_cot",
)


@dataclass(frozen=True)
class Term:
    model: str
    condition: str
    coefficient: float


@dataclass(frozen=True)
class Endpoint:
    name: str
    scope: str
    terms: tuple[Term, ...]
    answer_type: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not an object")
            rows.append(value)
    return rows


def _condition(item: dict[str, Any]) -> str:
    return f"{item['mode']}_{item['pmode']}"


def _load_group_results(
    model: str,
    path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"No judged rows found for {model}")
    seen = set()
    groups: dict[tuple[str, str, str, str], list[bool]] = collections.defaultdict(list)
    method_counts = collections.Counter()
    finish_counts = collections.Counter()
    for item in rows:
        key = (
            str(item.get("dataset") or ""),
            str(item.get("index") or ""),
            str(item.get("mode") or ""),
            str(item.get("pmode") or ""),
        )
        if not all(key) or key in seen:
            raise ValueError(f"{model} has an invalid or duplicate judged key: {key}")
        seen.add(key)
        if item.get("jerr") or item.get("judged") is None:
            raise ValueError(f"{model} has unresolved judged evidence for {key}")
        dataset = key[0]
        if dataset not in DATASETS:
            raise ValueError(f"{model} contains unsupported dataset {dataset}")
        answer_type = str(item.get("answer_type") or "")
        if answer_type not in {"mcq", "vqa"}:
            raise ValueError(f"{model} contains unsupported answer type {answer_type}")
        group = str(item.get("group") or item["index"])
        group_key = (dataset, _condition(item), group, answer_type)
        groups[group_key].append(bool(correct(item)))
        method_counts[str(item.get("judge_method") or "unknown")] += 1
        finish_counts[str(item.get("finish_reason") or "unknown")] += 1

    group_rows = [
        {
            "model": model,
            "dataset": dataset,
            "dimension": DIMENSIONS[dataset],
            "condition": condition,
            "evaluation_group": group,
            "answer_type": answer_type,
            "correct": int(all(values)),
            "variant_count": len(values),
        }
        for (dataset, condition, group, answer_type), values in sorted(
            groups.items()
        )
    ]
    frame = pd.DataFrame(group_rows)
    present_conditions = set(frame["condition"])
    if present_conditions != set(CONDITION_ORDER):
        raise ValueError(
            f"{model} condition coverage is incomplete: {sorted(present_conditions)}"
        )
    quality = {
        "model": model,
        "source": str(path),
        "source_sha256": _sha256(path),
        "judged_rows": len(rows),
        "scoring_groups": len(frame),
        "judge_method_counts": dict(method_counts),
        "finish_reason_counts": dict(finish_counts),
        "unresolved": 0,
    }
    return frame, quality


def _aligned_contrast_values(
    frames: dict[str, pd.DataFrame],
    endpoint: Endpoint,
) -> dict[str, np.ndarray]:
    term_series = []
    for term in endpoint.terms:
        frame = frames[term.model]
        selected = frame.loc[frame["condition"] == term.condition].copy()
        if endpoint.answer_type:
            selected = selected.loc[selected["answer_type"] == endpoint.answer_type]
        series = selected.set_index(
            ["dataset", "evaluation_group", "answer_type"]
        )["correct"].astype(float)
        series = series.rename(f"{term.model}:{term.condition}")
        term_series.append((term, series))
    aligned = pd.concat([series for _term, series in term_series], axis=1, join="inner")
    if aligned.empty:
        raise ValueError(f"Endpoint {endpoint.name} has no paired groups")
    values = np.zeros(len(aligned), dtype=float)
    for column_index, (term, _series) in enumerate(term_series):
        values += term.coefficient * aligned.iloc[:, column_index].to_numpy()
    datasets = aligned.index.get_level_values("dataset").to_numpy()
    return {
        dataset: values[datasets == dataset]
        for dataset in DATASETS
        if np.any(datasets == dataset)
    }


def _macro(values: dict[str, np.ndarray]) -> float:
    return float(np.mean([np.mean(dataset_values) for dataset_values in values.values()]))


def _bootstrap_interval(
    values: dict[str, np.ndarray],
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    estimates = np.empty(reps, dtype=float)
    arrays = list(values.values())
    for repeat in range(reps):
        dataset_means = [
            float(np.mean(array[rng.integers(0, len(array), size=len(array))]))
            for array in arrays
        ]
        estimates[repeat] = float(np.mean(dataset_means))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _randomization_p(
    values: dict[str, np.ndarray],
    reps: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(_macro(values))
    arrays = list(values.values())
    extreme = 0
    for _repeat in range(reps):
        randomized = [
            array * rng.choice((-1.0, 1.0), size=len(array))
            for array in arrays
        ]
        estimate = abs(float(np.mean([np.mean(array) for array in randomized])))
        extreme += int(estimate >= observed - 1e-15)
    return (extreme + 1) / (reps + 1)


def _holm(p_values: Iterable[float]) -> list[float]:
    values = list(p_values)
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _condition_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, frame in frames.items():
        for (dataset, dimension, condition, answer_type), group in frame.groupby(
            ["dataset", "dimension", "condition", "answer_type"],
            sort=False,
        ):
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "dimension": dimension,
                    "condition": condition,
                    "answer_type": answer_type,
                    "correct": int(group["correct"].sum()),
                    "total": len(group),
                    "accuracy": float(group["correct"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _endpoint_definitions(model_names: list[str]) -> list[Endpoint]:
    endpoints = []
    for model in model_names:
        endpoints.extend(
            [
                Endpoint(
                    f"{model}:main_cot_effect",
                    model,
                    (
                        Term(model, "main_cot", 1),
                        Term(model, "main_noncot", -1),
                    ),
                ),
                Endpoint(
                    f"{model}:noncot_image_benefit",
                    model,
                    (
                        Term(model, "main_noncot", 1),
                        Term(model, "noimage_noncot", -1),
                    ),
                ),
                Endpoint(
                    f"{model}:noncot_abstention_recovery",
                    model,
                    (
                        Term(model, "noimgpp_noncot", 1),
                        Term(model, "noimage_noncot", -1),
                    ),
                    answer_type="mcq",
                ),
                Endpoint(
                    f"{model}:cot_by_image_interaction",
                    model,
                    (
                        Term(model, "main_cot", 1),
                        Term(model, "main_noncot", -1),
                        Term(model, "noimage_cot", -1),
                        Term(model, "noimage_noncot", 1),
                    ),
                ),
            ]
        )
    if len(model_names) == 2:
        first, second = model_names
        endpoints.extend(
            [
                Endpoint(
                    f"{second}_minus_{first}:main_noncot",
                    "cross_model",
                    (
                        Term(second, "main_noncot", 1),
                        Term(first, "main_noncot", -1),
                    ),
                ),
                Endpoint(
                    f"{second}_minus_{first}:main_cot_effect",
                    "cross_model",
                    (
                        Term(second, "main_cot", 1),
                        Term(second, "main_noncot", -1),
                        Term(first, "main_cot", -1),
                        Term(first, "main_noncot", 1),
                    ),
                ),
                Endpoint(
                    f"{second}_minus_{first}:noncot_image_benefit",
                    "cross_model",
                    (
                        Term(second, "main_noncot", 1),
                        Term(second, "noimage_noncot", -1),
                        Term(first, "main_noncot", -1),
                        Term(first, "noimage_noncot", 1),
                    ),
                ),
            ]
        )
    return endpoints


def analyze(
    model_inputs: list[tuple[str, Path]],
    output: Path,
    bootstrap_reps: int,
    permutation_reps: int,
    seed: int,
) -> dict[str, Any]:
    if len(model_inputs) != 2:
        raise ValueError("The confirmatory Track-3 plan requires exactly two models")
    frames = {}
    quality = []
    for model, path in model_inputs:
        if model in frames:
            raise ValueError(f"Duplicate model label: {model}")
        frame, model_quality = _load_group_results(model, path)
        frames[model] = frame
        quality.append(model_quality)

    output.mkdir(parents=True, exist_ok=True)
    group_results = pd.concat(frames.values(), ignore_index=True)
    group_results.to_csv(output / "group_results.csv", index=False)
    _condition_summary(frames).to_csv(
        output / "condition_accuracy.csv",
        index=False,
    )

    endpoints = _endpoint_definitions(list(frames))
    endpoint_rows = []
    for endpoint_index, endpoint in enumerate(endpoints):
        values = _aligned_contrast_values(frames, endpoint)
        estimate = _macro(values)
        bootstrap_rng = np.random.default_rng(seed + endpoint_index * 2)
        permutation_rng = np.random.default_rng(seed + endpoint_index * 2 + 1)
        ci_low, ci_high = _bootstrap_interval(
            values,
            bootstrap_reps,
            bootstrap_rng,
        )
        p_value = _randomization_p(
            values,
            permutation_reps,
            permutation_rng,
        )
        endpoint_rows.append(
            {
                "endpoint": endpoint.name,
                "scope": endpoint.scope,
                "answer_type": endpoint.answer_type or "all",
                "dataset_count": len(values),
                "paired_group_count": sum(len(array) for array in values.values()),
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "randomization_p": p_value,
            }
        )
    adjusted = _holm(row["randomization_p"] for row in endpoint_rows)
    for row, adjusted_p in zip(endpoint_rows, adjusted):
        row["holm_p"] = adjusted_p
        row["confirmatory_significant"] = bool(
            adjusted_p < 0.05
            and (row["ci_low"] > 0 or row["ci_high"] < 0)
        )
    endpoints_frame = pd.DataFrame(endpoint_rows)
    endpoints_frame.to_csv(output / "confirmatory_endpoints.csv", index=False)

    plan_path = (
        Path(__file__).resolve().parents[2]
        / "evaluation"
        / "research"
        / "PRESPECIFIED_TRACK3_ANALYSIS_PLAN.md"
    )
    summary = {
        "schema_version": "ms-vista-track3-analysis/v1",
        "status": "complete",
        "seed": seed,
        "bootstrap_reps": bootstrap_reps,
        "permutation_reps": permutation_reps,
        "analysis_plan": {
            "path": str(plan_path),
            "sha256": _sha256(plan_path),
        },
        "quality": quality,
        "confirmatory_test_count": len(endpoint_rows),
        "significant_endpoints": [
            row for row in endpoint_rows if row["confirmatory_significant"]
        ],
        "endpoints": endpoint_rows,
        "limitations": [
            "Two controlled model checkpoints do not support model-family generalization.",
            "One greedy pass-at-1 run does not estimate decoding-seed variance.",
            "MCQ/VQA and per-dataset splits beyond the prespecified endpoints are descriptive.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Track-3 Prespecified Analysis",
        "",
        f"Quality gate: passed for {len(frames)} models. "
        f"Global confirmatory family: {len(endpoint_rows)} tests.",
        "",
        "| Endpoint | Delta | 95% CI | Holm p | Result |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in endpoint_rows:
        result = "significant" if row["confirmatory_significant"] else "not significant"
        lines.append(
            f"| {row['endpoint']} | {100 * row['estimate']:+.2f} | "
            f"[{100 * row['ci_low']:+.2f}, {100 * row['ci_high']:+.2f}] | "
            f"{row['holm_p']:.4g} | {result} |"
        )
    lines.extend(
        [
            "",
            "Deltas are percentage-point changes after multiplying by 100. "
            "All confirmatory tests use paired evaluation groups and equal dataset weights.",
            "",
            "Do not generalize cross-model effects beyond these two checkpoints.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must use LABEL=/path/to/judged.jsonl")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(f"Invalid model input: {value}")
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=_parse_model, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5_000)
    parser.add_argument("--permutation-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.bootstrap_reps < 100 or args.permutation_reps < 100:
        parser.error("bootstrap and permutation repetitions must each be at least 100")
    if not math.isfinite(args.seed):
        parser.error("--seed must be finite")
    return args


def main() -> None:
    args = parse_args()
    summary = analyze(
        args.model,
        args.output.expanduser().resolve(),
        args.bootstrap_reps,
        args.permutation_reps,
        args.seed,
    )
    print(
        f"Track-3 analysis complete: {summary['confirmatory_test_count']} tests, "
        f"{len(summary['significant_endpoints'])} significant after Holm correction"
    )


if __name__ == "__main__":
    main()
