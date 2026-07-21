#!/usr/bin/env python3
"""Build the official Spatial task bundle consumed by the leaderboard server.

Run this only on the trusted administrator machine that prepared the pinned
LMUData files. The generated public files contain identifiers and provenance,
not benchmark questions, images, or answers. ``ground_truth.json`` is private.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from run_track3_vllm import CANNOT, build_records
from judge_track3 import JUDGE_DECODING, JUDGE_SYS
from spatial_contract import (
    BENCHMARK_MANIFEST_SCHEMA_VERSION,
    DATASETS,
    DATASET_DISPLAY_NAMES,
    DATASET_TYPES,
    HARNESS_VERSION,
    JUDGE_REVISION,
    MODES,
    PROMPT_MODES,
    REQUIRED_CONDITIONS,
    condition_for,
    load_ablation_manifest,
    sha256_file,
    sha256_bytes,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl(path: Path, rows) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return count


def _merge_ground_truth(target: dict, record: dict, conditions: list[str]) -> None:
    question_id = record["question_id"]
    existing = target.setdefault(
        question_id,
        {
            "answer": record["gt"],
            "condition_answers": {},
            "conditions": [],
            "group": DATASET_DISPLAY_NAMES[record["dataset"]],
            "dataset": DATASET_DISPLAY_NAMES[record["dataset"]],
            "dataset_key": record["dataset"],
            "type": DATASET_TYPES[record["dataset"]],
            "evaluation_group": record["evaluation_group"],
            "source_index": record["source_index"],
            "rotation": record["rotation"],
        },
    )
    stable_fields = ("dataset_key", "evaluation_group", "source_index", "rotation")
    for field in stable_fields:
        if existing[field] != (
            record["dataset"] if field == "dataset_key" else record[field]
        ):
            raise ValueError(f"Conflicting metadata for spatial question_id {question_id}")
    for condition in conditions:
        previous = existing["condition_answers"].get(condition)
        if previous is not None and previous != record["gt"]:
            raise ValueError(f"Conflicting answer for {condition}/{question_id}")
        existing["condition_answers"][condition] = record["gt"]
        if condition not in existing["conditions"]:
            existing["conditions"].append(condition)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmudata", required=True)
    parser.add_argument("--benchmark-version", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "tasks" / "spatial"),
    )
    parser.add_argument(
        "--ground-truth-output",
        required=True,
        help="Private answer-key destination outside the public output directory.",
    )
    parser.add_argument(
        "--ablation-manifest",
        default=str(HERE / "ablation_manifest.json"),
    )
    parser.add_argument("--prompts-dir", default=str(HERE / "prompts"))
    args = parser.parse_args()

    if not args.benchmark_version.strip():
        parser.error("--benchmark-version cannot be empty")

    lmudata = Path(args.lmudata).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path = Path(args.ground_truth_output).resolve()
    try:
        ground_truth_path.relative_to(output_dir)
    except ValueError:
        pass
    else:
        parser.error(
            "--ground-truth-output must be outside --output-dir so the private answer key "
            "cannot be published with the public bundle"
        )
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_manifest = load_ablation_manifest(args.ablation_manifest)

    dataset_files = {}
    for dataset in DATASETS:
        path = lmudata / f"{dataset}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Dataset TSV is missing: {path}")
        dataset_files[dataset] = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    prompts = {}
    for prompt_mode, filename in (
        ("noncot", "base_noncot.txt"),
        ("cot", "cot_default.txt"),
    ):
        path = Path(args.prompts_dir).resolve() / filename
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Prompt file is missing or empty: {path}")
        prompts[prompt_mode] = {"filename": filename, "sha256": sha256_file(path)}

    ground_truth = {}
    template_rows = []
    condition_counts = Counter()
    condition_group_counts = Counter()
    dataset_condition_counts = {}
    dataset_condition_group_counts = {}

    for mode in MODES:
        conditions = [condition_for(mode, prompt_mode) for prompt_mode in PROMPT_MODES]
        for dataset in DATASETS:
            selected = ablation_manifest[dataset] if mode != "main" else None
            records = build_records(
                lmudata,
                dataset,
                mode,
                selected_indices=selected,
                include_payload=False,
            )
            groups = {record["evaluation_group"] for record in records}
            for condition in conditions:
                condition_counts[condition] += len(records)
                condition_group_counts[condition] += len(groups)
                dataset_condition_counts.setdefault(dataset, {})[condition] = len(records)
                dataset_condition_group_counts.setdefault(dataset, {})[condition] = len(groups)
                template_rows.extend(
                    {
                        "question_id": record["question_id"],
                        "condition": condition,
                        "answer": "",
                    }
                    for record in records
                )
            for record in records:
                _merge_ground_truth(ground_truth, record, conditions)
            print(f"[{mode}] {dataset}: {len(records)} rows, {len(groups)} scored groups")

    for value in ground_truth.values():
        value["conditions"].sort(key=list(REQUIRED_CONDITIONS).index)
        value["answer"] = value["condition_answers"].get(
            "main_noncot",
            next(iter(value["condition_answers"].values())),
        )

    questions = [
        {
            "question_id": question_id,
            "dataset": value["dataset"],
            "dataset_key": value["dataset_key"],
            "type": value["type"],
            "evaluation_group": value["evaluation_group"],
            "conditions": value["conditions"],
        }
        for question_id, value in ground_truth.items()
    ]

    questions_path = output_dir / "questions.json"
    questions_jsonl_path = output_dir / "questions.jsonl"
    template_path = output_dir / "submission_template.jsonl"
    _write_json(ground_truth_path, ground_truth)
    _write_json(
        questions_path,
        {
            "task_id": "spatial",
            "version": args.benchmark_version,
            "total_samples": len(questions),
            "conditions": list(REQUIRED_CONDITIONS),
            "samples": questions,
        },
    )
    questions_rows = _write_jsonl(questions_jsonl_path, questions)
    template_count = _write_jsonl(template_path, template_rows)

    manifest = {
        "schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "task_id": "spatial",
        "benchmark_version": args.benchmark_version,
        "harness_version": HARNESS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "demo": False,
        "datasets": list(DATASETS),
        "dataset_count": len(DATASETS),
        "required_conditions": list(REQUIRED_CONDITIONS),
        "primary_condition": "main_noncot",
        "condition_counts": dict(condition_counts),
        "condition_group_counts": dict(condition_group_counts),
        "dataset_condition_counts": dataset_condition_counts,
        "dataset_condition_group_counts": dataset_condition_group_counts,
        "unique_question_ids": len(ground_truth),
        "dataset_files": dataset_files,
        "prompts": prompts,
        "decoding": {
            "strategy": "greedy",
            "temperature": 0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
        },
        "judge": {
            "revision": JUDGE_REVISION,
            "system_prompt_sha256": sha256_bytes(JUDGE_SYS.encode("utf-8")),
            "decoding": JUDGE_DECODING,
        },
        "ablation_manifest": {
            "filename": Path(args.ablation_manifest).name,
            "sha256": sha256_file(args.ablation_manifest),
        },
        "no_image_plus_option": CANNOT,
        "artifacts": {
            "questions": {
                "filename": questions_jsonl_path.name,
                "rows": questions_rows,
                "sha256": sha256_file(questions_jsonl_path),
            },
            "submission_template": {
                "filename": template_path.name,
                "rows": template_count,
                "sha256": sha256_file(template_path),
            },
        },
        "distribution": (
            "Questions and images remain in their upstream datasets. This manifest pins "
            "the normalized files used by the evaluation harness."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Wrote official spatial bundle to {output_dir}")
    print(f"Wrote private spatial ground truth to {ground_truth_path}")
    print(f"Manifest SHA256: {sha256_file(manifest_path)}")
    print("Keep ground_truth.json private; publish the manifest and template only.")


if __name__ == "__main__":
    main()
