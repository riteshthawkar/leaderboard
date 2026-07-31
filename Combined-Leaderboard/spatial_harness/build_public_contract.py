"""Build the public and private contracts for a paper-aligned Track-3 release."""

from __future__ import annotations

import argparse
import collections
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatial_harness.judge_track3 import (
    JUDGE_SYS,
    JUDGE_VQA,
    PAPER_JUDGE_MODEL,
    PAPER_JUDGE_REVISION,
)
from spatial_harness.run_track3_vllm import (
    BASE_SYSTEM_PROMPT,
    COT_SYSTEM_PROMPT,
    DATASETS,
    MODES,
    PROMPT_MODES,
    build_records,
)
from spatial_harness.submission_contract import (
    BENCHMARK_MANIFEST_SCHEMA_VERSION,
    DATASET_TYPES,
    EVALUATION_POLICY,
    HARNESS_CONTRACT,
    HARNESS_VERSION,
    JUDGE_DECODING,
    REQUIRED_CONDITIONS,
    condition_for,
    public_evaluation_group,
    public_question_id,
    sha256_bytes,
    sha256_file,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return len(rows)


def _dataset_provenance(lmudata: Path, data_manifest: dict[str, Any]) -> dict[str, dict]:
    provenance = {}
    records = data_manifest.get("datasets") or {}
    for dataset in DATASETS:
        path = lmudata / f"{dataset}.tsv"
        source = records.get(dataset)
        if not path.is_file() or not isinstance(source, dict):
            raise ValueError(f"Track-3 data provenance is incomplete for {dataset}")
        actual_hash = sha256_file(path)
        if actual_hash != source.get("tsv_sha256"):
            raise ValueError(f"Track-3 dataset hash mismatch for {dataset}")
        provenance[dataset] = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": actual_hash,
            "rows": source.get("rows"),
            "answer_types": source.get("answer_types"),
            "source": {
                key: value
                for key, value in source.items()
                if key not in {"tsv_sha256", "rows", "answer_types"}
            },
        }
    return provenance


def build_contract(
    lmudata: Path,
    output_dir: Path,
    private_ground_truth_path: Path,
    benchmark_version: str,
) -> dict[str, Any]:
    data_manifest_path = lmudata / "track3_data_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    dataset_files = _dataset_provenance(lmudata, data_manifest)

    conditions_by_question: dict[str, set[str]] = collections.defaultdict(set)
    public_metadata: dict[str, dict[str, Any]] = {}
    ground_truth: dict[str, dict[str, Any]] = {}
    template_rows: list[dict[str, str]] = []
    condition_counts = collections.Counter()
    condition_groups: dict[str, set[str]] = collections.defaultdict(set)
    dataset_condition_counts = {
        dataset: collections.Counter() for dataset in DATASETS
    }
    dataset_condition_groups = {
        dataset: collections.defaultdict(set) for dataset in DATASETS
    }

    for mode in MODES:
        for dataset in DATASETS:
            records = build_records(
                lmudata,
                dataset,
                mode,
                include_payload=False,
            )
            for prompt_mode in PROMPT_MODES:
                condition = condition_for(mode, prompt_mode)
                for record in records:
                    question_id = public_question_id(dataset, record["index"])
                    evaluation_group = public_evaluation_group(
                        dataset,
                        record.get("group", record["index"]),
                    )
                    answer_type = str(record["answer_type"])
                    conditions_by_question[question_id].add(condition)
                    metadata = {
                        "question_id": question_id,
                        "dataset_key": dataset,
                        "type": DATASET_TYPES[dataset],
                        "answer_type": answer_type,
                        "evaluation_group": evaluation_group,
                    }
                    previous = public_metadata.setdefault(question_id, metadata)
                    if previous != metadata:
                        raise ValueError(
                            f"Inconsistent public metadata for {question_id}"
                        )
                    private = ground_truth.setdefault(
                        question_id,
                        {
                            **metadata,
                            "condition_answers": {},
                            "conditions": [],
                        },
                    )
                    answer = (
                        record["cannot_label"]
                        if mode == "noimgpp"
                        else record["gt"]
                    )
                    private["condition_answers"][condition] = str(answer)
                    template_rows.append(
                        {
                            "question_id": question_id,
                            "condition": condition,
                            "answer": "",
                        }
                    )
                    condition_counts[condition] += 1
                    condition_groups[condition].add(evaluation_group)
                    dataset_condition_counts[dataset][condition] += 1
                    dataset_condition_groups[dataset][condition].add(
                        evaluation_group
                    )

    public_questions = []
    for question_id in sorted(public_metadata):
        conditions = sorted(
            conditions_by_question[question_id],
            key=list(REQUIRED_CONDITIONS).index,
        )
        ground_truth[question_id]["conditions"] = conditions
        public_questions.append(
            {
                **public_metadata[question_id],
                "conditions": conditions,
            }
        )
    template_rows.sort(
        key=lambda row: (
            list(REQUIRED_CONDITIONS).index(row["condition"]),
            row["question_id"],
        )
    )

    questions_path = output_dir / "questions.jsonl"
    template_path = output_dir / "submission_template.jsonl"
    questions_count = _atomic_jsonl(questions_path, public_questions)
    template_count = _atomic_jsonl(template_path, template_rows)
    _atomic_json(private_ground_truth_path, ground_truth)

    prompt_provenance = {
        "noncot": {
            "source": "spatial_harness.run_track3_vllm.BASE_SYSTEM_PROMPT",
            "sha256": sha256_bytes(BASE_SYSTEM_PROMPT.encode("utf-8")),
        },
        "cot": {
            "source": "spatial_harness.run_track3_vllm.COT_SYSTEM_PROMPT",
            "sha256": sha256_bytes(COT_SYSTEM_PROMPT.encode("utf-8")),
        },
    }
    manifest = {
        "schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "task_id": "spatial",
        "benchmark_version": benchmark_version,
        "harness_contract": HARNESS_CONTRACT,
        "harness_version": HARNESS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "demo": False,
        "datasets": list(DATASETS),
        "dataset_count": len(DATASETS),
        "required_conditions": list(REQUIRED_CONDITIONS),
        "primary_condition": "main_noncot",
        "evaluation_policy": EVALUATION_POLICY,
        "condition_counts": {
            condition: condition_counts[condition]
            for condition in REQUIRED_CONDITIONS
        },
        "condition_group_counts": {
            condition: len(condition_groups[condition])
            for condition in REQUIRED_CONDITIONS
        },
        "dataset_condition_counts": {
            dataset: {
                condition: dataset_condition_counts[dataset][condition]
                for condition in REQUIRED_CONDITIONS
            }
            for dataset in DATASETS
        },
        "dataset_condition_group_counts": {
            dataset: {
                condition: len(dataset_condition_groups[dataset][condition])
                for condition in REQUIRED_CONDITIONS
            }
            for dataset in DATASETS
        },
        "unique_question_ids": len(public_questions),
        "dataset_files": dataset_files,
        "data_manifest": {
            "filename": data_manifest_path.name,
            "sha256": sha256_file(data_manifest_path),
        },
        "prompts": prompt_provenance,
        "decoding": {
            "strategy": "greedy",
            "temperature": 0,
            "top_p": 1.0,
            "seed": 0,
            "completion_budget": "server_context_remainder",
        },
        "judge": {
            "model": PAPER_JUDGE_MODEL,
            "revision": PAPER_JUDGE_REVISION,
            "system_prompt_sha256": {
                "mcq": sha256_bytes(JUDGE_SYS.encode("utf-8")),
                "vqa": sha256_bytes(JUDGE_VQA.encode("utf-8")),
            },
            "decoding": JUDGE_DECODING,
        },
        "artifacts": {
            "questions": {
                "filename": questions_path.name,
                "rows": questions_count,
                "sha256": sha256_file(questions_path),
            },
            "submission_template": {
                "filename": template_path.name,
                "rows": template_count,
                "sha256": sha256_file(template_path),
            },
        },
        "distribution": (
            "Question content and images remain in the pinned upstream datasets. "
            "This contract publishes stable identifiers and condition coverage only."
        ),
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmudata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-ground-truth", type=Path, required=True)
    parser.add_argument("--benchmark-version", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_contract(
        args.lmudata.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.private_ground_truth.expanduser().resolve(),
        str(args.benchmark_version).strip(),
    )
    print(
        "Built Track-3 public contract: "
        f"{sum(manifest['condition_counts'].values())} condition rows, "
        f"{manifest['unique_question_ids']} public question identifiers"
    )


if __name__ == "__main__":
    main()
