"""Create the single public proof package from a completed Track-3 v5 run."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatial_harness.judge_track3 import (
    PAPER_JUDGE_MODEL,
    PAPER_JUDGE_REVISION,
    correct,
    is_terminal_inference_failure,
)
from spatial_harness.run_track3_vllm import (
    DATASETS,
    HARNESS_CONTRACT,
    MODES,
    PROMPT_MODES,
)
from spatial_harness.submission_contract import (
    EVALUATION_POLICY,
    HARNESS_VERSION,
    INFERENCE_FAILURE_DISPOSITION,
    INFERENCE_FAILURE_JUDGE_METHOD,
    INFERENCE_FAILURE_MCQ_ANSWER,
    INFERENCE_FAILURE_POLICY,
    INFERENCE_FAILURE_SCHEMA_VERSION,
    INFERENCE_FAILURE_VQA_ANSWER,
    JUDGE_DECODING,
    REPORT_SCHEMA_VERSION,
    REQUIRED_CONDITIONS,
    RUN_MANIFEST_SCHEMA_VERSION,
    SUBMISSION_SCHEMA_VERSION,
    condition_for,
    public_evaluation_group,
    public_question_id,
    sha256_file,
    validate_contract_identity,
)


ANSWER_TAG_RE = re.compile(
    r"<answer>(.*?)</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)
MAX_PUBLIC_VQA_ANSWER_CHARS = 2_048
LONG_VQA_COMMITMENT_PREFIX = "LONG_RESPONSE_SHA256:"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name} line {line_number} is not an object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _item_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("dataset") or ""),
        str(item.get("index") or ""),
        str(item.get("mode") or ""),
        str(item.get("pmode") or ""),
    )


def _public_vqa_answer(output: Any) -> str:
    raw = str(output or "").strip()
    tagged = ANSWER_TAG_RE.findall(raw)
    answer = (tagged[-1] if tagged else raw).strip()
    if not answer:
        raise ValueError("A judged VQA response has no public final answer")
    if len(answer) > MAX_PUBLIC_VQA_ANSWER_CHARS:
        digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        return (
            f"{LONG_VQA_COMMITMENT_PREFIX}{digest};"
            f"UTF8_BYTES:{len(answer.encode('utf-8'))};CHARS:{len(answer)}"
        )
    return answer


def _load_template_keys(contract_dir: Path) -> set[tuple[str, str]]:
    keys = set()
    for row in _read_jsonl(contract_dir / "submission_template.jsonl"):
        key = (str(row.get("condition") or ""), str(row.get("question_id") or ""))
        if not all(key) or key in keys:
            raise ValueError("The Track-3 public template contains invalid or duplicate rows")
        keys.add(key)
    return keys


def _prediction_paths(input_dir: Path) -> list[Path]:
    return [
        input_dir / f"pred_{mode}_{prompt_mode}.jsonl"
        for mode in MODES
        for prompt_mode in PROMPT_MODES
    ]


def _aggregate_report(rows: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    group_states: dict[tuple[str, str, str], bool] = {}
    for row in rows:
        key = (row["dataset"], row["condition"], row["evaluation_group"])
        group_states[key] = group_states.get(key, True) and bool(row["correct"])
    aggregate = collections.defaultdict(lambda: [0, 0])
    for (dataset, condition, _group), all_correct in group_states.items():
        aggregate[(dataset, condition)][0] += int(all_correct)
        aggregate[(dataset, condition)][1] += 1

    datasets = []
    for dataset in DATASETS:
        experiments = {}
        for mode_name in ("main", "no_image", "no_image_plus"):
            experiments[mode_name] = {}
            for prompt_mode in PROMPT_MODES:
                condition = f"{mode_name}_{prompt_mode}"
                correct_count, total = aggregate[(dataset, condition)]
                experiments[mode_name][prompt_mode] = {
                    "correct": correct_count,
                    "total": total,
                    "accuracy": round(correct_count / total, 6),
                }
        datasets.append({"dataset": dataset, "experiments": experiments})

    summary = {}
    for condition in REQUIRED_CONDITIONS:
        values = [
            aggregate[(dataset, condition)][0]
            / aggregate[(dataset, condition)][1]
            for dataset in DATASETS
        ]
        summary[condition] = round(sum(values) / len(values), 6)
    summary["cot_delta"] = round(
        summary["main_cot"] - summary["main_noncot"],
        6,
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "model": model,
        "conditions": list(REQUIRED_CONDITIONS),
        "datasets": datasets,
        "summary": summary,
    }


def package_submission(
    input_dir: Path,
    contract_dir: Path,
    output_dir: Path,
    public_model_name: str | None = None,
) -> Path:
    run_config = _read_json(input_dir / "run_config.json")
    if (
        run_config.get("schema_version") != 5
        or run_config.get("harness_contract") != HARNESS_CONTRACT
        or run_config.get("modes") != list(MODES)
        or run_config.get("prompt_modes") != list(PROMPT_MODES)
        or run_config.get("datasets") != list(DATASETS)
        or int(run_config.get("limit") or 0) != 0
    ):
        raise ValueError("The source run is not a complete Track-3 v5 evaluation")

    benchmark_manifest_path = contract_dir / "manifest.json"
    benchmark_manifest = _read_json(benchmark_manifest_path)
    validate_contract_identity(benchmark_manifest)
    template_keys = _load_template_keys(contract_dir)

    prediction_rows = []
    for path in _prediction_paths(input_dir):
        if not path.is_file():
            raise FileNotFoundError(f"Track-3 prediction artifact is missing: {path}")
        prediction_rows.extend(_read_jsonl(path))
    predictions = {_item_key(row): row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("Track-3 prediction artifacts contain duplicate rows")
    invalid_prediction_failures = [
        key
        for key, row in predictions.items()
        if (
            row.get("error")
            or not str(row.get("output") or "").strip()
            or row.get("terminal_failure") is not None
        )
        and not is_terminal_inference_failure(row)
    ]
    if invalid_prediction_failures:
        raise ValueError(
            "Track-3 predictions contain unfinalized inference failures: "
            f"{invalid_prediction_failures[:5]}"
        )

    judged_rows = _read_jsonl(input_dir / "judged.jsonl")
    judged = {_item_key(row): row for row in judged_rows}
    if set(judged) != set(predictions):
        raise ValueError("Judged evidence does not cover the exact prediction set")
    prediction_failure_keys = {
        key for key, row in predictions.items() if is_terminal_inference_failure(row)
    }
    judged_failure_keys = {
        key for key, row in judged.items() if is_terminal_inference_failure(row)
    }
    if judged_failure_keys != prediction_failure_keys:
        raise ValueError(
            "Judged evidence does not preserve the finalized inference failures"
        )
    failure_policy = run_config.get("terminal_inference_failures")
    if prediction_failure_keys:
        unique_failures = {
            f"{dataset}:{index}"
            for dataset, index, _mode, _prompt_mode in prediction_failure_keys
        }
        if (
            not isinstance(failure_policy, dict)
            or failure_policy.get("schema_version")
            != INFERENCE_FAILURE_SCHEMA_VERSION
            or failure_policy.get("policy") != INFERENCE_FAILURE_POLICY
            or failure_policy.get("disposition")
            != INFERENCE_FAILURE_DISPOSITION
            or failure_policy.get("eligible_category")
            != "input_context_exceeded"
            or int(failure_policy.get("condition_rows") or 0)
            != len(prediction_failure_keys)
            or int(failure_policy.get("unique_samples") or 0)
            != len(unique_failures)
            or set(failure_policy.get("sample_ids") or []) != unique_failures
        ):
            raise ValueError(
                "run_config.json does not describe the exact finalized "
                "inference-failure set"
            )
    elif failure_policy is not None:
        raise ValueError(
            "run_config.json declares inference failures but predictions contain none"
        )

    evidence = []
    method_counts = collections.Counter()
    for key in sorted(judged):
        item = judged[key]
        if item.get("jerr") or item.get("judged") is None:
            raise ValueError(f"Unresolved judge output for {key}")
        terminal_failure = is_terminal_inference_failure(item)
        if terminal_failure and (
            item.get("judged") != "0"
            or item.get("judge_method") != INFERENCE_FAILURE_JUDGE_METHOD
            or int(item.get("judge_attempts") or 0) != 0
        ):
            raise ValueError(f"Invalid inference-failure disposition for {key}")
        dataset, index, mode, prompt_mode = key
        condition = condition_for(mode, prompt_mode)
        question_id = public_question_id(dataset, index)
        evaluation_group = public_evaluation_group(
            dataset,
            item.get("group", index),
        )
        answer_type = str(item.get("answer_type") or "")
        if answer_type not in {"mcq", "vqa"}:
            raise ValueError(f"Unsupported answer type for {question_id}")
        if answer_type == "vqa" and mode == "noimgpp":
            raise ValueError("No-Image++ cannot contain VQA rows")

        if terminal_failure:
            judge_method = INFERENCE_FAILURE_JUDGE_METHOD
        elif item.get("judge_method") == "explicit_abstention":
            judge_method = "explicit_abstention"
        elif answer_type == "vqa":
            judge_method = "paper_vqa_llm_judge"
        else:
            judge_method = "paper_mcq_llm_judge"
        judge_attempts = int(
            item.get("judge_attempts")
            if item.get("judge_attempts") is not None
            else 1
        )
        if terminal_failure:
            public_answer = (
                INFERENCE_FAILURE_VQA_ANSWER
                if answer_type == "vqa"
                else INFERENCE_FAILURE_MCQ_ANSWER
            )
        else:
            public_answer = (
                _public_vqa_answer(item.get("output"))
                if answer_type == "vqa"
                else str(item["judged"]).upper()
            )
        row = {
            "dataset": dataset,
            "question_id": question_id,
            "evaluation_group": evaluation_group,
            "answer_type": answer_type,
            "condition": condition,
            "answer": public_answer,
            "correct": bool(correct(item)),
            "judge_method": judge_method,
            "judge_attempts": judge_attempts,
        }
        evidence.append(row)
        method_counts[judge_method] += 1

    actual_keys = {(row["condition"], row["question_id"]) for row in evidence}
    if actual_keys != template_keys:
        missing = sorted(template_keys - actual_keys)[:10]
        extra = sorted(actual_keys - template_keys)[:10]
        raise ValueError(
            "Track-3 evidence does not match the public contract; "
            f"missing={missing}, extra={extra}"
        )
    condition_counts = collections.Counter(row["condition"] for row in evidence)
    if dict(condition_counts) != benchmark_manifest["condition_counts"]:
        raise ValueError("Track-3 evidence condition counts do not match the contract")

    model = {
        "name": public_model_name or str(run_config["model"]),
        "served_name": str(run_config["model"]),
        "revision": str(run_config.get("model_revision") or ""),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.jsonl"
    report_path = output_dir / "leaderboard.json"
    manifest_path = output_dir / "run_manifest.json"
    _atomic_jsonl(submission_path, evidence)
    report = _aggregate_report(evidence, model)
    _atomic_json(report_path, report)

    run_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "submission_schema_version": SUBMISSION_SCHEMA_VERSION,
        "harness_contract": HARNESS_CONTRACT,
        "harness_version": HARNESS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "datasets": list(DATASETS),
        "conditions": list(REQUIRED_CONDITIONS),
        "evaluation_policy": EVALUATION_POLICY,
        "dataset_files": benchmark_manifest["dataset_files"],
        "data_manifest": benchmark_manifest["data_manifest"],
        "prompts": benchmark_manifest["prompts"],
        "decoding": {
            "strategy": "greedy",
            "temperature": run_config["temperature"],
            "top_p": run_config["top_p"],
            "seed": run_config["seed"],
            "max_tokens_noncot": run_config["max_tokens_noncot"],
            "max_tokens_cot": run_config["max_tokens_cot"],
            "chat_template_kwargs": run_config["chat_template_kwargs"],
            "server_metadata": run_config["server_metadata"],
        },
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "judge": {
            "served_model": PAPER_JUDGE_MODEL,
            "model": PAPER_JUDGE_MODEL,
            "revision": PAPER_JUDGE_REVISION,
            "system_prompt_sha256": benchmark_manifest["judge"][
                "system_prompt_sha256"
            ],
            "method_counts": dict(method_counts),
            "decoding": JUDGE_DECODING,
        },
        "condition_counts": dict(condition_counts),
        "error_counts": {
            "inference": len(prediction_failure_keys),
            "judge": 0,
            "missing_outputs": len(prediction_failure_keys),
        },
        "inference_failure_policy": failure_policy,
        "public_evidence_policy": {
            "mcq": "paper_judge_option_letter",
            "vqa": (
                "last_answer_tag_or_complete_direct_response_with_"
                "sha256_commitment_above_2048_chars"
            ),
            "oversized_vqa_commitment": (
                "sha256_utf8_with_byte_and_character_counts"
            ),
            "inference_failure": INFERENCE_FAILURE_DISPOSITION,
            "reasoning_traces_included": False,
            "private_ground_truth_included": False,
        },
        "artifacts": {
            "submission": {
                "filename": submission_path.name,
                "sha256": sha256_file(submission_path),
                "rows": len(evidence),
                "size_bytes": submission_path.stat().st_size,
            },
            "leaderboard_report": {
                "filename": report_path.name,
                "sha256": sha256_file(report_path),
                "size_bytes": report_path.stat().st_size,
                "dataset_count": len(DATASETS),
            },
            "source_run_config": {
                "filename": "run_config.json",
                "sha256": sha256_file(input_dir / "run_config.json"),
            },
            "source_judged_evidence": {
                "filename": "judged.jsonl",
                "sha256": sha256_file(input_dir / "judged.jsonl"),
                "rows": len(judged_rows),
            },
        },
        "debug": False,
    }
    _atomic_json(manifest_path, run_manifest)

    archive_path = output_dir / "spatial_reasoning_submission.zip"
    temporary = archive_path.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.write(submission_path, arcname="submission.jsonl")
        archive.write(manifest_path, arcname="run_manifest.json")
        archive.write(report_path, arcname="leaderboard.json")
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, archive_path)
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--public-model-name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    archive = package_submission(
        input_dir,
        args.contract.expanduser().resolve(),
        (args.output or input_dir / "submission_package").expanduser().resolve(),
        args.public_model_name.strip() or None,
    )
    print(f"Wrote Track-3 submission package: {archive}")


if __name__ == "__main__":
    main()
