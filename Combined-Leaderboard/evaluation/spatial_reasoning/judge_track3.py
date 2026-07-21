#!/usr/bin/env python3
"""Judge spatial generations and emit leaderboard-ready final answers."""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from spatial_contract import (
    DATASETS,
    HARNESS_VERSION,
    JUDGE_REVISION,
    REPORT_SCHEMA_VERSION,
    REQUIRED_CONDITIONS,
    RUN_MANIFEST_SCHEMA_VERSION,
    SUBMISSION_SCHEMA_VERSION,
    condition_for,
    parse_csv_values,
    sha256_bytes,
    sha256_file,
)


JUDGE_SYS = (
    "You are a helpful assistant.\n\n Your task: given (1) a free-form \"Response\" and (2) a list of "
    "\"Options\", decide which option the response most likely corresponds to and return the option letter. "
    "If no option clearly matches, output \"0\".\n\n Inputs: - Response: free-form text that may include a "
    "letter, a phrase, or an explanation. - Options: A series of choices, each starting with a single "
    "uppercase letter followed by \".\", one option in each line.\n\n Output format: - STRICTLY OUTPUT "
    "EXACTLY ONE CHARACTER: a single uppercase option letter from the allowed set, or \"0\".\n - Do not "
    "output any explanation, spaces, punctuation, or additional text.\n\n Rules:\n 1) If the response "
    "explicitly names exactly one letter (patterns like \"A\", \"A)\", \"Option A\", \"Answer is C\"), return "
    "that letter immediately.\n 2) Only evaluate the explicitly provided choice. If the response is long and "
    "complex without an explicit final choice, return \"0\".\n 3) If multiple choices appear in the response, "
    "the last unambiguous one is the final choice.\n 4) Never judge factual correctness - only map the response "
    "to the best matching option letter from the given options.\n 5) If no explicit letter can be extracted "
    "from the response, compare the response's meaning to option texts. If exactly one option clearly restates "
    "or is a synonym, number, name, or unit match for the response, return its letter.\n 6) If the "
    "response uses standard MCQ phrases such as \"none of the above\" or \"all of the above\" and a matching "
    "option exists, map them. If there is no matching option, output \"0\".\n 7) If the response contains "
    "both an explicit letter and a conflicting phrase, prefer the explicit letter. If conflicts remain or are "
    "unclear, output \"0\".\n 8) If the response says \"I don't know\", \"Cannot determine\", or similar, "
    "output \"0\".\n"
)

ABSTENTION_PHRASES = {
    "cannot determine",
    "cannot determine from the image",
    "can not determine",
    "can not determine from the image",
    "cannot tell",
    "cannot tell from the image",
    "i cannot determine",
    "i cannot determine from the image",
    "i can not determine",
    "i can not determine from the image",
    "i cannot tell",
    "i cannot tell from the image",
    "it cannot be determined",
    "it cannot be determined from the image",
    "insufficient information",
    "insufficient visual information",
    "there is insufficient information",
    "not enough information",
    "not enough visual information",
    "there is not enough information",
    "unknown",
    "unanswerable",
}

PREDICTION_FILE_RE = re.compile(r"^pred_(main|noimage|noimgpp)_(noncot|cot)\.jsonl$")

JUDGE_DECODING = {
    "strategy": "greedy",
    "temperature": 0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
    "max_tokens": 4,
}


def _clear_stale_judge_artifacts(results_dir: Path) -> None:
    """Ensure a failed judge rerun cannot leave an older upload pair behind."""
    names = {
        "submission.jsonl",
        "submission.jsonl.tmp",
        "run_manifest.json",
        "leaderboard.json",
        "judge_errors.json",
        "spatial_reasoning_submission.zip",
        "spatial_reasoning_submission.zip.tmp",
    }
    for mode in ("main", "noimage", "noimgpp"):
        for prompt_mode in ("noncot", "cot"):
            names.update(
                {
                    f"judged_{mode}_{prompt_mode}.jsonl",
                    f"judged_{mode}_{prompt_mode}.jsonl.tmp",
                }
            )
    for name in names:
        (results_dir / name).unlink(missing_ok=True)


def _write_submission_package(
    results_dir: Path,
    submission_path: Path,
    run_manifest_path: Path,
    report_path: Path,
) -> Path:
    """Atomically create the only file users upload to the leaderboard."""
    package_path = results_dir / "spatial_reasoning_submission.zip"
    temporary = results_dir / "spatial_reasoning_submission.zip.tmp"
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as package:
        package.write(submission_path, arcname="submission.jsonl")
        package.write(run_manifest_path, arcname="run_manifest.json")
        package.write(report_path, arcname="leaderboard.json")
    with open(temporary, "rb+") as handle:
        os.fsync(handle.fileno())
    temporary.replace(package_path)
    return package_path


def judge_user(item: dict) -> str:
    options = item.get("options") or {}
    option_lines = "\n".join(
        f"{letter}. {options[letter]}" for letter in sorted(options)
    )
    return f"Response:\n{item.get('output') or ''}\n\nOptions:\n{option_lines}"


def parse_letter(value, allowed_letters) -> str | None:
    text = str(value or "").strip().upper()
    if not re.fullmatch(r"[A-Z0]", text):
        return None
    return text if text == "0" or text in set(allowed_letters) else None


def _normalized_answer_text(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^\s*(final\s+answer|answer|option|choice)\s*(?:is|=)?\s*[:\-]?\s*", "", text)
    text = text.strip().strip("<>()[]{}.,:;!?\"'")
    return re.sub(r"\s+", " ", text)


def explicit_abstention_letter(item: dict) -> str | None:
    """Map an explicit final abstention to the injected No-Image++ option."""
    if item.get("mode") != "noimgpp" or not item.get("cannot_label"):
        return None
    output = str(item.get("output") or "").strip()
    answer_tags = re.findall(r"<answer>(.*?)</answer>", output, flags=re.IGNORECASE | re.DOTALL)
    candidate = answer_tags[-1] if answer_tags else output if len(output) <= 240 else ""
    normalized = _normalized_answer_text(candidate)
    return str(item["cannot_label"]).upper() if normalized in ABSTENTION_PHRASES else None


async def judge_one(
    client,
    semaphore: asyncio.Semaphore,
    item: dict,
    model: str,
    max_retries: int,
) -> dict:
    abstention = explicit_abstention_letter(item)
    if abstention:
        item["judged"] = abstention
        item["judge_error"] = None
        item["judge_method"] = "explicit_abstention"
        item["judge_attempts"] = 0
        return item

    last_error = None
    allowed = sorted((item.get("options") or {}).keys())
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=JUDGE_DECODING["temperature"],
                    top_p=JUDGE_DECODING["top_p"],
                    max_tokens=JUDGE_DECODING["max_tokens"],
                    extra_body={
                        "top_k": JUDGE_DECODING["top_k"],
                        "repetition_penalty": JUDGE_DECODING["repetition_penalty"],
                    },
                    messages=[
                        {"role": "system", "content": JUDGE_SYS},
                        {"role": "user", "content": judge_user(item)},
                    ],
                )
            raw = response.choices[0].message.content
            judged = parse_letter(raw, allowed)
            if judged is None:
                raise RuntimeError(f"judge returned an invalid option value: {raw!r}")
            item["judged"] = judged
            item["judge_error"] = None
            item["judge_method"] = "qwen_llm_judge"
            item["judge_attempts"] = attempt + 1
            return item
        except Exception as exc:
            last_error = str(exc)[:500]
            if attempt < max_retries:
                await asyncio.sleep(min(2**attempt, 8))
    item["judged"] = None
    item["judge_error"] = last_error or "unknown judge error"
    item["judge_method"] = "qwen_llm_judge"
    item["judge_attempts"] = max_retries + 1
    return item


async def judge_batch(clients, items, model, concurrency, max_retries):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        judge_one(
            clients[index % len(clients)],
            semaphore,
            item,
            model,
            max_retries,
        )
        for index, item in enumerate(items)
    ]
    for future in asyncio.as_completed(tasks):
        await future
    return items


def is_correct(item: dict) -> bool:
    expected = item.get("cannot_label") if item.get("mode") == "noimgpp" else item.get("gt")
    return str(item.get("judged") or "").upper() == str(expected or "").upper()


def _load_inference_manifest(results_dir: Path) -> dict:
    path = results_dir / "inference_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Inference manifest is missing: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "ms-vista-spatial-inference/v1":
        raise ValueError("Unsupported or missing inference manifest schema_version")
    if manifest.get("harness_version") != HARNESS_VERSION:
        raise ValueError(
            f"Inference manifest was produced by harness {manifest.get('harness_version')!r}; "
            f"expected {HARNESS_VERSION!r}. Rerun inference with the current harness."
        )
    if manifest.get("debug"):
        raise ValueError("Debug or --limit inference runs cannot create leaderboard submissions")
    if manifest.get("datasets") != list(DATASETS):
        raise ValueError(
            "Inference manifest must contain all 13 official datasets in canonical order"
        )
    conditions = manifest.get("conditions")
    if conditions != list(REQUIRED_CONDITIONS):
        raise ValueError(
            "Inference manifest must contain the six official conditions in canonical order"
        )
    if any(int(value or 0) for value in (manifest.get("error_counts") or {}).values()):
        raise ValueError("Inference manifest contains failed model generations")
    if not manifest.get("benchmark_manifest_sha256"):
        raise ValueError(
            "Inference was not pinned to an official benchmark manifest. Rerun inference "
            "with --benchmark-manifest before creating leaderboard artifacts."
        )
    return manifest


def _prediction_artifacts(results_dir: Path, manifest: dict) -> list[tuple[Path, str, str, str]]:
    artifacts = manifest.get("artifacts") or {}
    paths = []
    seen_conditions = set()
    for filename, metadata in artifacts.items():
        match = PREDICTION_FILE_RE.fullmatch(filename)
        if not match:
            continue
        mode, prompt_mode = match.groups()
        condition = condition_for(mode, prompt_mode)
        path = results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prediction artifact is missing: {path}")
        if sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Prediction artifact hash mismatch: {filename}")
        seen_conditions.add(condition)
        paths.append((path, mode, prompt_mode, condition))
    if seen_conditions != set(REQUIRED_CONDITIONS):
        missing = sorted(set(REQUIRED_CONDITIONS) - seen_conditions)
        raise ValueError(f"Prediction artifacts are missing conditions: {', '.join(missing)}")
    paths.sort(key=lambda item: list(REQUIRED_CONDITIONS).index(item[3]))
    return paths


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} line {line_number} is invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name} line {line_number} must be a JSON object")
            yield line_number, row


def _aggregate_report(group_states: dict, datasets: list[str]) -> dict:
    aggregate = collections.defaultdict(lambda: [0, 0])
    for (dataset, condition, _group), state in group_states.items():
        aggregate[(dataset, condition)][0] += int(state["all_correct"])
        aggregate[(dataset, condition)][1] += 1

    rows = []
    for dataset in datasets:
        experiments = {}
        for mode_name in ("main", "no_image", "no_image_plus"):
            experiments[mode_name] = {}
            for prompt_mode in ("noncot", "cot"):
                condition = f"{mode_name}_{prompt_mode}"
                correct, total = aggregate.get((dataset, condition), [0, 0])
                experiments[mode_name][prompt_mode] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": round(correct / total, 6) if total else None,
                }
        rows.append({"dataset": dataset, "experiments": experiments})

    summary = {}
    for condition in REQUIRED_CONDITIONS:
        values = []
        for dataset in datasets:
            correct, total = aggregate.get((dataset, condition), [0, 0])
            if total:
                values.append(correct / total)
        summary[condition] = round(sum(values) / len(values), 6) if values else None
    if summary["main_noncot"] is not None and summary["main_cot"] is not None:
        summary["cot_delta"] = round(
            summary["main_cot"] - summary["main_noncot"], 6
        )
    else:
        summary["cot_delta"] = None
    return {"datasets": rows, "summary": summary}


def resolve_judge_endpoint_model(endpoint_model: str, legacy_model: str) -> str:
    endpoint_model = str(endpoint_model or "").strip()
    legacy_model = str(legacy_model or "").strip()
    if endpoint_model and legacy_model:
        raise ValueError("--model cannot be combined with --endpoint-model")
    return endpoint_model or legacy_model or "judge"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--endpoints", default="http://localhost:8100/v1")
    parser.add_argument(
        "--endpoint-model",
        default="",
        help="Exact judge model identifier sent to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Compatibility alias for --endpoint-model.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("SPATIAL_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY",
        help="Judge endpoint API key. Prefer SPATIAL_JUDGE_API_KEY instead of command history.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=os.getenv("SPATIAL_JUDGE_TIMEOUT_SECONDS", "120"),
        help="Per-request timeout for judge calls.",
    )
    parser.add_argument("--judge-revision", default=JUDGE_REVISION)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    if args.concurrency < 1 or args.batch_size < 1:
        parser.error("--concurrency and --batch-size must be at least 1")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if (
        not math.isfinite(args.request_timeout_seconds)
        or args.request_timeout_seconds <= 0
    ):
        parser.error("--request-timeout-seconds must be a positive finite number")
    if args.judge_revision != JUDGE_REVISION:
        parser.error(
            f"--judge-revision must be {JUDGE_REVISION!r} for leaderboard submissions"
        )
    try:
        endpoint_model = resolve_judge_endpoint_model(
            args.endpoint_model,
            args.model,
        )
    except ValueError as exc:
        parser.error(str(exc))
    endpoints = [value.strip() for value in args.endpoints.split(",") if value.strip()]
    if not endpoints:
        parser.error("At least one judge endpoint is required")

    results_dir = Path(args.results_dir).resolve()
    inference_manifest = _load_inference_manifest(results_dir)
    datasets = parse_csv_values(
        ",".join(inference_manifest.get("datasets") or []),
        DATASETS,
        "dataset",
    )
    prediction_files = _prediction_artifacts(results_dir, inference_manifest)

    from openai import AsyncOpenAI

    clients = [
        AsyncOpenAI(
            base_url=endpoint,
            api_key=args.api_key,
            timeout=args.request_timeout_seconds,
            max_retries=0,
        )
        for endpoint in endpoints
    ]
    _clear_stale_judge_artifacts(results_dir)
    submission_temp = results_dir / "submission.jsonl.tmp"
    seen = set()
    condition_counts = collections.Counter()
    judge_method_counts = collections.Counter()
    group_states = {}
    judge_errors = []
    evidence_artifacts = {}

    with open(submission_temp, "w", encoding="utf-8", newline="\n") as submission_handle:
        for path, expected_mode, expected_prompt_mode, expected_condition in prediction_files:
            evidence_temp = results_dir / f"judged_{expected_mode}_{expected_prompt_mode}.jsonl.tmp"
            batch = []
            input_rows = 0
            with open(evidence_temp, "w", encoding="utf-8", newline="\n") as evidence_handle:
                async def flush_batch():
                    nonlocal batch
                    if not batch:
                        return
                    await judge_batch(
                        clients,
                        batch,
                        endpoint_model,
                        args.concurrency,
                        args.max_retries,
                    )
                    for item in batch:
                        question_id = str(item.get("question_id") or "").strip()
                        condition = str(item.get("condition") or "").strip()
                        key = (condition, question_id)
                        if item.get("judge_error") or not item.get("judged"):
                            judge_errors.append(
                                {
                                    "condition": condition,
                                    "question_id": question_id,
                                    "error": item.get("judge_error") or "missing judged answer",
                                }
                            )
                            continue
                        if key in seen:
                            raise ValueError(
                                f"Duplicate judged output for {condition}/{question_id}"
                            )
                        seen.add(key)
                        condition_counts[condition] += 1
                        judge_method_counts[item["judge_method"]] += 1
                        correct = is_correct(item)
                        group_key = (
                            item["dataset"],
                            condition,
                            str(item.get("evaluation_group") or question_id),
                        )
                        state = group_states.setdefault(
                            group_key,
                            {"all_correct": True, "variants": 0},
                        )
                        state["all_correct"] = state["all_correct"] and correct
                        state["variants"] += 1
                        submission_handle.write(
                            json.dumps(
                                {
                                    "dataset": item["dataset"],
                                    "question_id": question_id,
                                    "evaluation_group": item.get("evaluation_group") or question_id,
                                    "condition": condition,
                                    "answer": item["judged"],
                                    "correct": correct,
                                    "judge_method": item["judge_method"],
                                    "judge_attempts": item["judge_attempts"],
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )
                        evidence_handle.write(
                            json.dumps(
                                {
                                    "dataset": item["dataset"],
                                    "question_id": question_id,
                                    "evaluation_group": item.get("evaluation_group"),
                                    "condition": condition,
                                    "answer": item["judged"],
                                    "correct": correct,
                                    "judge_method": item["judge_method"],
                                    "judge_attempts": item["judge_attempts"],
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )
                    if judge_errors:
                        error_path = results_dir / "judge_errors.json"
                        error_path.write_text(
                            json.dumps(
                                {"count": len(judge_errors), "errors": judge_errors[:100]},
                                indent=2,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        raise SystemExit(
                            f"The judge failed for {len(judge_errors)} item(s). Stopped early; "
                            f"no leaderboard submission was produced. Inspect {error_path} and rerun."
                        )
                    batch = []

                for line_number, item in _iter_jsonl(path):
                    input_rows += 1
                    if item.get("mode") != expected_mode or item.get("pmode") != expected_prompt_mode:
                        raise ValueError(
                            f"{path.name} line {line_number} has inconsistent mode metadata"
                        )
                    if item.get("condition") != expected_condition:
                        raise ValueError(
                            f"{path.name} line {line_number} has condition {item.get('condition')!r}; "
                            f"expected {expected_condition!r}"
                        )
                    if item.get("error") or not str(item.get("output") or "").strip():
                        raise ValueError(
                            f"{path.name} line {line_number} contains a failed or empty model generation"
                        )
                    if not item.get("question_id") or not item.get("dataset"):
                        raise ValueError(
                            f"{path.name} line {line_number} is missing question_id or dataset"
                        )
                    batch.append(item)
                    if len(batch) >= args.batch_size:
                        await flush_batch()
                await flush_batch()
                evidence_handle.flush()
                os.fsync(evidence_handle.fileno())
            expected_rows = int(
                (inference_manifest.get("artifacts") or {}).get(path.name, {}).get("rows")
                or 0
            )
            if input_rows != expected_rows:
                raise ValueError(
                    f"{path.name} contains {input_rows} rows; inference manifest declares {expected_rows}"
                )
            evidence_path = results_dir / evidence_temp.name.removesuffix(".tmp")
            evidence_temp.replace(evidence_path)
            evidence_artifacts[evidence_path.name] = {
                "sha256": sha256_file(evidence_path),
                "rows": input_rows,
            }
            print(f"judged {path.name}: {input_rows} rows", flush=True)
        submission_handle.flush()
        os.fsync(submission_handle.fileno())

    if judge_errors:
        submission_temp.unlink(missing_ok=True)
        error_path = results_dir / "judge_errors.json"
        error_path.write_text(
            json.dumps(
                {"count": len(judge_errors), "errors": judge_errors[:100]},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise SystemExit(
            f"The judge failed for {len(judge_errors)} item(s). No submission file was produced. "
            f"Inspect {error_path} and rerun the judge."
        )

    expected_counts = inference_manifest.get("condition_counts") or {}
    for condition in REQUIRED_CONDITIONS:
        if condition_counts[condition] != int(expected_counts.get(condition) or 0):
            submission_temp.unlink(missing_ok=True)
            raise ValueError(
                f"Condition {condition} produced {condition_counts[condition]} judged rows; "
                f"expected {expected_counts.get(condition)}"
            )

    submission_path = results_dir / "submission.jsonl"
    submission_temp.replace(submission_path)
    report = _aggregate_report(group_states, datasets)
    report.update(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "model": inference_manifest.get("model"),
            "conditions": list(REQUIRED_CONDITIONS),
        }
    )
    report_path = results_dir / "leaderboard.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inference_manifest_path = results_dir / "inference_manifest.json"
    run_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "submission_schema_version": SUBMISSION_SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": inference_manifest.get("model"),
        "datasets": datasets,
        "conditions": list(REQUIRED_CONDITIONS),
        "dataset_files": inference_manifest.get("dataset_files"),
        "prompts": inference_manifest.get("prompts"),
        "decoding": inference_manifest.get("decoding"),
        "ablation_manifest": inference_manifest.get("ablation_manifest"),
        "benchmark_manifest_sha256": inference_manifest.get(
            "benchmark_manifest_sha256"
        ),
        "judge": {
            "served_model": endpoint_model,
            "revision": args.judge_revision,
            "system_prompt_sha256": sha256_bytes(JUDGE_SYS.encode("utf-8")),
            "method_counts": dict(judge_method_counts),
            "decoding": JUDGE_DECODING,
        },
        "condition_counts": dict(condition_counts),
        "error_counts": {
            "inference": 0,
            "judge": 0,
            "missing_outputs": 0,
        },
        "artifacts": {
            "submission": {
                "filename": submission_path.name,
                "sha256": sha256_file(submission_path),
                "rows": sum(condition_counts.values()),
            },
            "leaderboard_report": {
                "filename": report_path.name,
                "sha256": sha256_file(report_path),
                "size_bytes": report_path.stat().st_size,
                "dataset_count": len(datasets),
            },
            "inference_manifest": {
                "filename": inference_manifest_path.name,
                "sha256": sha256_file(inference_manifest_path),
            },
            "judged_evidence": evidence_artifacts,
        },
        "debug": False,
    }
    run_manifest_path = results_dir / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    package_path = _write_submission_package(
        results_dir,
        submission_path,
        run_manifest_path,
        report_path,
    )
    print(f"\nwrote {submission_path}")
    print(f"wrote {run_manifest_path}")
    print(f"wrote {package_path} (upload this file)")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
