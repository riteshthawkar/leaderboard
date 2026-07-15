"""OpenAI-compatible inference runner for one visual benchmark track."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .visual_pipeline import (
    EvaluationPipelineError,
    VisualTrackConfig,
    export_submission,
    final_answer,
    image_for_openai,
    load_prompt,
    load_questions,
    read_diagnostics,
    write_diagnostics,
)


def _endpoints(value: str) -> list[str]:
    endpoints = [item.strip().rstrip("/") for item in value.split(",") if item.strip()]
    if not endpoints:
        raise EvaluationPipelineError("Provide at least one OpenAI-compatible endpoint.")
    invalid = [item for item in endpoints if urlparse(item).scheme not in {"http", "https"}]
    if invalid:
        raise EvaluationPipelineError(
            "Every endpoint must be an absolute HTTP(S) URL. Invalid: " + ", ".join(invalid)
        )
    return endpoints


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object ({exc.msg})") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def _message_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif hasattr(item, "text"):
                parts.append(str(item.text or ""))
        return "\n".join(part for part in parts if part).strip()
    return "" if content is None else str(content).strip()


async def _infer_one(
    client,
    semaphore: asyncio.Semaphore,
    item: dict,
    *,
    image_root: Path | None,
    system_prompt: str,
    model: str,
    max_tokens: int,
    seed: int,
    extra_body: dict[str, Any],
) -> dict:
    result = dict(item)
    async with semaphore:
        try:
            image = image_for_openai(item, image_root)
            request: dict[str, Any] = {
                "model": model,
                "temperature": 0,
                "seed": seed,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": item["question"]},
                            {"type": "image_url", "image_url": {"url": image}},
                        ],
                    },
                ],
            }
            if extra_body:
                request["extra_body"] = extra_body
            response = await client.chat.completions.create(
                **request,
            )
            result["output"] = _message_text(response.choices[0].message.content)
            if not result["output"]:
                result["error"] = "The model returned an empty response."
        except Exception as exc:  # Each failure is retained for a targeted rerun.
            result["output"] = None
            result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return result


def _usable_record(record: dict[str, Any], question: dict[str, Any]) -> bool:
    if record.get("error"):
        return False
    return bool(
        final_answer(record.get("output"), str(question.get("answer_type") or "text"))
    )


def _resume_records(
    diagnostics_path: Path,
    selected: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not diagnostics_path.is_file():
        return {}

    questions_by_id = {str(item["question_id"]): item for item in selected}
    rows = read_diagnostics([diagnostics_path])
    records_by_id: dict[str, dict[str, Any]] = {}
    unknown_ids: list[str] = []
    duplicate_ids: list[str] = []
    for row in rows:
        question_id = str(row.get("question_id") or "").strip()
        if question_id not in questions_by_id:
            unknown_ids.append(question_id or "<missing>")
            continue
        if question_id in records_by_id:
            duplicate_ids.append(question_id)
        records_by_id[question_id] = row

    if unknown_ids or duplicate_ids:
        details = []
        if unknown_ids:
            details.append(f"unknown question IDs: {', '.join(unknown_ids[:5])}")
        if duplicate_ids:
            details.append(f"duplicate question IDs: {', '.join(duplicate_ids[:5])}")
        raise EvaluationPipelineError(
            "Cannot resume from the diagnostics file because it contains "
            + "; ".join(details)
            + ". Remove the file or use the matching benchmark run."
        )

    return {
        question_id: row
        for question_id, row in records_by_id.items()
        if _usable_record(row, questions_by_id[question_id])
    }


def _invalid_records(
    records: list[dict[str, Any]], questions: list[dict[str, Any]]
) -> list[str]:
    questions_by_id = {str(item["question_id"]): item for item in questions}
    records_by_id = {str(item.get("question_id") or ""): item for item in records}
    invalid: list[str] = []
    for question_id, question in questions_by_id.items():
        record = records_by_id.get(question_id)
        if record is None:
            invalid.append(f"{question_id} (missing)")
        elif record.get("error"):
            invalid.append(f"{question_id} (inference error)")
        elif not _usable_record(record, question):
            invalid.append(f"{question_id} (empty or unparseable output)")
    return invalid


async def _run(
    args,
    track: VisualTrackConfig,
    diagnostics_path: Path,
) -> tuple[list[dict], list[dict]]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise EvaluationPipelineError(
            "The openai package is required. Install evaluation/requirements-vllm.txt."
        ) from exc

    questions = load_questions(args.questions, track)
    selected = questions[: args.limit] if args.limit else questions
    prompt = load_prompt(track, args.prompt_mode)
    endpoints = _endpoints(args.endpoints)
    clients = [
        AsyncOpenAI(
            base_url=endpoint,
            api_key=args.api_key,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        for endpoint in endpoints
    ]
    completed_by_id = (
        _resume_records(diagnostics_path, selected) if args.resume else {}
    )
    pending = [
        item for item in selected if str(item["question_id"]) not in completed_by_id
    ]
    if completed_by_id:
        print(
            f"[{track.task_id}] resuming with {len(completed_by_id)}/{len(selected)} "
            "validated responses",
            flush=True,
        )

    extra_body = dict(args.extra_body)
    if args.chat_template_kwargs:
        extra_body["chat_template_kwargs"] = args.chat_template_kwargs

    semaphore = asyncio.Semaphore(args.concurrency)
    jobs = [
        _infer_one(
            clients[index % len(clients)],
            semaphore,
            item,
            image_root=args.image_root,
            system_prompt=prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            seed=args.seed,
            extra_body=extra_body,
        )
        for index, item in enumerate(pending)
    ]

    started = time.monotonic()
    try:
        for future in asyncio.as_completed(jobs):
            record = await future
            completed_by_id[str(record["question_id"])] = record
            new_count = len(completed_by_id) - (len(selected) - len(pending))
            total_count = len(completed_by_id)
            if new_count % args.checkpoint_every == 0 or total_count == len(selected):
                ordered_checkpoint = [
                    completed_by_id[str(item["question_id"])]
                    for item in selected
                    if str(item["question_id"]) in completed_by_id
                ]
                write_diagnostics(diagnostics_path, ordered_checkpoint)
            if new_count % 100 == 0 or total_count == len(selected):
                rate = new_count / max(time.monotonic() - started, 0.001)
                print(
                    f"[{track.task_id}] {total_count}/{len(selected)} "
                    f"({rate:.1f} new samples/s)",
                    flush=True,
                )
    finally:
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)

    order = {item["question_id"]: index for index, item in enumerate(selected)}
    completed = list(completed_by_id.values())
    completed.sort(key=lambda item: order[item["question_id"]])
    write_diagnostics(diagnostics_path, completed)
    return questions, completed


def _parser(track: VisualTrackConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run {track.label} and create a leaderboard-ready JSONL submission."
    )
    parser.add_argument("--model", required=True, help="Exact model identifier served by the endpoint")
    parser.add_argument(
        "--endpoints",
        default="http://localhost:8000/v1",
        help="Comma-separated OpenAI-compatible API base URLs",
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--questions", type=Path, default=track.questions_path)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Optional dataset root; image_url is used when a local image is unavailable",
    )
    parser.add_argument("--prompt-mode", choices=("noncot", "cot"), default="noncot")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--chat-template-kwargs",
        type=_json_object,
        default={},
        metavar="JSON",
        help="JSON object passed to the model chat template",
    )
    parser.add_argument(
        "--extra-body",
        type=_json_object,
        default={},
        metavar="JSON",
        help="Additional JSON fields passed to the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep valid responses in the diagnostics file and retry the rest",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Atomically save diagnostics after this many new responses",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Diagnostic partial run; partial runs never create an upload file",
    )
    parser.add_argument("--out", type=Path, help="Canonical submission JSONL destination")
    parser.add_argument("--diagnostics", type=Path, help="Raw-output diagnostics destination")
    parser.add_argument(
        "--strict-partial",
        action="store_true",
        help="Fail a limited smoke run if any response is missing or unparseable",
    )
    return parser


def main(track: VisualTrackConfig, argv: list[str] | None = None) -> int:
    args = _parser(track).parse_args(argv)
    if args.limit < 0:
        print("Evaluation failed: --limit cannot be negative.", file=sys.stderr)
        return 2
    if (
        args.concurrency < 1
        or args.max_retries < 0
        or args.request_timeout <= 0
        or args.checkpoint_every < 1
    ):
        print(
            "Evaluation failed: concurrency, timeout, and checkpoint-every must be positive, "
            "and retries cannot be negative.",
            file=sys.stderr,
        )
        return 2
    if args.max_tokens is None:
        args.max_tokens = 2048 if args.prompt_mode == "cot" else 256
    if args.max_tokens < 1:
        print("Evaluation failed: --max-tokens must be positive.", file=sys.stderr)
        return 2

    output_path = args.out or track.default_output_path()
    diagnostics_path = args.diagnostics or output_path.with_name(
        f"{output_path.stem}.{args.prompt_mode}.diagnostics.jsonl"
    )
    try:
        all_questions, records = asyncio.run(_run(args, track, diagnostics_path))
        write_diagnostics(diagnostics_path, records)
        if args.limit:
            selected_questions = all_questions[: args.limit]
            invalid = _invalid_records(records, selected_questions)
            if args.strict_partial and invalid:
                preview = ", ".join(invalid[:5])
                raise EvaluationPipelineError(
                    f"Smoke run failed: {len(invalid)} response(s) were invalid, including {preview}."
                )
            print(
                f"Partial run complete. Wrote {len(records)} diagnostic rows to {diagnostics_path}. "
                "No submission file was created.",
                flush=True,
            )
            return 0
        report = export_submission(records, all_questions, output_path)
    except (EvaluationPipelineError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
