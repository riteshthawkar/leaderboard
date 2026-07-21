"""Run the public model-only answer extraction phase over saved inference."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from evaluation.answer_extraction_contract import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    METHOD,
    SYSTEM_PROMPT,
    answer_format,
    extractor_contract_sha256,
    response_format,
)
from evaluation.common.visual_pipeline import (
    EvaluationPipelineError,
    MISSING_ANSWER_TOKEN,
    export_submission,
    load_questions,
    read_diagnostics,
    write_diagnostics,
)
from evaluation.common.vllm_runner import INFERENCE_METHOD, _message_text
from visual_answer_contract import task_from_question_id


TRACKS = ("do_you_see_me", "minds_eye")


def _parse_answer(output: str) -> str | None:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        return None
    answer = payload["answer"]
    return answer if isinstance(answer, str) else None


def _load_rows(path: Path, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_diagnostics([path])
    expected_ids = [str(question["question_id"]) for question in questions]
    if [str(row.get("question_id") or "") for row in rows] != expected_ids:
        raise EvaluationPipelineError(f"{path} does not match question coverage/order.")
    for index, row in enumerate(rows, 1):
        output = row.get("output")
        if not isinstance(output, str) or not output.strip():
            raise EvaluationPipelineError(f"{path} row {index} has no response.")
        if row.get("error") or row.get("inference_error"):
            raise EvaluationPipelineError(f"{path} row {index} has an inference error.")
        digest = hashlib.sha256(output.encode("utf-8")).hexdigest()
        if (
            row.get("inference_method") != INFERENCE_METHOD
            or row.get("inference_output_sha256") != digest
        ):
            raise EvaluationPipelineError(f"{path} row {index} has invalid provenance.")
    return rows


def _current(row: dict[str, Any], *, model: str, revision: str, contract: str) -> bool:
    digest = hashlib.sha256(str(row.get("output") or "").encode("utf-8")).hexdigest()
    return bool(
        row.get("answer_extraction_method") == METHOD
        and row.get("extractor_model") == model
        and row.get("extractor_revision") == revision
        and row.get("extractor_contract_sha256") == contract
        and row.get("extractor_source_output_sha256") == digest
        and row.get("extractor_status") in {"extracted", "empty"}
        and "extracted_answer" in row
    )


async def _extract_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    question: dict[str, Any],
    args: argparse.Namespace,
    contract: str,
) -> dict[str, Any]:
    result = dict(row)
    answer_type = str(question.get("answer_type") or row.get("answer_type") or "text")
    task = str(question.get("task") or task_from_question_id(str(question["question_id"])))
    payload = {
        "candidate_response": str(row["output"]),
        "expected_answer_format": answer_format(answer_type, task)["description"],
    }
    extractor_output = ""
    finish_reason = None
    completion_tokens = None
    answer: str | None = None
    error = None
    async with semaphore:
        for attempt in range(args.retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    temperature=0,
                    top_p=1,
                    seed=0,
                    max_tokens=args.max_tokens,
                    response_format=response_format(answer_type, task),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = response.choices[0]
                extractor_output = _message_text(choice.message.content)
                finish_reason = getattr(choice, "finish_reason", None)
                usage = getattr(response, "usage", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                answer = _parse_answer(extractor_output)
                error = None if answer is not None else "Extractor returned invalid JSON."
            except Exception as exc:  # noqa: BLE001 - transport/schema retry
                error = f"{type(exc).__name__}: {exc}"[:500]
                answer = None
            if answer is not None or attempt == args.retries:
                break
            await asyncio.sleep(min(2**attempt, 8))
    if answer is None:
        raise EvaluationPipelineError(
            f"Extractor failed for {question['question_id']}: {error}"
        )
    result.update(
        {
            "answer_extraction_method": METHOD,
            "extractor_model": args.model,
            "extractor_revision": args.revision,
            "extractor_contract_sha256": contract,
            "extractor_output": extractor_output,
            "extractor_status": "extracted" if answer else "empty",
            "extractor_finish_reason": finish_reason,
            "extractor_completion_tokens": completion_tokens,
            "extractor_source_diagnostics": args.source.name,
            "extractor_source_output_sha256": hashlib.sha256(
                str(row["output"]).encode("utf-8")
            ).hexdigest(),
            "extractor_answer": answer,
            "extracted_answer": answer or MISSING_ANSWER_TOKEN,
        }
    )
    result.pop("extractor_error", None)
    return result


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    questions = load_questions(args.questions, args.track_config)
    source_rows = _load_rows(args.source, questions)
    contract = extractor_contract_sha256(args.model, args.max_tokens, args.revision)
    existing: dict[str, dict[str, Any]] = {}
    if args.diagnostics.is_file():
        existing = {
            str(row.get("question_id") or ""): row
            for row in read_diagnostics([args.diagnostics])
        }
    rows_by_id = {str(row["question_id"]): row for row in source_rows}
    for question_id, row in existing.items():
        source = rows_by_id.get(question_id)
        if source and row.get("output") == source.get("output") and _current(
            row, model=args.model, revision=args.revision, contract=contract
        ):
            rows_by_id[question_id] = row
    pending = [
        question
        for question in questions
        if not _current(
            rows_by_id[str(question["question_id"])],
            model=args.model,
            revision=args.revision,
            contract=contract,
        )
    ]
    endpoints = [
        endpoint.strip().rstrip("/")
        for value in args.endpoint
        for endpoint in value.split(",")
        if endpoint.strip()
    ]
    if pending and not endpoints:
        raise EvaluationPipelineError("Extraction is incomplete and no endpoint was provided.")
    clients = [
        AsyncOpenAI(base_url=endpoint, api_key=args.api_key, timeout=args.timeout, max_retries=0)
        for endpoint in endpoints
    ]
    semaphore = asyncio.Semaphore(args.concurrency)
    try:
        jobs = [
            _extract_one(
                clients[index % len(clients)],
                semaphore,
                rows_by_id[str(question["question_id"])],
                question,
                args,
                contract,
            )
            for index, question in enumerate(pending)
        ]
        completed = 0
        for future in asyncio.as_completed(jobs):
            row = await future
            rows_by_id[str(row["question_id"])] = row
            completed += 1
            if completed % args.checkpoint_every == 0 or completed == len(jobs):
                ordered = [rows_by_id[str(question["question_id"])] for question in questions]
                write_diagnostics(args.diagnostics, ordered, preserve_extra_fields=True)
                if completed % args.report_every == 0 or completed == len(jobs):
                    print(f"[{args.track}] extracted {completed}/{len(jobs)} new", flush=True)
    finally:
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)
    ordered = [rows_by_id[str(question["question_id"])] for question in questions]
    write_diagnostics(args.diagnostics, ordered, preserve_extra_fields=True)
    report = export_submission(ordered, questions, args.out, require_extracted_answers=True)
    return {
        **report,
        "extractor_contract_sha256": contract,
        "extracted_count": sum(row.get("extractor_status") == "extracted" for row in ordered),
        "empty_count": sum(row.get("extractor_status") == "empty" for row in ordered),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=DEFAULT_EXTRACTOR_MODEL)
    parser.add_argument("--revision", default=DEFAULT_EXTRACTOR_REVISION)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--report-every", type=int, default=100)
    args = parser.parse_args()
    from evaluation.do_you_see_me.config import TRACK as DYS_TRACK
    from evaluation.minds_eye.config import TRACK as MINDS_EYE_TRACK
    args.track_config = {"do_you_see_me": DYS_TRACK, "minds_eye": MINDS_EYE_TRACK}[args.track]
    return args


def main() -> int:
    args = parse_args()
    try:
        report = asyncio.run(_run(args))
    except (EvaluationPipelineError, OSError, RuntimeError) as exc:
        print(f"Answer extraction failed: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
