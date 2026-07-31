#!/usr/bin/env python3
"""Solve benchmark questions from validated text states without images."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common.visual_pipeline import MISSING_ANSWER_TOKEN  # noqa: E402
from evaluation.common.vllm_runner import _extract_one, _message_text  # noqa: E402
from evaluation.research.common import read_jsonl, write_jsonl  # noqa: E402
from evaluation.research.run_state_extractor_vllm import (  # noqa: E402
    task_for,
    validate_state,
)


SYSTEM_PROMPTS = {
    "noncot": (
        "Solve the question using only the supplied answer-blind structured visual "
        "state. Do not assume access to the image. Return a clear final answer in "
        "the answer domain requested by the question."
    ),
    "cot": (
        "Solve the question using only the supplied answer-blind structured visual "
        "state. Reason through the transformation or relation before committing to "
        "a clear final answer in the answer domain requested by the question."
    ),
}


async def infer_and_extract(
    *,
    inference_client: Any,
    extractor_client: Any,
    semaphore: asyncio.Semaphore,
    question: dict[str, Any],
    state_record: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    question_id = str(question["question_id"])
    state = state_record["state"]
    task = str(state_record["task"])
    validate_state(task, state)
    user_payload = {
        "question": question["question"],
        "answer_type": question.get("answer_type", "text"),
        "visual_state": state,
    }
    result: dict[str, Any] = {
        "question_id": question_id,
        "question": question["question"],
        "answer_type": question.get("answer_type", "text"),
        "condition": args.condition,
        "reasoner_model": args.model,
        "reasoner_revision": args.model_revision,
        "state_model": state_record.get("state_model"),
        "state_model_revision": state_record.get("state_model_revision"),
        "state_sha256": hashlib.sha256(
            json.dumps(state, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    async with semaphore:
        try:
            request: dict[str, Any] = {
                "model": args.model,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPTS[args.reasoning_mode],
                    },
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
            }
            extra_body = dict(args.extra_body)
            if args.chat_template_kwargs:
                extra_body["chat_template_kwargs"] = args.chat_template_kwargs
            if extra_body:
                request["extra_body"] = extra_body
            response = await inference_client.chat.completions.create(**request)
            choice = response.choices[0]
            result["output"] = _message_text(choice.message.content)
            result["finish_reason"] = getattr(choice, "finish_reason", None)
            if not result["output"]:
                result["inference_error"] = "The reasoner returned an empty response."
        except Exception as exc:
            result["output"] = None
            result["inference_error"] = f"{type(exc).__name__}: {exc}"[:1000]
        extracted = await _extract_one(
            extractor_client,
            None,
            result,
            question,
            model=args.extractor_model,
            max_tokens=args.extractor_max_tokens,
            seed=args.extractor_seed,
            max_final_answer_tokens=None,
            source_diagnostics=str(args.diagnostics),
            chat_template_kwargs=args.extractor_chat_template_kwargs,
            extractor_revision=args.extractor_revision,
        )
    return extracted


def resumable(record: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        record.get("reasoner_model") == args.model
        and record.get("reasoner_revision") == args.model_revision
        and record.get("extractor_model") == args.extractor_model
        and record.get("extractor_revision") == args.extractor_revision
        and record.get("extractor_status") in {"resolved", "unresolved"}
        and not record.get("inference_error")
    )


async def run(args: argparse.Namespace) -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install evaluation/requirements-vllm.txt before running inference"
        ) from exc
    questions = read_jsonl(args.questions)
    states = {str(row["question_id"]): row for row in read_jsonl(args.states)}
    questions = [
        row for row in questions if str(row["question_id"]) in states
    ]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise ValueError("No question IDs overlap the state file")
    for question in questions:
        record = states[str(question["question_id"])]
        if record.get("status") != "validated":
            raise ValueError(
                f"{question['question_id']} has an unvalidated state record"
            )
        validate_state(task_for(record), record["state"])

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.diagnostics.is_file():
        for record in read_jsonl(args.diagnostics):
            if resumable(record, args):
                completed[str(record["question_id"])] = record
    pending = [
        question
        for question in questions
        if str(question["question_id"]) not in completed
    ]
    inference_client = AsyncOpenAI(
        base_url=args.endpoint.rstrip("/"),
        api_key=args.api_key,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    extractor_client = AsyncOpenAI(
        base_url=args.extractor_endpoint.rstrip("/"),
        api_key=args.extractor_api_key,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    jobs = [
        infer_and_extract(
            inference_client=inference_client,
            extractor_client=extractor_client,
            semaphore=semaphore,
            question=question,
            state_record=states[str(question["question_id"])],
            args=args,
        )
        for question in pending
    ]
    for completed_count, future in enumerate(asyncio.as_completed(jobs), start=1):
        record = await future
        completed[str(record["question_id"])] = record
        if (
            completed_count % args.checkpoint_every == 0
            or completed_count == len(jobs)
        ):
            ordered = [
                completed[str(question["question_id"])]
                for question in questions
                if str(question["question_id"]) in completed
            ]
            write_jsonl(args.diagnostics, ordered)
            print(f"Text intervention: {len(ordered)}/{len(questions)} persisted", flush=True)
    await inference_client.close()
    await extractor_client.close()

    ordered = [completed[str(question["question_id"])] for question in questions]
    infrastructure_failures = [
        record
        for record in ordered
        if record.get("inference_error") or record.get("extractor_status") == "failed"
    ]
    if infrastructure_failures:
        print(
            f"{len(infrastructure_failures)} infrastructure or extractor failure(s) "
            "remain. Rerun with --resume; no submission was exported.",
            file=sys.stderr,
        )
        return 2
    submissions = [
        {
            "question_id": record["question_id"],
            "condition": args.condition,
            "answer": (
                record.get("extracted_answer")
                if record.get("extractor_status") == "resolved"
                else MISSING_ANSWER_TOKEN
            ),
        }
        for record in ordered
    ]
    write_jsonl(args.output, submissions)
    print(f"Wrote {len(submissions)} text-only answers to {args.output}")
    return 0


def json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "local"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--extractor-endpoint")
    parser.add_argument("--extractor-api-key")
    parser.add_argument("--extractor-model", required=True)
    parser.add_argument("--extractor-revision", required=True)
    parser.add_argument("--extractor-max-tokens", type=int, default=512)
    parser.add_argument("--extractor-seed", type=int, default=0)
    parser.add_argument("--reasoning-mode", choices=("noncot", "cot"), default="noncot")
    parser.add_argument("--condition", default="state_only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-body", type=json_object, default={})
    parser.add_argument("--chat-template-kwargs", type=json_object, default={})
    parser.add_argument(
        "--extractor-chat-template-kwargs",
        type=json_object,
        default={"enable_thinking": False},
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.questions = args.questions.resolve()
    args.states = args.states.resolve()
    args.output = args.output.resolve()
    args.diagnostics = args.diagnostics.resolve()
    args.extractor_endpoint = args.extractor_endpoint or args.endpoint
    args.extractor_api_key = args.extractor_api_key or args.api_key
    if (
        args.concurrency < 1
        or args.max_tokens < 1
        or args.extractor_max_tokens < 1
        or args.checkpoint_every < 1
    ):
        parser.error("token, concurrency, and checkpoint values must be positive")
    return args


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
