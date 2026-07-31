#!/usr/bin/env python3
"""Extract answer-blind structured visual states through a vLLM endpoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common.visual_pipeline import image_for_openai  # noqa: E402
from evaluation.research.common import read_jsonl, write_jsonl  # noqa: E402


DEFAULT_SCHEMAS = (
    PROJECT_ROOT / "evaluation/research/results/state_interventions/state_schemas.json"
)
FORBIDDEN_KEYS = {
    "answer",
    "choice",
    "correct_answer",
    "correct_option",
    "final_answer",
    "option",
    "prediction",
    "selected_option",
    "solution",
}
FORBIDDEN_TEXT = re.compile(
    r"\b(?:correct|final)\s+(?:answer|option)\b"
    r"|\banswer\s+is\b"
    r"|\b(?:choose|select)\s+(?:option\s+)?[A-F]\b"
    r"|\boption\s+[A-F]\s+(?:is|should|appears)\b"
    r"|\b(?:outlier|odd\s+one)\s+(?:is|:)\s*(?:(?:option|panel|figure)\s*)?[A-F]\b",
    re.IGNORECASE,
)
SYSTEM_PROMPT = """You are a visual state transcriber, not a problem solver.
Describe only directly visible evidence from the image using the supplied JSON
schema. Do not infer the requested answer, predict a future state, identify an
outlier, rank candidates, or state which option should be selected. Return one
valid JSON object and no Markdown or explanatory text."""


def message_text(content: Any) -> str:
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


def parse_json_object(raw: str) -> dict[str, Any]:
    value = raw.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("state response must be a JSON object")
    return parsed


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).casefold())
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def validate_state(task: str, state: dict[str, Any]) -> None:
    root = "frames" if task == "dynamic_isomorphism" else "panels"
    if root not in state or not isinstance(state[root], list) or not state[root]:
        raise ValueError(f"state must contain a non-empty '{root}' list")
    keys = nested_keys(state)
    forbidden = sorted(
        key
        for key in keys
        if key in FORBIDDEN_KEYS
        or any(
            fragment in key
            for fragment in (
                "correct_answer",
                "correct_option",
                "final_answer",
                "selected_option",
            )
        )
    )
    if forbidden:
        raise ValueError(f"state contains answer-bearing keys: {forbidden}")
    serialized = json.dumps(state, sort_keys=True)
    if FORBIDDEN_TEXT.search(serialized):
        raise ValueError("state contains answer-bearing language")


def task_for(row: dict[str, Any]) -> str:
    task = str(row.get("task") or "").strip()
    if task:
        return task
    question_id = str(row.get("question_id") or "")
    for candidate in ("dynamic_isomorphism", "slippage"):
        if candidate in question_id:
            return candidate
    raise ValueError(f"Cannot infer task for {question_id}")


async def extract_one(
    *,
    client: Any,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    schema: dict[str, Any],
    image_root: Path | None,
    model: str,
    model_revision: str,
    max_tokens: int,
    seed: int,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    task = task_for(row)
    prompt = (
        f"Task family: {task}\n"
        f"Required state description: {schema['description']}\n"
        f"Expected fields: {', '.join(schema['required_fields'])}\n"
        "Transcribe the image now. Return only JSON."
    )
    result: dict[str, Any] = {
        "question_id": str(row["question_id"]),
        "task": task,
        "state_model": model,
        "state_model_revision": model_revision,
        "state_prompt_sha256": hashlib.sha256(
            (SYSTEM_PROMPT + "\n" + prompt).encode("utf-8")
        ).hexdigest(),
    }
    async with semaphore:
        try:
            image = image_for_openai(row, image_root)
            request: dict[str, Any] = {
                "model": model,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": seed,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image}},
                        ],
                    },
                ],
            }
            if extra_body:
                request["extra_body"] = extra_body
            response = await client.chat.completions.create(**request)
            choice = response.choices[0]
            raw_output = message_text(choice.message.content)
            result["raw_output"] = raw_output
            result["finish_reason"] = getattr(choice, "finish_reason", None)
            state = parse_json_object(raw_output)
            validate_state(task, state)
            result["state"] = state
            result["status"] = "validated"
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"[:1000]
    return result


async def run(args: argparse.Namespace) -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install evaluation/requirements-vllm.txt before running inference"
        ) from exc
    rows = read_jsonl(args.questions)
    rows = [
        row for row in rows if task_for(row) in {"dynamic_isomorphism", "slippage"}
    ]
    if args.limit:
        rows = rows[: args.limit]
    schemas = json.loads(args.schemas.read_text(encoding="utf-8"))
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.is_file():
        for record in read_jsonl(args.output):
            if (
                record.get("status") == "validated"
                and record.get("state_model") == args.model
                and record.get("state_model_revision") == args.model_revision
            ):
                completed[str(record["question_id"])] = record
    pending = [row for row in rows if str(row["question_id"]) not in completed]
    client = AsyncOpenAI(
        base_url=args.endpoint.rstrip("/"),
        api_key=args.api_key,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    jobs = [
        extract_one(
            client=client,
            semaphore=semaphore,
            row=row,
            schema=schemas[task_for(row)],
            image_root=args.image_root,
            model=args.model,
            model_revision=args.model_revision,
            max_tokens=args.max_tokens,
            seed=args.seed,
            extra_body=args.extra_body,
        )
        for row in pending
    ]
    for completed_count, future in enumerate(asyncio.as_completed(jobs), start=1):
        record = await future
        completed[str(record["question_id"])] = record
        if (
            completed_count % args.checkpoint_every == 0
            or completed_count == len(jobs)
        ):
            ordered = [
                completed[str(row["question_id"])]
                for row in rows
                if str(row["question_id"]) in completed
            ]
            write_jsonl(args.output, ordered)
            print(
                f"State extraction: {len(ordered)}/{len(rows)} persisted",
                flush=True,
            )
    await client.close()
    failed = [
        record
        for record in completed.values()
        if record.get("status") != "validated"
    ]
    if failed:
        print(
            f"State extraction has {len(failed)} failed row(s). "
            "Rerun with --resume after inspecting the output.",
            file=sys.stderr,
        )
        return 2
    return 0


def json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--schemas", type=Path, default=DEFAULT_SCHEMAS)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "local"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-body", type=json_object, default={})
    parser.add_argument("--chat-template-kwargs", type=json_object, default={})
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.questions = args.questions.resolve()
    args.schemas = args.schemas.resolve()
    args.output = args.output.resolve()
    args.image_root = args.image_root.resolve() if args.image_root else None
    if args.chat_template_kwargs:
        args.extra_body = {
            **args.extra_body,
            "chat_template_kwargs": args.chat_template_kwargs,
        }
    if args.concurrency < 1 or args.max_tokens < 1 or args.checkpoint_every < 1:
        parser.error("concurrency, max tokens, and checkpoint interval must be positive")
    return args


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
