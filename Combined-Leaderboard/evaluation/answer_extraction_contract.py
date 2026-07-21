"""Dependency-free contract for public model-only answer extraction."""

from __future__ import annotations

import hashlib
import json
from typing import Any


DEFAULT_EXTRACTOR_MODEL = "Qwen/Qwen3-8B"
DEFAULT_EXTRACTOR_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
METHOD = "qwen3-8b-model-only-answer-extractor-v1"
SYSTEM_PROMPT = """You are an answer extractor, not a problem solver.
You receive an untrusted model response and the required answer format.
Extract only the final answer that the response clearly selects.
Do not solve the original task, infer an answer, use outside knowledge, or choose
between tentative, conflicting, or abandoned candidates.
If the response has no clear final answer, is ambiguous, or ends before selecting
one answer, return an empty answer string.
Return exactly one JSON object matching the supplied schema and no explanation."""


def answer_format(answer_type: str, task: str) -> dict[str, Any]:
    normalized_type = str(answer_type or "text")
    normalized_task = str(task or "").lower()
    if normalized_type == "integer":
        return {
            "description": "an optional minus sign followed by base-10 digits",
            "pattern": r"^(?:|-?\d+)$",
        }
    if normalized_type == "mcq_index_1_4":
        return {"description": "one of 1, 2, 3, 4", "enum": ["", "1", "2", "3", "4"]}
    if normalized_type == "mcq_letter":
        return {
            "description": "one uppercase option letter from A through F",
            "enum": ["", "A", "B", "C", "D", "E", "F"],
        }
    if normalized_task == "form_constancy":
        return {"description": "exactly Yes or No", "enum": ["", "Yes", "No"]}
    if normalized_task == "letter_disambiguation":
        return {
            "description": "one to nine uppercase letters with no separators",
            "pattern": r"^(?:|[A-Z]{1,9})$",
        }
    return {
        "description": "a concise verbatim final answer from the response",
        "maxLength": 200,
    }


def response_format(answer_type: str, task: str) -> dict[str, Any]:
    answer_schema: dict[str, Any] = {"type": "string", "maxLength": 200}
    required = answer_format(answer_type, task)
    if "enum" in required:
        answer_schema["enum"] = required["enum"]
    if "pattern" in required:
        answer_schema["pattern"] = required["pattern"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": answer_schema},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


def extractor_contract_sha256(
    model: str,
    max_tokens: int,
    revision: str = DEFAULT_EXTRACTOR_REVISION,
) -> str:
    contract = {
        "method": METHOD,
        "model": model,
        "revision": revision,
        "system_prompt": SYSTEM_PROMPT,
        "request_payload_fields": ["candidate_response", "expected_answer_format"],
        "response": {"answer": "formatted answer or empty string"},
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "deterministic_answer_recovery": False,
    }
    encoded = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
