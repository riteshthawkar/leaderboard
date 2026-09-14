"""Dependency-free definition of the production visual evidence contract."""

from __future__ import annotations

import hashlib
import json

from visual_answer_contract import PRODUCTION_EXTRACTION_METHOD


DEFAULT_EXTRACTOR_MODEL = "Qwen/Qwen3-8B"
DEFAULT_EXTRACTOR_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
METHOD = PRODUCTION_EXTRACTION_METHOD
SYSTEM_PROMPT = """You are a forensic response auditor, not a problem solver.
You receive an original question, its answer contract, response metadata, and an
untrusted candidate model response. Determine only whether the candidate response
explicitly commits to one final answer or has no single final commitment. Never
solve the question, infer from an image, use outside knowledge, or choose an answer
merely because it is mentioned among candidates.

Return exactly one JSON object with three string fields:
{"verdict":"COMMITTED|UNRESOLVED","answer":"...","evidence":"..."}

The evidence must be a short exact quote copied verbatim from the candidate
response that expresses its commitment. For UNRESOLVED, answer and evidence must
both be empty strings. For COMMITTED, normalize only the explicitly selected final
answer into the supplied expected_answer_domain: for example, "Option 2" becomes
"2", "Figure D" becomes "D", and a letter list such as "E T O N" becomes "ETON".
The evidence remains verbatim even when answer is normalized. If the response
commits to multiple answers where exactly one is required, commits outside the
domain, or is cut off before selecting an answer, return UNRESOLVED. Direct
statements such as "no circles" may be represented as 0. Reasoning that merely
discusses candidates is not a final selection. You are not given the reference
answer and must not infer correctness."""


def expected_answer_domain(answer_type: str, task: str = "") -> str:
    normalized_type = str(answer_type or "text").strip()
    normalized_task = str(task or "").strip().casefold()
    if normalized_type == "integer":
        return "one base-10 integer written with digits"
    if normalized_type == "mcq_index_1_4":
        return "exactly one digit: 1, 2, 3, or 4"
    if normalized_type == "mcq_letter":
        return "exactly one uppercase option letter: A, B, C, D, E, or F"
    if normalized_task == "form_constancy":
        return "exactly Yes or No"
    if normalized_task == "letter_disambiguation":
        return "one to nine uppercase letters with no spaces or punctuation"
    return "one concise answer of at most 200 characters"


def _answer_pattern(answer_type: str, task: str = "") -> str:
    normalized_type = str(answer_type or "text").strip()
    normalized_task = str(task or "").strip().casefold()
    if normalized_type == "integer":
        return r"^(?:|-?[0-9]+)$"
    if normalized_type == "mcq_index_1_4":
        return r"^(?:|[1-4])$"
    if normalized_type == "mcq_letter":
        return r"^(?:|[A-F])$"
    if normalized_task == "form_constancy":
        return r"^(?:|Yes|No)$"
    if normalized_task == "letter_disambiguation":
        return r"^(?:|[A-Z]{1,9})$"
    return r"^.{0,200}$"


def extractor_response_format(answer_type: str, task: str = "") -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "commitment_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["COMMITTED", "UNRESOLVED"],
                    },
                    "answer": {
                        "type": "string",
                        "pattern": _answer_pattern(answer_type, task),
                        "maxLength": 200,
                    },
                    "evidence": {"type": "string", "maxLength": 800},
                },
                "required": ["verdict", "answer", "evidence"],
                "additionalProperties": False,
            },
        },
    }


EXTRACTOR_RESPONSE_FORMAT = extractor_response_format("text")


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
        "request_payload_fields": [
            "answer_type",
            "candidate_response",
            "question",
            "response_metadata",
            "task",
        ],
        "response_formats": {
            "integer": extractor_response_format("integer"),
            "mcq_index_1_4": extractor_response_format("mcq_index_1_4"),
            "mcq_letter": extractor_response_format("mcq_letter"),
            "form_constancy": extractor_response_format("text", "form_constancy"),
            "letter_disambiguation": extractor_response_format(
                "text", "letter_disambiguation"
            ),
            "text": EXTRACTOR_RESPONSE_FORMAT,
        },
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    canonical = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
