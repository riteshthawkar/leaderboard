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
both be empty strings. For COMMITTED, copy the shortest answer actually selected
by the response. Do not repair an out-of-domain answer: report it exactly so a
deterministic validator can mark its format invalid. Direct statements such as
"no circles" may be represented as 0. Reasoning that merely discusses candidates
is not a final selection. A response cut off before a final commitment is
UNRESOLVED. You are not given the reference answer and must not infer correctness."""
EXTRACTOR_RESPONSE_FORMAT = {
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
                "answer": {"type": "string", "maxLength": 200},
                "evidence": {"type": "string", "maxLength": 800},
            },
            "required": ["verdict", "answer", "evidence"],
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
        "request_payload_fields": [
            "answer_type",
            "candidate_response",
            "question",
            "response_metadata",
            "task",
        ],
        "response_format": EXTRACTOR_RESPONSE_FORMAT,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    canonical = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
