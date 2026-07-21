"""Audit non-format visual responses with a gold-blind commitment extractor."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from evaluation.common.visual_pipeline import INTEGER_WORDS, final_answer
from evaluation.common.vllm_runner import _answer_is_supported_by_output
from visual_answer_contract import (
    INVALID_FORMAT_TOKEN,
    PRODUCTION_EXTRACTION_METHOD,
    canonicalize_extracted_answer,
)


TRACKS = ("do_you_see_me", "minds_eye")
DEFAULT_EXTRACTOR_MODEL = "Qwen/Qwen3-8B"
DEFAULT_EXTRACTOR_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_EXCLUDED_VARIANTS = (
    "qwen-35-thinking-disabled",
    "qwen35-thinking-enabled",
)
METHOD = PRODUCTION_EXTRACTION_METHOD
TERMINAL_FALLBACK_METHOD = "deterministic-terminal-response-classifier-v1"
VERDICTS = {"COMMITTED", "UNRESOLVED"}
BLOCKING_EXTRACTOR_STATUSES = {
    "request_error",
    "invalid_extractor_output",
    "invalid_verdict",
    "invalid_unresolved_payload",
    "invalid_committed_payload",
}
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


class GroundTruthError(ValueError):
    pass


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def contract_exact(track: str, answer_type: str, output: str) -> bool:
    output = output.strip()
    if track == "minds_eye":
        return bool(
            re.fullmatch(
                r"(?:<think>.*?</think>\s*)?<answer>\s*[A-F]\s*</answer>",
                output,
                flags=re.I | re.S,
            )
        )
    if answer_type == "integer":
        return bool(re.fullmatch(r"-?\d+", output))
    if answer_type == "mcq_index_1_4":
        return bool(re.fullmatch(r"[1-4]", output))
    if answer_type == "mcq_letter":
        return bool(re.fullmatch(r"[A-F]", output, flags=re.I))
    return bool(output and "\n" not in output and len(output) <= 200)


def candidate_category(track: str, answer_type: str, record: dict[str, Any]) -> str | None:
    output = str(record.get("output") or "")
    if contract_exact(track, answer_type, output):
        return None
    local = final_answer(output, answer_type)
    explicit = bool(
        re.search(
            r"(?is)<answer>.*?</answer>|\b(?:final\s+)?answer\s*(?:is|:|=)"
            r"|\b(?:option|choice)\s*[A-F1-4]",
            output,
        )
    )
    if explicit and local:
        return "deterministic_explicit"
    if local:
        return "heuristic_local_parse"
    extracted = record.get("extracted_answer")
    if extracted is not None and final_answer(extracted, answer_type):
        return "prior_model_extractor"
    return "unresolved_raw"


def parse_extractor_output(value: str) -> tuple[str, str, str] | None:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I | re.S)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "verdict",
        "answer",
        "evidence",
    }:
        return None
    if not all(isinstance(payload[field], str) for field in payload):
        return None
    verdict = payload["verdict"].strip().upper()
    answer = payload["answer"].strip()
    evidence = payload["evidence"].strip()
    return verdict, answer, evidence


def valid_answer(answer: str, answer_type: str) -> bool:
    if answer_type == "integer":
        return bool(re.fullmatch(r"-?\d+", answer))
    if answer_type == "mcq_index_1_4":
        return bool(re.fullmatch(r"[1-4]", answer))
    if answer_type == "mcq_letter":
        return bool(re.fullmatch(r"[A-F]", answer, flags=re.I))
    return bool(answer and len(answer) <= 200 and "\n" not in answer)


def answers_equal(left: Any, right: Any, answer_type: str) -> bool:
    left_answer = final_answer(left, answer_type)
    right_answer = final_answer(right, answer_type)
    if not left_answer or not right_answer:
        return False
    if answer_type == "text":
        return _normalize(left_answer) == _normalize(right_answer)
    return left_answer == right_answer


def _gold_entries(path: Path) -> list[tuple[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise GroundTruthError(f"Ground-truth file not found: {path}")

    def entry(question_id: Any, value: Any) -> tuple[str, Any]:
        question_id = str(question_id or "").strip()
        if not question_id:
            raise GroundTruthError(f"Ground-truth entry in {path} has no question_id.")
        if isinstance(value, dict):
            if "answer" not in value:
                raise GroundTruthError(
                    f"Ground-truth entry '{question_id}' in {path} has no answer."
                )
            value = value["answer"]
        return question_id, value

    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GroundTruthError(
                    f"Invalid JSON on {path} line {line_number}: {exc.msg}."
                ) from exc
            if not isinstance(row, dict):
                raise GroundTruthError(
                    f"Ground-truth line {line_number} in {path} is not an object."
                )
            rows.append(
                entry(row.get("question_id") or row.get("sample_id"), row)
            )
        return rows

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise GroundTruthError(f"Invalid JSON in {path}: {exc.msg}.") from exc
    if isinstance(payload, list):
        rows = []
        for row in payload:
            if not isinstance(row, dict):
                raise GroundTruthError(f"Ground-truth list in {path} contains a non-object.")
            rows.append(
                entry(row.get("question_id") or row.get("sample_id"), row)
            )
        return rows
    if not isinstance(payload, dict):
        raise GroundTruthError(f"Ground truth in {path} must be an object or list.")

    rows = []
    for question_id, value in payload.items():
        if question_id in TRACKS and isinstance(value, dict) and "answer" not in value:
            rows.extend(entry(nested_id, nested_value) for nested_id, nested_value in value.items())
        else:
            rows.append(entry(question_id, value))
    return rows


def load_gold_answers(
    project_root: Path, ground_truth_paths: list[Path]
) -> tuple[dict[str, str], str]:
    questions: dict[str, dict[str, str]] = {}
    ids_by_track: dict[str, set[str]] = {}
    for track in TRACKS:
        path = project_root / "tasks" / track / "questions.jsonl"
        track_ids = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            question_id = str(row["question_id"])
            if question_id in questions:
                raise GroundTruthError(f"Canonical question_id is duplicated: {question_id}")
            questions[question_id] = {
                "track": track,
                "answer_type": str(row.get("answer_type") or "text"),
            }
            track_ids.add(question_id)
        ids_by_track[track] = track_ids

    raw_answers: dict[str, Any] = {}
    for path in ground_truth_paths:
        for question_id, answer in _gold_entries(path):
            if question_id in raw_answers:
                raise GroundTruthError(f"Ground truth repeats question_id '{question_id}'.")
            raw_answers[question_id] = answer

    expected_ids = set(questions)
    supplied_ids = set(raw_answers)
    missing = expected_ids - supplied_ids
    unknown = supplied_ids - expected_ids
    if missing or unknown:
        missing_by_track = {
            track: len(ids_by_track[track] & missing) for track in TRACKS
        }
        details = []
        if missing:
            details.append("missing examples: " + ", ".join(sorted(missing)[:3]))
        if unknown:
            details.append("unknown examples: " + ", ".join(sorted(unknown)[:3]))
        raise GroundTruthError(
            "Ground truth must exactly cover all canonical questions: "
            f"{len(missing)} missing ({missing_by_track}), {len(unknown)} unknown"
            + ("; " + "; ".join(details) if details else "")
            + "."
        )

    answers: dict[str, str] = {}
    for question_id, metadata in questions.items():
        answer_type = metadata["answer_type"]
        answer = final_answer(raw_answers[question_id], answer_type)
        if not valid_answer(answer, answer_type):
            raise GroundTruthError(
                f"Ground truth for '{question_id}' is invalid for {answer_type}."
            )
        answers[question_id] = answer
    canonical = json.dumps(
        answers, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return answers, hashlib.sha256(canonical).hexdigest()


def commitment_verdict(answer: str, gold_answer: str, answer_type: str) -> str:
    return (
        "GOLD_COMMITTED"
        if answers_equal(answer, gold_answer, answer_type)
        else "OTHER_COMMITTED"
    )


def _evidence_states_answer(evidence: str, answer: str, answer_type: str) -> bool:
    if _answer_is_supported_by_output(evidence, answer, answer_type):
        return True
    if answer_type == "integer":
        value = int(answer)
        digits = {
            int(item.replace(",", ""))
            for item in re.findall(
                r"(?<![A-Za-z0-9])-?\d(?:[\d,]*\d)?(?![A-Za-z0-9])", evidence
            )
        }
        words = {
            INTEGER_WORDS[item.casefold()]
            for item in re.findall(
                rf"\b({'|'.join(INTEGER_WORDS)})\b", evidence, flags=re.I
            )
        }
        if value in digits | words:
            return True
        return value == 0 and bool(
            re.search(
                r"(?i)\b(?:no|none|zero|without)\b|\bdoes\s+not\s+(?:contain|show|have)\s+any\b",
                evidence,
            )
        )
    if answer_type == "mcq_index_1_4":
        names = {"1": "first", "2": "second", "3": "third", "4": "fourth"}
        return bool(
            re.search(
                rf"(?i)\b(?:option|choice|answer|figure)\s*(?:is\s*)?[\[(]?{answer}[\])]?\b"
                rf"|<answer>\s*{answer}\b"
                rf"|\\boxed\{{\s*{answer}\s*\}}"
                rf"|\bonly\s+[\[(]?{answer}[\])]?(?:\s+\w+){{0,3}}\s+(?:matches|fits|corresponds)\b"
                rf"|\b{names[answer]}\s+(?:option|choice|figure)\b",
                evidence,
            )
        )
    if answer_type == "mcq_letter":
        return bool(
            re.search(
                rf"(?i)\b(?:option|choice|answer|figure)\s*(?:is\s*)?[\[(]?{answer}[\])]?(?=\W|$)"
                rf"|<answer>\s*{answer}(?=\W|$)"
                rf"|<answer>\s*{answer}\s*answer>"
                rf"|\\boxed\{{\s*{answer}\s*\}}"
                rf"|\*{{1,2}}answer\s*:?\*{{1,2}}\s*:?\s*[\[(]?{answer}[\])]?(?=\W|$)"
                rf"|\bonly\s+[\[(]?{answer}[\])]?(?:\s+\w+){{0,3}}\s+(?:matches|fits|corresponds)\b"
                rf"|\b(?:the\s+)?figure\s+that\s+does\s+not\s+(?:adhere|follow)"
                rf".{{0,180}}?\s+is\s+\*{{0,2}}(?:figure\s+)?[\[(]?{answer}[\])]?\*{{0,2}}(?=\W|$)"
                rf"|\b{answer}\s+(?:is|does|would|appears|seems)\b",
                evidence,
            )
        )
    normalized_answer = re.sub(r"[\W_]+", "", answer.casefold())
    normalized_evidence = re.sub(r"[\W_]+", "", evidence.casefold())
    if normalized_answer and normalized_answer in normalized_evidence:
        return True
    return normalized_answer in {"noletter", "noletters", "none"} and bool(
        re.search(
            r"(?i)\b(?:no|without)\s+(?:discernible\s+|visible\s+)?letters?\b"
            r"|\b(?:unable|cannot|can't)\s+to\s+(?:identify|see|find)\s+any\s+letters?\b",
            evidence,
        )
        or re.search(
            r"(?i)\b(?:there\s+)?(?:do(?:es)?n't|do(?:es)?\s+not)\s+"
            r"(?:appear\s+to\s+be\s+)?any\s+(?:recognizable\s+)?letters?\b",
            evidence,
        )
    )


def _evidence_is_commitment(
    response: str, evidence: str, answer: str, answer_type: str
) -> bool:
    if re.search(r"(?i)\b(?:could|might|may|possibly|perhaps|unsure|unclear)\b", evidence):
        return False
    if answer_type in {"mcq_letter", "mcq_index_1_4"}:
        alternative = r"(?:option\s+)?[\[(]?[A-F1-4][\])]?(?=\W|$)"
        if re.search(
            rf"(?i){alternative}\s+(?:or|and/or)\s+{alternative}", evidence
        ):
            return False
    if answer_type == "integer" and re.search(
        r"(?i)\b(?:either\b|\d+\s+or\s+\d+\b)", evidence
    ):
        return False
    escaped = re.escape(answer)
    if answer_type in {"mcq_letter", "mcq_index_1_4"} and re.search(
        rf"(?i)(?:\b(?:option|choice|figure)\s*)?[\[(]?{escaped}[\])]?(?:\s+\w+){{0,2}}\s+"
        r"(?:is\s+not|isn't|cannot\s+be|can't\s+be|is\s+incorrect|is\s+wrong)"
        rf"|\b(?:option|choice|figure)\s*[\[(]?{escaped}[\])]?(?:\s+\w+){{0,6}}\s+"
        r"(?:does\s+not|doesn't|fails\s+to)\s+(?:match|fit|correspond)",
        evidence,
    ):
        return False

    normalized_evidence_answer = final_answer(evidence, answer_type)
    if normalized_evidence_answer and answers_equal(
        normalized_evidence_answer, answer, answer_type
    ):
        stripped_evidence = evidence.strip()
        if response.rstrip().endswith(stripped_evidence):
            return True
        escaped_evidence = re.escape(stripped_evidence)
        if re.search(
            rf"(?is)<answer>\s*{escaped_evidence}\s*</answer>"
            rf"|<\|begin_of_box\|>\s*{escaped_evidence}\s*<\|end_of_box\|>",
            response,
        ):
            return True

    bare_token = bool(
        re.fullmatch(
            r"[\[(]?(?:-?\d+|[A-F])[\])]?\s*[.,:]?", evidence.strip(), flags=re.I
        )
    )
    if bare_token and response.rstrip().casefold().endswith(
        evidence.strip().casefold()
    ):
        return True
    if bare_token and re.search(
        rf"(?is)<answer>\s*{escaped}(?=\W|$)"
        rf"|<\|begin_of_box\|>\s*{escaped}(?=\W|$)"
        rf"|\\boxed\{{\s*{escaped}\s*\}}",
        response,
    ):
        return True

    explicit = bool(
        re.search(
            rf"(?is)<answer>\s*{escaped}(?=\W|$)"
            rf"|<answer>\s*{escaped}\s*answer>"
            rf"|<\|begin_of_box\|>\s*{escaped}\s*<\|end_of_box\|>"
            rf"|\\boxed\{{\s*{escaped}\s*\}}"
            rf"|\*{{1,2}}answer\s*:?\*{{1,2}}\s*:?\s*[\[(]?{escaped}(?=\W|$)"
            rf"|\b(?:final\s+)?(?:answer|response)\s*(?:is|:|=|-)?\s*[\[(]?{escaped}(?=\W|$)"
            rf"|\b(?:the\s+)?correct\s+(?:option|choice|answer|figure)\s*(?:is|:|=|-)?\s*[\[(]?{escaped}(?=\W|$)"
            rf"|\b(?:choose|select(?:ed)?|pick(?:ed)?|conclude|go\s+with)\s*(?:option|choice|figure)?\s*[\[(]?{escaped}(?=\W|$)",
            evidence,
        )
    )
    if explicit:
        return True
    if answer_type in {"mcq_letter", "mcq_index_1_4"}:
        return bool(
            re.search(
                rf"(?is)\b(?:option|choice|figure)\s*[\[(]?{escaped}[\])]?\s+"
                r"(?:is\s+)?(?:the\s+)?(?:correct|selected|the\s+answer|the\s+match|best)\b"
                rf"|\b(?:option|choice|figure)\s*[\[(]?{escaped}[\])]?.{{0,180}}\b"
                r"(?:matches|fits|corresponds\s+to)\s+(?:the\s+)?"
                r"(?:expected|described|required)\b"
                rf"|\bonly\s+[\[(]?{escaped}[\])]?(?:\s+\w+){{0,3}}\s+(?:matches|fits|corresponds)\b"
                rf"|\b(?:the\s+)?figure\s+that\s+does\s+not\s+adhere"
                rf"(?:\s+to\s+(?:(?:this|the|a)\s+)?"
                rf"(?:common\s+|underlying\s+|recursive\s+)?(?:visual\s+)?concept)?"
                rf"\s+is\s+(?:figure\s+)?[\[(]?{escaped}[\])]?(?=\W|$)"
                rf"|\b(?:the\s+)?figure\s+that\s+does\s+not\s+(?:adhere|follow)"
                rf".{{0,180}}?\s+is\s+\*{{0,2}}(?:figure\s+)?[\[(]?{escaped}[\])]?\*{{0,2}}(?=\W|$)"
                rf"|\bfigure\s+\*{{0,2}}[\[(]?{escaped}[\])]?\*{{0,2}}\s+"
                rf"is\s+the\s+one\s+that\s+does\s+not\s+(?:adhere|follow)\b"
                rf"|\b(?:the\s+)?correct\s+rotational\s+transformation"
                rf".{{0,160}}?(?:is|matches)\s+(?:option\s+)?[\[(]?{escaped}[\])]?(?=\W|$)"
                rf"|\b(?:the\s+)?figure\s+that\s+can\s+be\s+constructed"
                rf".{{0,160}}?\s+is\s+(?:option\s+)?[\[(]?{escaped}[\])]?(?=\W|$)"
                rf"|\bfigure\s*[\[(]?{escaped}[\])]?\s+"
                r"(?:does\s+not|fails\s+to|violates|is\s+(?:the\s+)?(?:odd|different|asymmetric))\b",
                evidence,
            )
        )
    if answer_type == "integer":
        return bool(
            re.search(
                r"(?i)\b(?:there\s+(?:are|is)|i\s+(?:count|see|find)|count|total|number|that(?:'s|\s+is))\b",
                evidence,
            )
            or re.search(
                r"(?i)\b(?:the\s+)?answer\s+(?:should|would)\s+be\b",
                evidence,
            )
            or (answer == "0" and re.search(r"(?i)\b(?:no|none|without)\b", evidence))
        )
    return bool(
        re.search(r"(?i)\b(?:final\s+)?(?:answer|response)\s*(?:is|:|=)", evidence)
        or re.search(
            r"(?i)^\s*(?:there\s+(?:are|is)\s+no\b|the\s+letters?\s+visible\b"
            r"|(?:i(?:'m|\s+am)?\s+)?unable\s+to\s+(?:identify|see|find)\b"
            r"|(?:(?:therefore|so),?\s+)?the\s+answer\s+(?:should|would)\s+be\b"
            r"|(?:therefore,?\s+|so,?\s+)?the\s+letters?\b"
            r"|i(?:'ll|\s+will)\s+go\s+with\b)",
            evidence,
        )
        or re.search(
            r"(?i)^\s*(?:there\s+)?(?:do(?:es)?n't|do(?:es)?\s+not)\s+"
            r"(?:appear\s+to\s+be\s+)?any\s+(?:recognizable\s+)?letters?\b",
            evidence,
        )
        or re.search(
            r"(?i)\b(?:letter|letters|word)\b.{0,120}\b"
            r"(?:is|are|spell|spells|seen|visible|appear|appears|can\s+be\s+seen)\b",
            evidence,
        )
        or response.rstrip().casefold().endswith(evidence.strip().casefold())
    )


def evidence_supports(
    response: str, evidence: str, answer: str, answer_type: str
) -> bool:
    if not evidence or evidence not in response:
        return False
    return _evidence_states_answer(
        evidence, answer, answer_type
    ) and _evidence_is_commitment(response, evidence, answer, answer_type)


def length_limited_evidence_supports(
    response: str,
    evidence: str,
    answer: str,
) -> bool:
    escaped = re.escape(answer.strip())
    if escaped and re.search(
        rf"(?is)<answer>\s*{escaped}\s*</answer>"
        rf"|<\|begin_of_box\|>\s*{escaped}\s*<\|end_of_box\|>"
        rf"|\\boxed\{{\s*{escaped}\s*\}}",
        response,
    ):
        return True
    position = response.rfind(evidence)
    if position < 0:
        return False
    trailing = response[position + len(evidence) :]
    if re.sub(r"[\s`*_#.,;:]+", "", trailing):
        return False
    return bool(
        re.search(
            r"(?i)\b(?:final\s+)?(?:answer|response)\s*(?:is|:|=|-)"
            r"|\b(?:choose|select(?:ed)?|conclude|go\s+with)\b"
            r"|\bthe\s+correct\s+(?:option|choice|answer|figure)\b",
            evidence,
        )
    )


def classify_extractor_output(
    candidate: dict[str, Any], extractor_output: str, error: str | None = None
) -> dict[str, Any]:
    parsed = parse_extractor_output(extractor_output)
    extractor_verdict = answer = evidence = ""
    proposed_answer = ""
    status = "request_error" if error else "invalid_extractor_output"
    if parsed is not None:
        extractor_verdict, answer, evidence = parsed
        if extractor_verdict not in VERDICTS:
            status = "invalid_verdict"
        elif extractor_verdict == "UNRESOLVED":
            status = (
                "unresolved"
                if not answer and not evidence
                else "invalid_unresolved_payload"
            )
        else:
            if not answer or not evidence:
                status = "invalid_committed_payload"
            else:
                proposed_answer = answer
                evidence_answer_type = (
                    candidate["answer_type"]
                    if valid_answer(answer, candidate["answer_type"])
                    else "text"
                )
                supported = evidence_supports(
                    candidate["response"], evidence, answer, evidence_answer_type
                )
                if not supported:
                    status = "unsupported_by_evidence"
                elif (
                    str(candidate.get("response_finish_reason") or "").lower()
                    == "length"
                    and not length_limited_evidence_supports(
                        candidate["response"], evidence, answer
                    )
                ):
                    status = "unresolved_truncated_response"
                else:
                    canonical = canonicalize_extracted_answer(
                        answer,
                        answer_type=str(candidate["answer_type"]),
                        task=str(candidate.get("task") or ""),
                    )
                    answer = canonical.value
                    status = (
                        "committed" if canonical.valid else "invalid_format_committed"
                    )

        if status in {"unsupported_by_evidence", "unresolved_truncated_response"}:
            answer = ""
        elif status == "invalid_format_committed":
            answer = INVALID_FORMAT_TOKEN

    result = {
        "extractor_verdict": extractor_verdict,
        "answer": answer,
        "evidence": evidence,
        "status": status,
    }
    if proposed_answer:
        result["proposed_answer"] = proposed_answer
    if status == "committed":
        if "gold_answer" in candidate:
            verdict = commitment_verdict(
                answer, candidate["gold_answer"], candidate["answer_type"]
            )
            status = verdict.casefold()
            result["status"] = status
            result["verdict"] = verdict
        result["submission_comparison"] = (
            "confirmed"
            if answers_equal(
                candidate["current_submission_answer"],
                answer,
                candidate["answer_type"],
            )
            else "recovered"
        )
    return result


def terminal_source_classification(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Classify only literal terminal commitments after extractor retries fail."""
    response = str(candidate.get("response") or "").strip()
    if not response:
        return None
    answer_type = str(candidate.get("answer_type") or "text")
    task = str(candidate.get("task") or "")

    closed_matches = []
    for pattern in (
        r"(?is)<answer>\s*(.*?)\s*</answer>",
        r"(?is)<\|begin_of_box\|>\s*(.*?)\s*<\|end_of_box\|>",
        r"(?is)\\boxed\{\s*([^{}]+?)\s*\}",
    ):
        closed_matches.extend(
            match
            for match in re.finditer(pattern, response)
            if not re.sub(r"[\s`*_#.,;:!?]+", "", response[match.end() :])
        )
    if closed_matches:
        match = max(closed_matches, key=lambda item: item.start())
        proposed = match.group(1).strip()
        canonical = canonicalize_extracted_answer(
            proposed,
            answer_type=answer_type,
            task=task,
        )
        return {
            "extractor_verdict": "COMMITTED",
            "answer": canonical.value,
            "proposed_answer": proposed,
            "evidence": match.group(0),
            "status": (
                "committed" if canonical.valid else "invalid_format_committed"
            ),
        }

    no_answer_patterns = [
        r"(?is)((?:none\s+of\s+the\s+(?:options|choices)[^\n]*"
        r"|there\s+is\s+no\s+correct\s+answer[^\n]*))$",
    ]
    if answer_type == "text" and task == "letter_disambiguation":
        no_answer_patterns.append(
            r"(?is)(there\s+are\s+no\s+(?:recognizable\s+)?letters?[^\n]*[.!]?)$"
        )
    for pattern in no_answer_patterns:
        match = re.search(pattern, response)
        if match:
            return {
                "extractor_verdict": "COMMITTED",
                "answer": INVALID_FORMAT_TOKEN,
                "evidence": match.group(1).strip(),
                "status": "invalid_format_committed",
            }

    if answer_type == "text" and task == "letter_disambiguation":
        direct_symbol_string = not bool(re.search(r"\s", response))
        direct_list = explicit_letter_report = False
        if len(response) <= 800:
            direct_list = bool(
                re.fullmatch(r"\[\s*['\"][A-Za-z]['\"].*\]", response, re.S)
            )
            explicit_letter_report = bool(
                re.fullmatch(
                    r"(?is)(?:(?:on|in)\b[^.\n]{0,100}\bletters?\b[^.\n]{0,100}"
                    r"\b(?:visible|shown|present|can\s+be\s+seen)\b[.!]?\s*)+",
                    response,
                )
            )
        if direct_list or direct_symbol_string or explicit_letter_report:
            canonical = canonicalize_extracted_answer(
                response,
                answer_type=answer_type,
                task=task,
            )
            if not canonical.valid:
                return {
                    "extractor_verdict": "COMMITTED",
                    "answer": INVALID_FORMAT_TOKEN,
                    "proposed_answer": response,
                    "evidence": response,
                    "status": "invalid_format_committed",
                }
    return None


def finalize_persistent_extractor_failure(
    candidate: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any] | None:
    if row.get("status") not in BLOCKING_EXTRACTOR_STATUSES:
        return None
    if not row.get("extractor_attempts"):
        return None
    classification = terminal_source_classification(candidate)
    if classification is None:
        return None
    return {
        **row,
        "extractor_attempts": [
            *list(row.get("extractor_attempts") or []),
            _checkpoint_attempt(row),
        ],
        **classification,
        "terminal_fallback_method": TERMINAL_FALLBACK_METHOD,
        "terminal_fallback_from_status": str(row.get("status") or ""),
    }


def load_candidates(
    project_root: Path,
    canonical_root: Path,
    policy: str,
    gold_answers: dict[str, str] | None = None,
    excluded_variants: set[str] | None = None,
) -> list[dict[str, Any]]:
    questions = {
        track: {
            str(row["question_id"]): row
            for row in (
                json.loads(line)
                for line in (project_root / "tasks" / track / "questions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            )
        }
        for track in TRACKS
    }
    excluded_variants = excluded_variants or set()
    index_path = canonical_root / "index.json"
    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.is_file()
        else {}
    )
    sources: list[tuple[str, str, Path]] = []
    if isinstance(index.get("variants"), list):
        for variant in index["variants"]:
            model_slug = str(variant.get("variant_id") or "")
            if not model_slug or model_slug in excluded_variants:
                continue
            for track in TRACKS:
                track_record = variant.get("tracks", {}).get(track, {})
                relative_dir = str(track_record.get("relative_dir") or "")
                if relative_dir:
                    sources.append((model_slug, track, canonical_root / relative_dir))
    else:
        for model_dir in sorted(path for path in canonical_root.iterdir() if path.is_dir()):
            if model_dir.name in excluded_variants:
                continue
            sources.extend((model_dir.name, track, model_dir) for track in TRACKS)

    candidates = []
    for model_slug, track, model_dir in sources:
        diagnostics = model_dir / f"{track}.diagnostics.jsonl"
        submission = model_dir / f"{track}_submission.jsonl"
        if not diagnostics.is_file() or not submission.is_file():
            continue
        submissions = {
            str(row["question_id"]): row
            for row in (
                json.loads(line)
                for line in submission.read_text(encoding="utf-8").splitlines()
                if line
            )
        }
        for record in (
            json.loads(line)
            for line in diagnostics.read_text(encoding="utf-8").splitlines()
            if line
        ):
            question_id = str(record["question_id"])
            question = questions[track][question_id]
            answer_type = str(
                question.get("answer_type") or record.get("answer_type") or "text"
            )
            category = candidate_category(track, answer_type, record)
            if policy == "all":
                selected = True
            elif policy == "all_nonexact":
                selected = category is not None
            elif policy == "unresolved":
                selected = category == "unresolved_raw"
            else:
                selected = category in {
                    "heuristic_local_parse",
                    "prior_model_extractor",
                    "unresolved_raw",
                }
            if not selected:
                continue
            response = str(record.get("output") or "")
            candidate = {
                "model_slug": model_slug,
                "source_relative_dir": str(model_dir.relative_to(canonical_root)),
                "track": track,
                "question_id": question_id,
                "answer_type": answer_type,
                "task": str(question.get("task") or ""),
                "category": category or "contract_exact",
                "question": str(question.get("question") or ""),
                "response": response,
                "response_finish_reason": record.get("finish_reason"),
                "response_completion_tokens": record.get("completion_tokens"),
                "response_sha256": hashlib.sha256(
                    response.encode("utf-8")
                ).hexdigest(),
                "current_submission_answer": str(
                    submissions[question_id].get("answer") or ""
                ),
            }
            if gold_answers is not None:
                candidate["gold_answer"] = gold_answers[question_id]
            candidates.append(candidate)
    return candidates


def candidate_key(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        candidate["model_slug"],
        candidate["track"],
        candidate["question_id"],
    )


def _checkpoint_attempt(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "status",
        "extractor_verdict",
        "answer",
        "proposed_answer",
        "evidence",
        "extractor_output",
        "finish_reason",
        "completion_tokens",
        "error",
    )
    return {field: row[field] for field in fields if field in row}


def _atomic_write_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def finalize_audit_checkpoint(
    path: Path,
    candidates_by_key: dict[tuple[str, str, str], dict[str, Any]],
    extractor_contract: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    finalized_counts: Counter[str] = Counter()
    unresolved_blockers: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = candidate_key(row)
        if key in seen:
            raise RuntimeError(f"Audit checkpoint repeats candidate {key}.")
        seen.add(key)
        candidate = candidates_by_key.get(key)
        if candidate is None:
            raise RuntimeError(f"Audit checkpoint has unexpected candidate {key}.")
        if row.get("method") != METHOD:
            raise RuntimeError(f"Audit checkpoint method changed for {key}.")
        if row.get("extractor_contract_sha256") != extractor_contract:
            raise RuntimeError(f"Audit checkpoint contract changed for {key}.")
        if row.get("ground_truth_loaded") is not False:
            raise RuntimeError(f"Audit checkpoint loaded ground truth for {key}.")
        if row.get("ground_truth_supplied_to_extractor") is not False:
            raise RuntimeError(f"Audit checkpoint supplied ground truth for {key}.")
        if row.get("response_sha256") != candidate["response_sha256"]:
            raise RuntimeError(f"Candidate response changed for {key}.")
        if row.get("status") in BLOCKING_EXTRACTOR_STATUSES:
            finalized = finalize_persistent_extractor_failure(candidate, row)
            if finalized is None:
                unresolved_blockers.append(key)
            else:
                row = finalized
                finalized_counts[str(row["status"])] += 1
        rows.append(row)

    missing = set(candidates_by_key) - seen
    if missing:
        raise RuntimeError(f"Audit checkpoint is missing {len(missing)} candidates.")
    if unresolved_blockers:
        preview = ", ".join(map(str, unresolved_blockers[:5]))
        raise RuntimeError(
            f"Cannot terminalize {len(unresolved_blockers)} blocking rows: {preview}."
        )
    if finalized_counts:
        _atomic_write_audit(path, rows)
    return {
        "rows": len(rows),
        "terminalized": sum(finalized_counts.values()),
        "terminalized_statuses": dict(finalized_counts),
        "blocking_remaining": 0,
        "output": str(path),
    }


def load_audit_checkpoint(
    path: Path,
    candidates_by_key: dict[tuple[str, str, str], dict[str, Any]],
    extractor_contract: str,
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], list[dict[str, Any]]],
]:
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    retry_history: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    retained_rows: list[dict[str, Any]] = []
    if not path.is_file():
        return existing, retry_history

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = candidate_key(row)
        if key in existing or key in retry_history:
            raise RuntimeError(f"Audit checkpoint repeats candidate {key}.")
        if key not in candidates_by_key:
            raise RuntimeError(f"Audit checkpoint has unexpected candidate {key}.")
        candidate = candidates_by_key[key]
        if row.get("method") != METHOD:
            raise RuntimeError(
                "Audit checkpoint uses an incompatible extraction method. "
                "Use a new output path for the gold-blind audit."
            )
        if row.get("extractor_contract_sha256") != extractor_contract:
            raise RuntimeError(
                "Audit checkpoint uses an incompatible extractor contract. "
                "Use a new output path."
            )
        if row.get("ground_truth_loaded") is not False:
            raise RuntimeError(
                "Audit checkpoint is not from a ground-truth-isolated process."
            )
        if row.get("response_sha256") != candidate["response_sha256"]:
            raise RuntimeError(f"Candidate response changed for {key}.")

        if row.get("status") in BLOCKING_EXTRACTOR_STATUSES:
            retry_history[key] = [
                *list(row.get("extractor_attempts") or []),
                _checkpoint_attempt(row),
            ]
            continue
        existing[key] = row
        retained_rows.append(row)

    if retry_history:
        _atomic_write_audit(path, retained_rows)
    return existing, retry_history


def extractor_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": str(candidate["question"]),
        "answer_type": str(candidate["answer_type"]),
        "candidate_response": str(candidate["response"]),
    }
    if candidate.get("task"):
        payload["task"] = str(candidate["task"])
    if "response_finish_reason" in candidate or "response_completion_tokens" in candidate:
        payload["response_metadata"] = {
            "finish_reason": candidate.get("response_finish_reason"),
            "completion_tokens": candidate.get("response_completion_tokens"),
        }
    return payload


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


async def wait_for_extractor_clients(
    clients: list[AsyncOpenAI],
    endpoints: list[str],
    model: str,
    timeout: float,
    *,
    poll_interval: float = 5,
) -> None:
    deadline = time.monotonic() + timeout
    pending_clients = set(range(len(clients)))
    last_errors: dict[int, str] = {}
    while pending_clients and time.monotonic() < deadline:
        for index in tuple(pending_clients):
            try:
                served = {item.id for item in (await clients[index].models.list()).data}
                if model not in served:
                    last_errors[index] = f"serves {sorted(served)}, not {model}"
                    continue
                pending_clients.remove(index)
                last_errors.pop(index, None)
            except Exception as exc:  # noqa: BLE001 - model server may be starting
                last_errors[index] = f"{type(exc).__name__}: {exc}"[:500]
        if pending_clients:
            await asyncio.sleep(poll_interval)
    if pending_clients:
        details = "; ".join(
            f"{endpoints[index]}: {last_errors.get(index, 'unavailable')}"
            for index in sorted(pending_clients)
        )
        raise RuntimeError(
            f"Extractor endpoints did not become ready within {timeout}s ({details})."
        )


async def run(args: argparse.Namespace) -> None:
    candidates = load_candidates(
        args.project_root,
        args.canonical_root,
        args.policy,
        excluded_variants=set(args.exclude_variants),
    )
    extractor_contract = extractor_contract_sha256(
        args.model, args.max_tokens, args.revision
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates_by_key = {candidate_key(item): item for item in candidates}
    existing, retry_history = load_audit_checkpoint(
        args.output,
        candidates_by_key,
        extractor_contract,
    )
    pending = [item for item in candidates if candidate_key(item) not in existing]
    if not pending:
        print(
            json.dumps(
                {
                    "candidates": len(candidates),
                    "resumed": len(candidates),
                    "new": 0,
                    "ground_truth_loaded": False,
                    "ground_truth_supplied_to_extractor": False,
                    "extractor_contract_sha256": extractor_contract,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return
    clients = [
        AsyncOpenAI(base_url=endpoint.rstrip("/"), api_key=args.api_key, timeout=args.timeout)
        for endpoint in args.endpoints
    ]
    print(
        f"Prepared {len(candidates)} evidence candidates; waiting for extractor endpoints.",
        flush=True,
    )
    await wait_for_extractor_clients(
        clients,
        args.endpoints,
        args.model,
        args.endpoint_start_timeout,
    )
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)
    lock = asyncio.Lock()
    completed = 0

    async def extract(client: AsyncOpenAI, candidate: dict[str, Any]) -> dict[str, Any]:
        payload = extractor_payload(candidate)
        error = None
        extractor_output = ""
        finish_reason = None
        completion_tokens = None
        classification: dict[str, Any] = {}
        for attempt in range(args.retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    temperature=0,
                    top_p=1,
                    seed=0,
                    max_tokens=args.max_tokens,
                    response_format=EXTRACTOR_RESPONSE_FORMAT,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = response.choices[0]
                extractor_output = str(choice.message.content or "").strip()
                finish_reason = getattr(choice, "finish_reason", None)
                usage = getattr(response, "usage", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                error = None
            except Exception as exc:  # noqa: BLE001 - checkpoint request failures
                error = f"{type(exc).__name__}: {exc}"[:500]
            classification = classify_extractor_output(
                candidate, extractor_output, error
            )
            retryable = classification["status"] in {
                "request_error",
                "invalid_extractor_output",
                "invalid_verdict",
                "invalid_unresolved_payload",
                "invalid_committed_payload",
                "unsupported_by_evidence",
            }
            if not retryable or attempt == args.retries:
                break
            await asyncio.sleep(min(2**attempt, 8))
        result = {
            key: candidate[key]
            for key in (
                "model_slug",
                "source_relative_dir",
                "track",
                "question_id",
                "answer_type",
                "task",
                "category",
                "response_finish_reason",
                "response_sha256",
            )
        } | {
            "method": METHOD,
            "extractor_contract_sha256": extractor_contract,
            "ground_truth_loaded": False,
            "ground_truth_supplied_to_extractor": False,
            "extractor_model": args.model,
            "extractor_revision": args.revision,
            **classification,
            "extractor_output": extractor_output,
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens,
            **({"error": error} if error else {}),
        }
        history = retry_history.get(candidate_key(candidate))
        if history:
            result["extractor_attempts"] = history
            finalized = finalize_persistent_extractor_failure(candidate, result)
            if finalized is not None:
                result = finalized
        return result

    async def worker(index: int) -> None:
        nonlocal completed
        client = clients[index % len(clients)]
        while True:
            try:
                candidate = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                result = await extract(client, candidate)
                async with lock:
                    with args.output.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    completed += 1
                    if completed % args.report_every == 0:
                        print(f"extracted {completed}/{len(pending)} new", flush=True)
            finally:
                queue.task_done()

    await asyncio.gather(
        *(worker(index) for index in range(min(args.concurrency, len(pending))))
    )
    for client in clients:
        await client.close()
    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "resumed": len(candidates) - len(pending),
                "new": len(pending),
                "ground_truth_loaded": False,
                "ground_truth_supplied_to_extractor": False,
                "extractor_contract_sha256": extractor_contract,
                "output": str(args.output),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--canonical-root", type=Path, default=project_root / "evaluation/results/final"
    )
    parser.add_argument("--endpoint", action="append", dest="endpoints", default=[])
    parser.add_argument("--model", default=DEFAULT_EXTRACTOR_MODEL)
    parser.add_argument("--revision", default=DEFAULT_EXTRACTOR_REVISION)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--policy",
        choices=("all", "unresolved", "high_risk", "all_nonexact"),
        default="all",
    )
    parser.add_argument(
        "--exclude-variant",
        action="append",
        dest="exclude_variants",
        default=list(DEFAULT_EXCLUDED_VARIANTS),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--finalize-checkpoint",
        action="store_true",
        help="Atomically terminalize persistent blocking rows without model access.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--endpoint-start-timeout", type=float, default=1800)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=25)
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.max_tokens < 1
        or args.report_every < 1
        or args.endpoint_start_timeout <= 0
    ):
        parser.error(
            "concurrency, max-tokens, report-every, and endpoint-start-timeout "
            "must be positive"
        )
    if not args.finalize_checkpoint and not args.endpoints:
        parser.error("at least one --endpoint is required for extraction")
    return args


def main() -> None:
    try:
        args = parse_args()
        if args.finalize_checkpoint:
            candidates = load_candidates(
                args.project_root,
                args.canonical_root,
                args.policy,
                excluded_variants=set(args.exclude_variants),
            )
            result = finalize_audit_checkpoint(
                args.output,
                {candidate_key(item): item for item in candidates},
                extractor_contract_sha256(
                    args.model,
                    args.max_tokens,
                    args.revision,
                ),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            asyncio.run(run(args))
    except GroundTruthError as exc:
        raise SystemExit(f"Gold-aware audit refused: {exc}") from exc


if __name__ == "__main__":
    main()