"""Shared canonical answer contract for full visual benchmark submissions."""

from __future__ import annotations

import re
from dataclasses import dataclass


UNRESOLVED_TOKEN = "UNRESOLVED"
INVALID_FORMAT_TOKEN = "__INVALID_FORMAT__"
PRODUCTION_EXTRACTION_METHOD = "qwen3-8b-gold-blind-evidence-extractor-v4"
LETTER_DISAMBIGUATION_MAX_LENGTH = 9

_DYSM_ID_RE = re.compile(
    r"^t1_(?:2d|3d)_(?P<task>.+)_(?:easy|medium|hard)_\d+$"
)
_MINDS_EYE_ID_RE = re.compile(r"^t2_(?P<task>.+)_\d+$")
_LETTER_SEQUENCE_RE = re.compile(r"^[A-Za-z\s,.;:!?_/'\"\-\[\]()]+$")


@dataclass(frozen=True)
class CanonicalAnswer:
    value: str
    valid: bool
    transformed: bool
    reason: str


def task_from_question_id(question_id: str) -> str:
    identifier = str(question_id or "").strip().lower()
    match = _DYSM_ID_RE.fullmatch(identifier) or _MINDS_EYE_ID_RE.fullmatch(
        identifier
    )
    return match.group("task") if match else ""


def canonicalize_extracted_answer(
    answer,
    *,
    answer_type: str,
    task: str,
) -> CanonicalAnswer:
    if isinstance(answer, (dict, list)) or answer is None:
        return CanonicalAnswer(INVALID_FORMAT_TOKEN, False, True, "non_scalar")
    original = str(answer).strip()
    if not original:
        return CanonicalAnswer(INVALID_FORMAT_TOKEN, False, True, "empty")
    if original == UNRESOLVED_TOKEN:
        return CanonicalAnswer(UNRESOLVED_TOKEN, False, False, "unresolved")
    if original == INVALID_FORMAT_TOKEN:
        return CanonicalAnswer(INVALID_FORMAT_TOKEN, False, False, "invalid_format")

    normalized_type = str(answer_type or "text").strip()
    normalized_task = str(task or "").strip().lower()
    value = original

    if normalized_type == "integer":
        valid = bool(re.fullmatch(r"-?\d+", value))
    elif normalized_type == "mcq_index_1_4":
        valid = bool(re.fullmatch(r"[1-4]", value))
    elif normalized_type == "mcq_letter":
        value = value.upper()
        valid = bool(re.fullmatch(r"[A-F]", value))
    elif normalized_type == "text" and normalized_task == "form_constancy":
        aliases = {
            "yes": "Yes",
            "true": "Yes",
            "no": "No",
            "false": "No",
        }
        value = aliases.get(value.casefold(), value)
        valid = value in {"Yes", "No"}
    elif normalized_type == "text" and normalized_task == "letter_disambiguation":
        if not _LETTER_SEQUENCE_RE.fullmatch(value):
            value = ""
        else:
            tokens = re.findall(r"[A-Za-z]+", value)
            tokens = [token for token in tokens if token.casefold() != "and"]
            if len(tokens) == 1 or all(len(token) == 1 for token in tokens):
                value = "".join(tokens).upper()
            else:
                value = ""
        valid = bool(
            re.fullmatch(rf"[A-Z]{{1,{LETTER_DISAMBIGUATION_MAX_LENGTH}}}", value)
        )
    else:
        valid = bool(value)

    if not valid:
        return CanonicalAnswer(INVALID_FORMAT_TOKEN, False, True, "out_of_domain")
    return CanonicalAnswer(value, True, value != original, "canonical")


def is_canonical_visual_answer(answer, *, answer_type: str, task: str) -> bool:
    if isinstance(answer, (dict, list)) or answer is None:
        return False
    value = str(answer).strip()
    normalized_task = str(task or "").strip().lower()
    if not value or value in {UNRESOLVED_TOKEN, INVALID_FORMAT_TOKEN}:
        return False
    if answer_type == "integer":
        return bool(re.fullmatch(r"-?\d+", value))
    if answer_type == "mcq_index_1_4":
        return bool(re.fullmatch(r"[1-4]", value))
    if answer_type == "mcq_letter":
        return bool(re.fullmatch(r"[A-Fa-f]", value))
    if answer_type == "text" and normalized_task == "form_constancy":
        return value.casefold() in {"yes", "no"}
    if answer_type == "text" and normalized_task == "letter_disambiguation":
        return bool(
            re.fullmatch(
                rf"[A-Za-z]{{1,{LETTER_DISAMBIGUATION_MAX_LENGTH}}}", value
            )
        )
    return True


def canonical_answers_equal(left, right, *, answer_type: str, task: str) -> bool:
    left_result = canonicalize_extracted_answer(
        left,
        answer_type=answer_type,
        task=task,
    )
    right_result = canonicalize_extracted_answer(
        right,
        answer_type=answer_type,
        task=task,
    )
    if not left_result.valid or not right_result.valid:
        return False
    if answer_type == "integer":
        return int(left_result.value) == int(right_result.value)
    if answer_type in {"mcq_letter", "text"}:
        return left_result.value.casefold() == right_result.value.casefold()
    return left_result.value == right_result.value
