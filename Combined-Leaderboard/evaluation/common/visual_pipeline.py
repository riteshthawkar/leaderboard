"""Strict input loading and canonical JSONL export for visual benchmarks."""

from __future__ import annotations

import base64
import binascii
import io
import json
import mimetypes
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen


STANDARD_CONDITION = "standard"
MAX_REMOTE_IMAGE_BYTES = 50 * 1024 * 1024
SUPPORTED_ANSWER_TYPES = {"integer", "mcq_index_1_4", "mcq_letter", "text"}
INTEGER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


class EvaluationPipelineError(ValueError):
    """Raised when an evaluation run cannot produce a safe upload artifact."""


@dataclass(frozen=True)
class VisualTrackConfig:
    task_id: str
    label: str
    source_subsets: tuple[str, ...]
    questions_path: Path
    package_dir: Path

    @property
    def results_dir(self) -> Path:
        return self.package_dir / "results"

    def prompt_path(self, prompt_mode: str) -> Path:
        return self.package_dir / "prompts" / f"{prompt_mode}.txt"

    def default_output_path(self) -> Path:
        return self.results_dir / f"{self.task_id}_submission.jsonl"


def load_questions(path: Path, track: VisualTrackConfig) -> list[dict[str, Any]]:
    """Load and validate the public JSONL question bundle for one track."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise EvaluationPipelineError(f"Question bundle not found: {path}")
    if path.suffix.lower() != ".jsonl":
        raise EvaluationPipelineError("The question bundle must be a .jsonl file.")

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise EvaluationPipelineError(
            f"The question bundle is not valid UTF-8 text at byte {exc.start}."
        ) from exc

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationPipelineError(
                f"Question bundle line {line_number} is invalid JSON ({exc.msg})."
            ) from exc
        if not isinstance(row, dict):
            raise EvaluationPipelineError(
                f"Question bundle line {line_number} must contain a JSON object."
            )

        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise EvaluationPipelineError(
                f"Question bundle line {line_number} is missing question_id."
            )
        if question_id in seen_ids:
            raise EvaluationPipelineError(
                f"Question bundle line {line_number} repeats question_id '{question_id}'."
            )
        seen_ids.add(question_id)

        question = str(row.get("question") or "").strip()
        if not question:
            raise EvaluationPipelineError(
                f"Question '{question_id}' has no question text."
            )
        answer_type = str(row.get("answer_type") or "text").strip().lower()
        if answer_type not in SUPPORTED_ANSWER_TYPES:
            raise EvaluationPipelineError(
                f"Question '{question_id}' uses unsupported answer_type '{answer_type}'."
            )

        source_subset = str(row.get("source_subset") or row.get("subset") or "").strip()
        if source_subset and source_subset not in track.source_subsets:
            raise EvaluationPipelineError(
                f"Question '{question_id}' belongs to subset '{source_subset}', not {track.label}."
            )
        image = str(row.get("image_path") or row.get("image") or "").strip()
        image_url = str(row.get("image_url") or "").strip()
        if not image and not image_url:
            raise EvaluationPipelineError(
                f"Question '{question_id}' has no local image path or image_url."
            )

        questions.append(
            {
                "question_id": question_id,
                "question": question,
                "answer_type": answer_type,
                "source_subset": source_subset,
                "image": image,
                "image_url": image_url,
            }
        )

    if not questions:
        raise EvaluationPipelineError("The question bundle contains no questions.")
    return questions


def load_prompt(track: VisualTrackConfig, prompt_mode: str) -> str:
    if prompt_mode not in {"noncot", "cot"}:
        raise EvaluationPipelineError("prompt_mode must be 'noncot' or 'cot'.")
    path = track.prompt_path(prompt_mode)
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise EvaluationPipelineError(f"Could not read prompt file: {path}") from exc
    if not prompt:
        raise EvaluationPipelineError(f"Prompt file is empty: {path}")
    return prompt


def _local_image_path(item: dict[str, Any], image_root: Path | None) -> Path | None:
    raw_path = str(item.get("image") or "").strip()
    if not raw_path:
        return None
    direct = Path(raw_path).expanduser()
    if direct.is_absolute():
        return direct.resolve() if direct.is_file() else None
    if image_root is None:
        return None

    root = Path(image_root).expanduser().resolve()
    subset = str(item.get("source_subset") or "").strip()
    candidates = [root / raw_path]
    if subset:
        candidates.insert(0, root / subset / raw_path)
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None


def resolve_image_source(
    item: dict[str, Any], image_root: Path | None
) -> tuple[str, Path | str]:
    local_path = _local_image_path(item, image_root)
    if local_path is not None:
        return "local", local_path

    image_url = str(item.get("image_url") or "").strip()
    parsed = urlparse(image_url)
    if parsed.scheme in {"http", "https", "data"}:
        return "remote", image_url
    question_id = item.get("question_id", "unknown")
    raise EvaluationPipelineError(
        f"No readable image was found for question_id '{question_id}'. "
        "Pass --image-root for a local dataset checkout or use a bundle with valid image_url values."
    )


def image_for_openai(item: dict[str, Any], image_root: Path | None) -> str:
    source_type, source = resolve_image_source(item, image_root)
    if source_type == "remote":
        return str(source)

    path = Path(source)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if not media_type.startswith("image/"):
        raise EvaluationPipelineError(f"Unsupported local image type: {path.name}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def image_for_hf(item: dict[str, Any], image_root: Path | None):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    source_type, source = resolve_image_source(item, image_root)
    if source_type == "local":
        with Image.open(source) as image:
            image.load()
            return image.copy()

    url = str(source)
    if url.startswith("data:"):
        try:
            _header, encoded = url.split(",", 1)
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise EvaluationPipelineError("The question contains an invalid data image URL.") from exc
        if len(raw) > MAX_REMOTE_IMAGE_BYTES:
            raise EvaluationPipelineError(
                f"Embedded image exceeds {MAX_REMOTE_IMAGE_BYTES // (1024 * 1024)} MB."
            )
    else:
        request = Request(url, headers={"User-Agent": "MS-VISTA-evaluation/1.0"})
        try:
            with urlopen(request, timeout=60) as response:
                declared_size = int(response.headers.get("Content-Length") or 0)
                if declared_size > MAX_REMOTE_IMAGE_BYTES:
                    raise EvaluationPipelineError(
                        f"Remote image exceeds {MAX_REMOTE_IMAGE_BYTES // (1024 * 1024)} MB."
                    )
                raw = response.read(MAX_REMOTE_IMAGE_BYTES + 1)
        except EvaluationPipelineError:
            raise
        except OSError as exc:
            raise EvaluationPipelineError(f"Could not download image: {url}") from exc
        if len(raw) > MAX_REMOTE_IMAGE_BYTES:
            raise EvaluationPipelineError(
                f"Remote image exceeds {MAX_REMOTE_IMAGE_BYTES // (1024 * 1024)} MB."
            )
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            return image.copy()
    except OSError as exc:
        raise EvaluationPipelineError("The downloaded image could not be decoded.") from exc


def final_answer(raw_output: Any, answer_type: str) -> str:
    """Extract the final response token while preserving free-form text answers."""
    if isinstance(raw_output, (dict, list)) or raw_output is None:
        return ""
    text = str(raw_output).strip()
    if not text:
        return ""

    answer_blocks = re.findall(
        r"<answer>((?:(?!<answer>).)*?)</answer>",
        text,
        flags=re.I | re.S,
    )
    if answer_blocks:
        text = answer_blocks[-1].strip()
    else:
        native_box_blocks = re.findall(
            r"<\|begin_of_box\|>"
            r"((?:(?!<\|begin_of_box\|>).)*?)"
            r"<\|end_of_box\|>",
            text,
            flags=re.I | re.S,
        )
        if native_box_blocks:
            text = native_box_blocks[-1].strip()
    if re.search(r"<think>", text, flags=re.I) and not re.search(
        r"</think>", text, flags=re.I
    ):
        return ""
    elif "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1].strip()

    text = text.strip().strip("`").strip()
    if answer_type == "integer":
        values = re.findall(
            r"(?<![A-Za-z0-9])-?\d(?:[\d,]*\d)?(?![A-Za-z0-9])", text
        )
        if values:
            return values[-1].replace(",", "")
        matches = re.findall(
            rf"\b({'|'.join(INTEGER_WORDS)})\b", text, flags=re.I
        )
        normalized = {INTEGER_WORDS[word.lower()] for word in matches}
        return str(normalized.pop()) if len(normalized) == 1 else ""
    if answer_type == "mcq_index_1_4":
        exact = re.fullmatch(r"[\[(]?([1-4])[\])]?[.,:]?", text, flags=re.I)
        if exact:
            return exact.group(1)
        explicit = re.findall(
            r"(?i)(?:(?:final\s+)?(?:answer|response)|(?:selected\s+)?(?:option|choice)|(?:choose|select(?:ed)?))"
            r"\s*(?:is|:|=|-)?\s*[\[(]?([1-4])[\])]?(?=\W|$)",
            text,
        )
        if explicit:
            return explicit[-1]
        final_line = text.splitlines()[-1].strip() if text.splitlines() else ""
        line_match = re.fullmatch(
            r"[\[(]?([1-4])[\])]?[.,:]?", final_line, flags=re.I
        )
        return line_match.group(1) if line_match else ""
    if answer_type == "mcq_letter":
        exact = re.fullmatch(r"[\[(]?([A-Fa-f])[\])]?[.,:]?", text)
        if exact:
            return exact.group(1).upper()
        explicit = re.findall(
            r"(?i)(?:(?:final\s+)?(?:answer|response)|(?:selected\s+)?(?:option|choice)|(?:choose|select(?:ed)?))"
            r"\s*(?:is|:|=|-)?\s*[\[(]?([A-F])[\])]?(?=\W|$)",
            text,
        )
        if explicit:
            return explicit[-1].upper()
        odd_figure_matches = re.findall(
            r"(?i)\bfigure\s+([A-F])\s+does\s+not\s+adhere\b"
            r"|\b(?:the\s+)?figure\s+that\s+does\s+not\s+adhere"
            r"(?:\s+to\s+(?:this|the)\s+concept)?\s+is\s+([A-F])\b",
            text,
        )
        odd_figure_letters = {
            letter.upper()
            for match in odd_figure_matches
            for letter in match
            if letter
        }
        if len(odd_figure_letters) == 1:
            return odd_figure_letters.pop()
        final_line = text.splitlines()[-1].strip() if text.splitlines() else ""
        line_match = re.fullmatch(r"[\[(]?([A-Fa-f])[\])]?[.,:]?", final_line)
        return line_match.group(1).upper() if line_match else ""

    text = re.sub(
        r"^(?:final\s+)?(?:answer|response)\s*(?:is|:)?\s*",
        "",
        text,
        flags=re.I,
    ).strip()
    return text.strip('"\' ').strip()


def record_answer(record: dict[str, Any], answer_type: str) -> str:
    extracted = record.get("extracted_answer")
    if extracted is not None:
        normalized = final_answer(extracted, answer_type)
        if normalized:
            return normalized
    return final_answer(record.get("output"), answer_type)


def stated_integer_values(raw_output: Any) -> set[int]:
    if isinstance(raw_output, (dict, list)) or raw_output is None:
        return set()
    text = str(raw_output)
    values = {
        int(value.replace(",", ""))
        for value in re.findall(
            r"(?<![A-Za-z0-9])-?\d(?:[\d,]*\d)?(?![A-Za-z0-9])", text
        )
    }
    words = re.findall(rf"\b({'|'.join(INTEGER_WORDS)})\b", text, flags=re.I)
    values.update(INTEGER_WORDS[word.lower()] for word in words)
    return values


def _atomic_write(path: Path, content: str) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_diagnostics(path: Path, records: Sequence[dict[str, Any]]) -> None:
    rows = []
    for item in records:
        row = {
            "question_id": str(item.get("question_id") or ""),
            "source_subset": str(item.get("source_subset") or ""),
            "answer_type": str(item.get("answer_type") or "text"),
            "output": item.get("output"),
        }
        if item.get("error"):
            row["error"] = str(item["error"])
        if item.get("finish_reason"):
            row["finish_reason"] = str(item["finish_reason"])
        if item.get("completion_tokens") is not None:
            row["completion_tokens"] = int(item["completion_tokens"])
        if item.get("final_answer_tokens") is not None:
            row["final_answer_tokens"] = int(item["final_answer_tokens"])
        if item.get("answer_extraction_method"):
            row["answer_extraction_method"] = str(item["answer_extraction_method"])
        if item.get("extractor_model"):
            row["extractor_model"] = str(item["extractor_model"])
        if item.get("extractor_output") is not None:
            row["extractor_output"] = item["extractor_output"]
        if item.get("extractor_finish_reason"):
            row["extractor_finish_reason"] = str(item["extractor_finish_reason"])
        if item.get("extractor_completion_tokens") is not None:
            row["extractor_completion_tokens"] = int(
                item["extractor_completion_tokens"]
            )
        if item.get("extractor_error"):
            row["extractor_error"] = str(item["extractor_error"])
        if item.get("extractor_source_diagnostics"):
            row["extractor_source_diagnostics"] = str(
                item["extractor_source_diagnostics"]
            )
        if item.get("extractor_source_output_sha256"):
            row["extractor_source_output_sha256"] = str(
                item["extractor_source_output_sha256"]
            )
        if item.get("extractor_attempts"):
            row["extractor_attempts"] = list(item["extractor_attempts"])
        if item.get("extracted_answer") is not None:
            row["extracted_answer"] = str(item["extracted_answer"])
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    _atomic_write(path, "\n".join(rows) + ("\n" if rows else ""))


def export_submission(
    records: Sequence[dict[str, Any]],
    expected_questions: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    raw_output_fallback: bool = False,
) -> dict[str, Any]:
    """Validate complete coverage and write backend-ready three-field JSONL."""
    expected_ids = [str(item["question_id"]) for item in expected_questions]
    expected_set = set(expected_ids)
    records_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for item in records:
        question_id = str(item.get("question_id") or "").strip()
        if question_id in records_by_id:
            duplicate_ids.append(question_id)
        records_by_id[question_id] = item
    if duplicate_ids:
        preview = ", ".join(sorted(set(duplicate_ids))[:5])
        raise EvaluationPipelineError(
            f"Cannot export submission: duplicate model outputs for {preview}."
        )

    received_set = set(records_by_id)
    missing = [question_id for question_id in expected_ids if question_id not in received_set]
    unknown = sorted(received_set - expected_set)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"{len(missing)} missing question_id(s), including {', '.join(missing[:5])}")
        if unknown:
            details.append(f"{len(unknown)} unknown question_id(s), including {', '.join(unknown[:5])}")
        raise EvaluationPipelineError(
            "Cannot export submission because coverage is incomplete: " + "; ".join(details) + "."
        )

    rows: list[dict[str, str]] = []
    raw_output_fallback_question_ids: list[str] = []
    invalid: list[str] = []
    for expected in expected_questions:
        question_id = str(expected["question_id"])
        record = records_by_id[question_id]
        if record.get("error"):
            invalid.append(f"{question_id} (inference error)")
            continue
        answer = record_answer(record, str(expected.get("answer_type") or "text"))
        if not answer and raw_output_fallback:
            raw_output = record.get("output")
            if not isinstance(raw_output, (dict, list)) and raw_output is not None:
                raw_answer = str(raw_output)
                if raw_answer.strip():
                    answer = raw_answer
                    raw_output_fallback_question_ids.append(question_id)
        if not answer:
            invalid.append(f"{question_id} (empty or unparseable output)")
            continue
        rows.append(
            {
                "question_id": question_id,
                "condition": STANDARD_CONDITION,
                "answer": answer,
            }
        )

    if invalid:
        preview = ", ".join(invalid[:5])
        remainder = len(invalid) - min(len(invalid), 5)
        suffix = f", and {remainder} more" if remainder else ""
        raise EvaluationPipelineError(
            f"Cannot export submission: {len(invalid)} output(s) require attention: {preview}{suffix}. "
            "Review the diagnostics file and rerun failed samples."
        )

    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write(output_path, content)
    return {
        "output_path": str(Path(output_path).expanduser().resolve()),
        "row_count": len(rows),
        "schema": ["question_id", "condition", "answer"],
        "condition": STANDARD_CONDITION,
        "raw_output_fallback_question_ids": raw_output_fallback_question_ids,
    }


def read_diagnostics(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise EvaluationPipelineError(f"Diagnostics file not found: {resolved}")
        try:
            text = resolved.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise EvaluationPipelineError(
                f"Diagnostics file is not valid UTF-8: {resolved}"
            ) from exc
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise EvaluationPipelineError(
                    f"{resolved.name} line {line_number} is invalid JSON ({exc.msg})."
                ) from exc
            if not isinstance(row, dict):
                raise EvaluationPipelineError(
                    f"{resolved.name} line {line_number} must contain an object."
                )
            records.append(row)
    return records
