"""Flexible parsers shared by custom Track-3 dataset loaders."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .common import intermediate_to_row, normalize_options, resolve_mcq_answer, to_b64


def _split_paren_options(question: str):
    """Split ``stem (A) x (B) y`` and repair SpatialBench source quirks."""
    marks = list(re.finditer(r"\(([A-H])\)\s*", question, flags=re.I))
    if len(marks) < 2:
        return question.strip(), None
    instruction = r"Answer with the option'?s letter[^.\n]*\.?"
    stem = re.sub(
        r"\s*" + instruction + r"\s*$",
        "",
        question[: marks[0].start()].strip(),
        flags=re.I,
    ).strip()
    options: dict[str, str] = {}
    for index, mark in enumerate(marks):
        letter = mark.group(1).upper()
        if letter in options:
            letter = "ABCDEFGH"[len(options)]
        start = mark.end()
        end = marks[index + 1].start() if index + 1 < len(marks) else len(question)
        text = re.sub(
            r"\s*" + instruction + r"\s*",
            " ",
            question[start:end],
            flags=re.I,
        ).strip().rstrip(".")
        options[letter] = text
    return stem, options


def mmvp_rows(snapshot: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(snapshot / "Questions.csv")
    rows = []
    for _, record in frame.iterrows():
        sample_index = int(record["Index"])
        _unused_stem, options = _split_paren_options(str(record["Options"]))
        row, error = intermediate_to_row(
            sample_index,
            {
                "image": snapshot / "MMVP Images" / f"{sample_index}.jpg",
                "question": record["Question"],
                "options": options,
                "answer": record["Correct Answer"],
            },
        )
        if row is None:
            raise ValueError(f"MMVP row {sample_index} is unusable: {error}")
        rows.append(row)
    return rows


def _split_line_options(question: str):
    question = re.sub(
        r"\s*Please answer directly.*$", "", question.strip(), flags=re.I | re.S
    ).strip()
    marks = list(re.finditer(r"(?m)^[ \t]*([A-H])[\.:]\s*", question))
    if len(marks) < 2:
        return question, None
    stem = question[: marks[0].start()].strip()
    options = {}
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(question)
        options[mark.group(1).upper()] = question[mark.end() : end].strip()
    return stem, options


def realworldqa_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for index, record in frame.reset_index(drop=True).iterrows():
        question, options = _split_line_options(str(record["question"]))
        row, error = intermediate_to_row(
            index,
            {
                "image": record["image"],
                "question": question,
                "options": options,
                "answer": record["answer"],
            },
        )
        if row is None:
            raise ValueError(f"RealWorldQA row {index} is unusable: {error}")
        rows.append(row)
    return rows


def _serialized_list(value: Any, field: str) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid {field} list: {value!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field} is not a list: {value!r}")
    return parsed


def mindcube_rows(frame: pd.DataFrame, asset_root: Path) -> list[dict[str, Any]]:
    rows = []
    for row_index, record in frame.iterrows():
        candidates = [str(value).strip() for value in _serialized_list(record["candidates"], "candidates")]
        options = {}
        for position, candidate in enumerate(candidates):
            match = re.fullmatch(r"([A-H])\.\s*(.+)", candidate, flags=re.I | re.S)
            if not match:
                raise ValueError(f"MindCube row {row_index} has malformed candidate: {candidate!r}")
            options[match.group(1).upper()] = match.group(2).strip()
        answer, error = resolve_mcq_answer(record["answer"], options, 0)
        if error:
            raise ValueError(f"MindCube row {row_index} has unresolved answer: {error}")

        question = re.sub(r"<image>\s*", "", str(record["question"])).strip()
        suffix = " ".join(candidates)
        if question.endswith(suffix):
            question = question[: -len(suffix)].rstrip()
        else:
            marker = question.rfind(f" {candidates[0]}")
            if marker >= 0:
                question = question[:marker].rstrip()

        image_paths = [
            asset_root / str(value).lstrip("/\\")
            for value in _serialized_list(record["image_path"], "image_path")
        ]
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"MindCube row {row_index} is missing images: {missing[:2]}"
            )
        rows.append(
            {
                "index": record["index"],
                "image": json.dumps([str(path.resolve()) for path in image_paths]),
                "question": question,
                **options,
                "answer": answer,
                "answer_type": "mcq",
                "category": record.get("category"),
            }
        )
    return rows


def _iter_records(value: Any, source_key: str | None = None):
    if isinstance(value, list):
        for item in value:
            yield from _iter_records(item, source_key)
    elif isinstance(value, dict):
        if any(key in value for key in ("question", "prompt", "query")):
            record = dict(value)
            if source_key is not None:
                record.setdefault("_source_key", source_key)
            yield record
        else:
            for key, item in value.items():
                yield from _iter_records(item, str(key))


def _first(record: dict[str, Any], names: tuple[str, ...]):
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def _find_one_image(root: Path, category: str, value: Any) -> Path:
    candidates = []
    if value is not None:
        candidates.extend((root / str(value), root / category / str(value)))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if not candidate.suffix:
            for suffix in (".jpg", ".jpeg", ".png", ".webp"):
                with_suffix = candidate.with_suffix(suffix)
                if with_suffix.is_file():
                    return with_suffix
    raise FileNotFoundError(f"Cannot resolve {category} image: {value}")


def _find_image(root: Path, category: str, record: dict[str, Any]) -> Path | list[Path]:
    value = _first(
        record,
        ("image", "image_path", "rgb", "rgb_path", "img", "filename", "file_name"),
    )
    if isinstance(value, (list, tuple)):
        return [_find_one_image(root, category, item) for item in value]
    if value is not None:
        return _find_one_image(root, category, value)
    source_key = record.get("_source_key")
    if source_key:
        return _find_one_image(root, category, source_key)
    raise FileNotFoundError(f"Cannot resolve {category} image for an unnamed record")


def spatialbench_rows(snapshot: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    index = 0
    for metadata in sorted(snapshot.glob("*.json")):
        category = metadata.stem
        import json

        data = json.loads(metadata.read_text(encoding="utf-8"))
        for record in _iter_records(data):
            question = str(_first(record, ("question", "prompt", "query")) or "").strip()
            options = _first(record, ("options", "choices", "answer_choices"))
            if isinstance(options, str):
                try:
                    options = json.loads(options)
                except json.JSONDecodeError:
                    options = None
            if not options:
                question, options = _split_paren_options(question)
            answer = _first(record, ("answer", "gt", "label", "correct_answer"))
            try:
                image = _find_image(snapshot, category, record)
                row, error = intermediate_to_row(
                    index,
                    {
                        "image": image,
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "category": category,
                    },
                    index_base=0,
                )
            except (FileNotFoundError, TypeError, ValueError):
                row, error = None, "image_error"
            if row is None:
                skipped[error or "unknown"] = skipped.get(error or "unknown", 0) + 1
                continue
            rows.append(row)
            index += 1
    return rows, skipped


def omnispatial_rows(dataset_root: Path) -> list[dict[str, Any]]:
    import json

    data_path = next(dataset_root.rglob("data.json"), None)
    if data_path is None:
        raise FileNotFoundError(f"OmniSpatial data.json not found under {dataset_root}")
    records = json.loads(data_path.read_text(encoding="utf-8"))
    rows = []
    for index, record in enumerate(records):
        raw_id = str(record["id"])
        task_type = str(record["task_type"])
        image = data_path.parent / task_type / f"{raw_id.split('_')[0]}.png"
        if not image.is_file():
            raise FileNotFoundError(f"OmniSpatial image not found: {image}")
        row, error = intermediate_to_row(
            index,
            {
                "image": image,
                "question": record["question"],
                "options": record.get("options"),
                "answer": record.get("answer"),
                "category": task_type,
            },
            index_base=0,
        )
        if row is None:
            raise ValueError(f"OmniSpatial row {raw_id} is unusable: {error}")
        row["qid"] = raw_id
        row["sub_task_type"] = record.get("sub_task_type")
        rows.append(row)
    return rows


def sat_real_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for index, record in frame.iterrows():
        image_values = record.get("image_bytes")
        if hasattr(image_values, "tolist"):
            image_values = image_values.tolist()
        if not isinstance(image_values, (list, tuple)):
            image_values = [image_values]
        images = [to_b64(value) for value in image_values if value is not None]
        answer_values = record.get("answers")
        if hasattr(answer_values, "tolist"):
            answer_values = answer_values.tolist()
        options, _letters = normalize_options(answer_values)
        answer, error = resolve_mcq_answer(record.get("correct_answer"), options, 0)
        if error:
            raise ValueError(f"SAT-Real row {index} has unresolved answer: {error}")
        rows.append(
            {
                "index": index,
                "image": __import__("json").dumps(images),
                "question": str(record.get("question") or "").strip(),
                **options,
                "answer": answer,
                "answer_type": "mcq",
            }
        )
    return rows