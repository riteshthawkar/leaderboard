"""Shared normalization helpers for Track-3 dataset loaders."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


OPTS = list("ABCDEFGH")


def read_tsv(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a Track-3 TSV without converting answer text such as ``None`` to NA."""
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def to_b64(image: Any) -> str:
    """Return an image as a plain base64 string suitable for a VLMEvalKit TSV."""
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return to_b64(image["bytes"])
        if image.get("path") is not None:
            return to_b64(image["path"])
        raise ValueError("Image mapping has neither bytes nor path.")
    if isinstance(image, (str, os.PathLike)):
        value = os.fspath(image).strip()
        if value.startswith("data:image/") and "," in value:
            return value.split(",", 1)[1]
        path = Path(value).expanduser()
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if is_file:
            return base64.b64encode(path.read_bytes()).decode("ascii")
        try:
            base64.b64decode(value, validate=True)
            return value
        except (ValueError, base64.binascii.Error):
            raise ValueError(f"Image string is neither a file nor valid base64: {value[:80]}")
    if isinstance(image, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(image)).decode("ascii")
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    raise TypeError(f"Unsupported image value: {type(image).__name__}")


def to_image_cell(image: Any) -> str:
    """Encode one image or a structured list of images for a TSV cell."""
    if isinstance(image, (list, tuple)):
        if not image:
            raise ValueError("An image list cannot be empty.")
        return json.dumps([to_b64(item) for item in image])
    return to_b64(image)


def normalize_options(options: Any) -> tuple[dict[str, str], list[str]]:
    """Normalize list/dict options into ordered uppercase option letters."""
    if options is None:
        return {}, []
    if isinstance(options, dict):
        values = []
        for key, value in options.items():
            text = str(value).strip()
            if text:
                values.append((str(key).strip().upper(), text))
        values.sort(key=lambda item: item[0])
        normalized = {
            (key if key in OPTS else OPTS[index]): value
            for index, (key, value) in enumerate(values)
            if index < len(OPTS)
        }
        return normalized, list(normalized)
    if isinstance(options, (list, tuple)):
        normalized = {
            OPTS[index]: str(value).strip()
            for index, value in enumerate(options)
            if index < len(OPTS) and str(value).strip()
        }
        return normalized, list(normalized)
    return {}, []


def resolve_mcq_answer(
    answer: Any, options: dict[str, str], index_base: int
) -> tuple[str | None, str | None]:
    if answer is None or not str(answer).strip():
        return None, "no_answer"
    raw = str(answer).strip()
    letter_match = re.fullmatch(r"(?:option\s*)?[\(\[]?([A-H])[\)\]]?", raw, re.I)
    if letter_match:
        letter = letter_match.group(1).upper()
        return (letter, None) if letter in options else (None, "answer_out_of_range")
    if re.fullmatch(r"-?\d+", raw):
        position = int(raw) - index_base
        letters = list(options)
        if 0 <= position < len(letters):
            return letters[position], None
    normalized_answer = re.sub(r"\s+", " ", raw).strip().casefold()
    matches = [
        letter
        for letter, text in options.items()
        if re.sub(r"\s+", " ", text).strip().casefold() == normalized_answer
    ]
    if len(matches) == 1:
        return matches[0], None
    return None, "unresolved_answer"


def intermediate_to_row(idx: Any, inter: dict[str, Any], index_base: int = 0):
    """
    Convert an intermediate loader record into a VLMEvalKit TSV row.

    ``inter`` contains ``image``, ``question``, ``options``, ``answer`` and an
    optional ``category``. Non-MCQ items are retained as VQA rows whenever a
    usable text answer exists.
    """
    opt_dict, _letters = normalize_options(inter.get("options"))
    if len(opt_dict) < 2:
        answer = inter.get("answer")
        if answer is None or str(answer).strip() == "":
            return None, "no_answer"
        row = {
            "index": idx,
            "image": to_image_cell(inter["image"]),
            "question": str(inter.get("question", "")).strip(),
            "answer": str(answer).strip(),
            "answer_type": "vqa",
        }
        if inter.get("category") is not None:
            row["category"] = inter["category"]
        return row, None

    answer, error = resolve_mcq_answer(inter.get("answer"), opt_dict, index_base)
    if error:
        return None, error
    row = {
        "index": idx,
        "image": to_image_cell(inter["image"]),
        "question": str(inter.get("question", "")).strip(),
        **opt_dict,
        "answer": answer,
        "answer_type": "mcq",
    }
    if inter.get("category") is not None:
        row["category"] = inter["category"]
    return row, None


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Return loader rows in stable VLMEvalKit column order."""
    if not rows:
        raise ValueError("Cannot build an empty Track-3 dataset.")
    frame = pd.DataFrame(rows)
    head = [
        "index",
        "image",
        "question",
        *[column for column in OPTS if column in frame.columns],
        "answer",
        "answer_type",
    ]
    columns = [column for column in head if column in frame.columns]
    columns.extend(column for column in frame.columns if column not in head)
    return frame[columns]


def rows_to_tsv(rows: list[dict[str, Any]], path: str | os.PathLike[str]):
    """Order VLMEvalKit columns and write a TSV file."""
    frame = rows_to_frame(rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, destination)
    return len(frame), str(destination)