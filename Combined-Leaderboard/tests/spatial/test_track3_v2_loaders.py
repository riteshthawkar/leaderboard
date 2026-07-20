import base64
from pathlib import Path

import pandas as pd
from PIL import Image

from spatial_harness.loaders.common import (
    intermediate_to_row,
    read_tsv,
    rows_to_tsv,
    to_b64,
)
from spatial_harness.loaders.prepare_custom import _split_paren_options


def _image() -> Image.Image:
    return Image.new("RGB", (2, 2), (10, 20, 30))


def test_non_mcq_record_is_retained_as_vqa():
    row, error = intermediate_to_row(
        "sample-1",
        {"image": _image(), "question": "How many?", "options": None, "answer": 3},
    )

    assert error is None
    assert row["answer"] == "3"
    assert row["answer_type"] == "vqa"
    assert base64.b64decode(row["image"])


def test_non_mcq_without_answer_is_rejected():
    row, error = intermediate_to_row(
        "sample-1",
        {"image": _image(), "question": "How many?", "options": [], "answer": ""},
    )

    assert row is None
    assert error == "no_answer"


def test_long_base64_is_not_probed_as_a_file_path():
    payload = base64.b64encode(b"embedded-image" * 100).decode("ascii")
    assert len(payload) > 255
    assert to_b64(payload) == payload


def test_tsv_reader_preserves_literal_none_answer_choice(tmp_path: Path):
    path = tmp_path / "dataset.tsv"
    pd.DataFrame(
        [
            {
                "index": 1,
                "image": "a" * 80,
                "question": "How many?",
                "A": "Two",
                "B": "None",
                "C": "Four",
                "answer": "B",
                "answer_type": "mcq",
            }
        ]
    ).to_csv(path, sep="\t", index=False)

    frame = read_tsv(path)

    assert frame.loc[0, "B"] == "None"


def test_mcq_record_has_explicit_answer_type_and_tsv_column(tmp_path: Path):
    row, error = intermediate_to_row(
        7,
        {
            "image": _image(),
            "question": "Where?",
            "options": ["left", "right", "above"],
            "answer": 2,
        },
        index_base=1,
    )
    assert error is None
    assert row["answer"] == "B"
    assert row["answer_type"] == "mcq"

    path = tmp_path / "dataset.tsv"
    count, _ = rows_to_tsv([row], path)
    frame = pd.read_csv(path, sep="\t")
    assert count == 1
    assert list(frame.columns) == [
        "index",
        "image",
        "question",
        "A",
        "B",
        "C",
        "answer",
        "answer_type",
    ]


def test_spatialbench_parser_repairs_duplicate_marker_and_instruction():
    stem, options = _split_paren_options(
        "Where is the cone? Answer with the option's letter only. "
        "(A) left (B) right (C) above (A) below "
        "Answer with the option's letter only."
    )

    assert stem == "Where is the cone?"
    assert options == {"A": "left", "B": "right", "C": "above", "D": "below"}