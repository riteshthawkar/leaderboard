#!/usr/bin/env python3
"""Prepare answer-blind abstraction-cue interventions for Mind's Eye."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.research.common import (  # noqa: E402
    build_manifest,
    read_jsonl,
    stable_seed,
    write_json,
    write_jsonl,
)


DEFAULT_QUESTIONS = PROJECT_ROOT / "tasks/minds_eye/questions.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/research/results/state_interventions"
DYNAMIC_OPERATIONS = (
    "rotate_back_and_forth",
    "compress_and_stretch",
    "spin_and_bounce",
    "mirror_horizontal",
    "mirror_vertical",
    "contract_x",
    "contract_y",
    "stretch_x",
    "stretch_y",
    "mirror_x",
    "mirror_y",
    "pulsate",
    "bounce",
    "wiggle",
    "swirl",
    "tilt",
    "jump",
    "rotate",
    "scale",
    "flip",
)
STATE_SCHEMAS = {
    "dynamic_isomorphism": {
        "description": (
            "A frame-by-frame, answer-blind inventory of visible shapes and their "
            "positions, sizes, orientations, and appearance. Do not infer the "
            "future frame or mention an answer option."
        ),
        "required_fields": [
            "frames",
            "visible_objects",
            "position",
            "size",
            "orientation",
            "appearance",
        ],
    },
    "slippage": {
        "description": (
            "An answer-blind inventory of each panel's objects, count, position, "
            "alignment, spacing, enclosure, fill, symmetry, topology, and border. "
            "Do not identify an outlier or mention an answer option."
        ),
        "required_fields": [
            "panels",
            "objects",
            "count",
            "position",
            "alignment",
            "spacing",
            "enclosure",
            "fill",
            "symmetry",
            "topology",
            "border",
        ],
    },
}


def humanize(value: str) -> str:
    return value.replace("_", " ")


def dynamic_pair(image_path: str) -> tuple[str, str]:
    stem = Path(image_path).stem
    match = re.fullmatch(r"pair_\d+_(.+)_grid", stem)
    if not match:
        raise ValueError(f"Unexpected dynamic image name: {image_path}")
    descriptor = match.group(1)
    matches = [
        (first, second)
        for first in DYNAMIC_OPERATIONS
        for second in DYNAMIC_OPERATIONS
        if f"{first}_{second}" == descriptor
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely parse dynamic operations from {image_path}: {matches}"
        )
    return matches[0]


def slippage_concept(image_path: str) -> str:
    stem = Path(image_path).stem
    match = re.fullmatch(r"\d+_(.+)", stem)
    if not match:
        raise ValueError(f"Unexpected slippage image name: {image_path}")
    return match.group(1)


def cue_for(row: dict[str, Any]) -> str:
    if row["task"] == "dynamic_isomorphism":
        first, second = dynamic_pair(str(row["image"]))
        return (
            "The two independent transformation families in the sequence are "
            f"{humanize(first)} and {humanize(second)}."
        )
    if row["task"] == "slippage":
        return (
            "The intended visual regularity category is "
            f"{humanize(slippage_concept(str(row['image'])))}."
        )
    raise ValueError(f"Unsupported task for automatic cue: {row['task']}")


def public_question(row: dict[str, Any], question: str) -> dict[str, Any]:
    return {
        "question_id": row["question_id"],
        "task": row["task"],
        "question": question,
        "answer_type": row["answer_type"],
        "source_subset": row["source_subset"],
        "image": row["image"],
        "image_url": row["image_url"],
    }


def prepare(questions_path: Path, output: Path, seed: int) -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(questions_path)
        if row.get("task") in STATE_SCHEMAS
    ]
    rows.sort(key=lambda row: str(row["question_id"]))
    cues = {str(row["question_id"]): cue_for(row) for row in rows}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for task in STATE_SCHEMAS:
        by_task[task] = [row for row in rows if row["task"] == task]

    mismatch_by_id: dict[str, str] = {}
    for task, task_rows in by_task.items():
        ordered = sorted(
            task_rows,
            key=lambda row: (
                stable_seed(seed, task, row["question_id"]),
                row["question_id"],
            ),
        )
        for index, row in enumerate(ordered):
            own_cue = cues[str(row["question_id"])]
            for offset in range(1, len(ordered)):
                candidate = cues[str(ordered[(index + offset) % len(ordered)]["question_id"])]
                if candidate != own_cue:
                    mismatch_by_id[str(row["question_id"])] = candidate
                    break
            else:
                raise RuntimeError(f"Task {task} does not contain distinct cues")

    conditions: dict[str, list[dict[str, Any]]] = {
        "baseline": [],
        "oracle_abstraction": [],
        "mismatched_abstraction": [],
    }
    item_map: list[dict[str, Any]] = []
    for row in rows:
        question_id = str(row["question_id"])
        conditions["baseline"].append(public_question(row, str(row["question"])))
        conditions["oracle_abstraction"].append(
            public_question(
                row,
                f"{row['question']}\n\nAdditional answer-blind cue: {cues[question_id]}",
            )
        )
        conditions["mismatched_abstraction"].append(
            public_question(
                row,
                (
                    f"{row['question']}\n\nAdditional answer-blind cue: "
                    f"{mismatch_by_id[question_id]}"
                ),
            )
        )
        item_map.append(
            {
                "question_id": question_id,
                "task": row["task"],
                "oracle_abstraction": cues[question_id],
                "mismatched_abstraction": mismatch_by_id[question_id],
            }
        )

    output_files: dict[str, Path] = {}
    for condition, question_rows in conditions.items():
        path = output / condition / "questions.jsonl"
        write_jsonl(path, question_rows)
        output_files[f"{condition}_questions"] = path
    map_path = output / "items.jsonl"
    schema_path = output / "state_schemas.json"
    write_jsonl(map_path, item_map)
    write_json(schema_path, STATE_SCHEMAS)
    output_files["item_map"] = map_path
    output_files["state_schemas"] = schema_path
    manifest = build_manifest(
        project_root=PROJECT_ROOT,
        experiment="minds_eye_abstraction_interventions",
        parameters={
            "seed": seed,
            "tasks": sorted(STATE_SCHEMAS),
            "conditions": sorted(conditions),
            "interpretation": {
                "oracle_abstraction": (
                    "A filename-derived task concept or operation cue. This is not "
                    "a complete oracle perceptual state and must not be described as one."
                ),
                "mismatched_abstraction": (
                    "A deterministic, answer-blind cue from another item in the same task."
                ),
            },
            "leakage_control": (
                "Cues contain no option letter, ground-truth answer, or answer-dependent "
                "description. Full perceptual states must be produced by the separate "
                "image-to-state runner."
            ),
        },
        inputs={"minds_eye_questions": questions_path},
        outputs=output_files,
        item_count=len(rows),
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare(args.questions.resolve(), args.output.resolve(), args.seed)
    print(
        f"Prepared {manifest['item_count']} Mind's Eye cue interventions at "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
