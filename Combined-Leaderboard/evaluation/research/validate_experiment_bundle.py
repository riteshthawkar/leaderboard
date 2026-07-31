#!/usr/bin/env python3
"""Validate research manifests, hashes, coverage, and answer-blind public files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.research.common import (  # noqa: E402
    SCHEMA_VERSION,
    describe_path,
    read_jsonl,
    sha256_file,
)


def assert_public_questions(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    identifiers = [str(row.get("question_id") or "") for row in rows]
    if not rows or any(not value for value in identifiers):
        raise ValueError(f"{path}: public questions are empty or have missing IDs")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path}: public questions contain duplicate IDs")
    prohibited = {"answer", "ground_truth", "correct_answer", "label"}
    for row in rows:
        leaked = prohibited & set(row)
        if leaked:
            raise ValueError(
                f"{path}: {row['question_id']} exposes answer fields {sorted(leaked)}"
            )
    return rows


def validate_causal(root: Path) -> None:
    questions = assert_public_questions(root / "questions.jsonl")
    truth = read_jsonl(root / "private_ground_truth.jsonl")
    pairs = read_jsonl(root / "pairs.jsonl")
    question_ids = {str(row["question_id"]) for row in questions}
    truth_ids = {str(row["question_id"]) for row in truth}
    if question_ids != truth_ids:
        raise ValueError("Causal question and ground-truth coverage differs")
    if len(questions) != len(pairs) * 3:
        raise ValueError("Causal bundle does not contain exactly three variants per pair")
    for row in questions:
        image = root / str(row["image"])
        if not image.is_file():
            raise FileNotFoundError(f"Missing causal image: {image}")
    truth_by_pair: dict[str, dict[str, str]] = {}
    for row in truth:
        truth_by_pair.setdefault(str(row["pair_id"]), {})[
            str(row["variant"])
        ] = str(row["answer"])
    for pair_id, answers in truth_by_pair.items():
        if set(answers) != {"base", "causal", "nuisance"}:
            raise ValueError(f"{pair_id}: variant coverage is incomplete")
        if answers["base"] != answers["nuisance"]:
            raise ValueError(f"{pair_id}: nuisance edit changed the answer")
        if answers["base"] == answers["causal"]:
            raise ValueError(f"{pair_id}: causal edit did not change the answer")


def validate_fidelity(root: Path) -> None:
    items = read_jsonl(root / "items.jsonl")
    by_track: dict[str, set[str]] = {}
    for item in items:
        track = str(item["track"])
        question_id = str(item["question_id"])
        by_track.setdefault(track, set()).add(question_id)
        for variant, expected_hash in item["variant_sha256"].items():
            path = root / track / variant / "images" / f"{question_id}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing fidelity image: {path}")
            if sha256_file(path) != expected_hash:
                raise ValueError(f"Fidelity image hash drift: {path}")
    for track, expected_ids in by_track.items():
        for variant in ("native", "downsample_50", "downsample_25"):
            questions = assert_public_questions(
                root / track / variant / "questions.jsonl"
            )
            actual_ids = {str(row["question_id"]) for row in questions}
            if actual_ids != expected_ids:
                raise ValueError(f"{track}/{variant}: paired IDs differ")


def validate_state_interventions(root: Path) -> None:
    ids_by_condition = {}
    for condition in (
        "baseline",
        "oracle_abstraction",
        "mismatched_abstraction",
    ):
        questions = assert_public_questions(root / condition / "questions.jsonl")
        ids_by_condition[condition] = {str(row["question_id"]) for row in questions}
    if len({frozenset(values) for values in ids_by_condition.values()}) != 1:
        raise ValueError("State-intervention condition IDs differ")
    for row in read_jsonl(root / "items.jsonl"):
        if row["oracle_abstraction"] == row["mismatched_abstraction"]:
            raise ValueError(f"{row['question_id']}: mismatch cue equals oracle cue")


def validate_curriculum(root: Path) -> None:
    metadata = read_jsonl(root / "metadata.jsonl")
    metadata_by_split: dict[str, list[dict[str, Any]]] = {}
    for row in metadata:
        metadata_by_split.setdefault(str(row["split"]), []).append(row)
        for field in (
            "perception_curriculum_image",
            "recognition_control_image",
        ):
            if not (root / str(row[field])).is_file():
                raise FileNotFoundError(f"Missing curriculum image: {root / row[field]}")
    for split, split_rows in metadata_by_split.items():
        target = json.loads(
            (root / f"annotations/perception_curriculum_{split}.json").read_text()
        )
        control = json.loads(
            (root / f"annotations/recognition_control_{split}.json").read_text()
        )
        if len(target) != len(control) or len(target) != len(split_rows):
            raise ValueError(f"{split}: curriculum annotation counts differ")
        for target_row, control_row in zip(target, control, strict=True):
            if target_row["conversations"] != control_row["conversations"]:
                raise ValueError(f"{split}: paired prompt or answer drift")


EXPERIMENT_VALIDATORS = {
    "causal_visual_transformations": validate_causal,
    "visual_fidelity": validate_fidelity,
    "minds_eye_abstraction_interventions": validate_state_interventions,
    "perception_curriculum_transfer": validate_curriculum,
}


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema version")
    for section in ("inputs", "outputs"):
        for name, expected in manifest.get(section, {}).items():
            actual = describe_path(Path(expected["path"]))
            for field in ("type", "sha256", "bytes", "file_count"):
                if field in expected and actual.get(field) != expected[field]:
                    raise ValueError(
                        f"{path}: {section}.{name}.{field} drifted: "
                        f"{actual.get(field)!r} != {expected[field]!r}"
                    )
    experiment = str(manifest.get("experiment") or "")
    validator = EXPERIMENT_VALIDATORS.get(experiment)
    if validator:
        validator(path.parent.resolve())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in args.manifest:
        manifest = validate_manifest(path.resolve())
        print(
            f"Validated {manifest['experiment']}: {manifest['item_count']} item(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
