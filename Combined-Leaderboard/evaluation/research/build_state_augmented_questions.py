#!/usr/bin/env python3
"""Attach validated answer-blind states to image questions."""

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

from evaluation.research.common import read_jsonl, write_jsonl  # noqa: E402
from evaluation.research.run_state_extractor_vllm import validate_state  # noqa: E402


def build(questions_path: Path, states_path: Path, output: Path) -> int:
    questions = read_jsonl(questions_path)
    states = {str(row["question_id"]): row for row in read_jsonl(states_path)}
    rows: list[dict[str, Any]] = []
    for question in questions:
        question_id = str(question["question_id"])
        if question_id not in states:
            continue
        record = states[question_id]
        if record.get("status") != "validated":
            raise ValueError(f"{question_id} does not have a validated state")
        task = str(record["task"])
        state = record["state"]
        validate_state(task, state)
        rows.append(
            {
                **question,
                "question": (
                    f"{question['question']}\n\n"
                    "Answer-blind structured visual state:\n"
                    f"{json.dumps(state, sort_keys=True)}"
                ),
            }
        )
    if not rows:
        raise ValueError("No question IDs overlap the validated state file")
    return write_jsonl(output, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = build(
        args.questions.resolve(), args.states.resolve(), args.output.resolve()
    )
    print(f"Wrote {count} image-plus-state questions to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
