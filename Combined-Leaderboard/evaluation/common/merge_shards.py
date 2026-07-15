"""Merge visual inference diagnostics and create one strict upload file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .visual_pipeline import (
    EvaluationPipelineError,
    VisualTrackConfig,
    export_submission,
    load_questions,
    read_diagnostics,
)


def main(track: VisualTrackConfig, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Merge {track.label} Hugging Face shards into canonical JSONL."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="All shard diagnostics files")
    parser.add_argument("--questions", type=Path, default=track.questions_path)
    parser.add_argument("--prompt-mode", choices=("noncot", "cot"), default="noncot")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    output_path = args.out or track.default_output_path()
    try:
        questions = load_questions(args.questions, track)
        records = read_diagnostics(args.inputs)
        report = export_submission(records, questions, output_path)
    except (EvaluationPipelineError, OSError) as exc:
        print(f"Shard merge failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
