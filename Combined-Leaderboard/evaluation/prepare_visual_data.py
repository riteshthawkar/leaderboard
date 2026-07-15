"""Download and verify the public image data used by both visual benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from evaluation.common.visual_pipeline import EvaluationPipelineError, load_questions
from evaluation.do_you_see_me.config import TRACK as DYSM_TRACK
from evaluation.minds_eye.config import TRACK as MINDS_EYE_TRACK


DATASET_REPO_ID = "amolharsh/visual-intelligence-leaderboard"
DATASET_REVISION = "cc41be90e74679a9d3c9dd295834b2cee9100b9d"
DATASET_PATTERNS = (
    "dysm_2d_v1/images/**",
    "dysm_3d_v1/images/**",
    "minds_eye_fresh_v1/images/**",
)
TRACKS = (DYSM_TRACK, MINDS_EYE_TRACK)


def verify_visual_data(root: Path) -> dict[str, object]:
    root = Path(root).expanduser().resolve()
    missing: list[str] = []
    unsafe: list[str] = []
    subset_counts: Counter[str] = Counter()
    track_counts: dict[str, int] = {}

    for track in TRACKS:
        questions = load_questions(track.questions_path, track)
        track_counts[track.task_id] = len(questions)
        for question in questions:
            subset = str(question["source_subset"])
            relative_image = str(question["image"])
            candidate = (root / subset / relative_image).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                unsafe.append(str(candidate))
                continue
            if not candidate.is_file():
                missing.append(f"{question['question_id']}: {candidate}")
                continue
            subset_counts[subset] += 1

    if unsafe:
        raise EvaluationPipelineError(
            "Dataset verification found image paths outside the dataset root, including "
            + ", ".join(unsafe[:3])
            + "."
        )
    if missing:
        preview = "; ".join(missing[:5])
        raise EvaluationPipelineError(
            f"Dataset verification failed: {len(missing)} image(s) are missing. "
            f"Examples: {preview}."
        )

    return {
        "dataset_root": str(root),
        "total_questions": sum(track_counts.values()),
        "tracks": track_counts,
        "subsets": dict(sorted(subset_counts.items())),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the public MS-VISTA visual evaluation images."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=DATASET_REPO_ID)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not contact Hugging Face; only validate the existing local directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    try:
        if not args.validate_only:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise EvaluationPipelineError(
                    "huggingface-hub is required to download the visual evaluation data."
                ) from exc
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                allow_patterns=list(DATASET_PATTERNS),
                local_dir=output,
            )
        report = verify_visual_data(output)
    except (EvaluationPipelineError, OSError, RuntimeError) as exc:
        print(f"Visual dataset preparation failed: {exc}", file=sys.stderr)
        return 2

    report.update(
        {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "validated_only": args.validate_only,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
