#!/usr/bin/env python3
"""Write an immutable, secret-free provenance record for a GPU condition run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.research.common import build_manifest, write_json  # noqa: E402


ORCHESTRATION_SCRIPTS = {
    "causal_visual_transformations": "run_single_gpu_causal_analysis.sh",
    "controlled_visual_analysis": "run_single_gpu_analysis_queue.sh",
    "qwen35_thinking_ablation": "run_qwen35_thinking_ablation.sh",
    "visual_condition_matrix": "run_visual_condition_matrix.sh",
}

ANALYSIS_PLANS = {
    "qwen35_thinking_ablation": "PRESPECIFIED_QWEN35_THINKING_PLAN.md",
}


def parse_key_value(value: str) -> tuple[str, str]:
    key, separator, item = value.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("parameters must use KEY=VALUE")
    return key, item


def parse_input(value: str) -> tuple[str, Path]:
    key, item = parse_key_value(value)
    path = Path(item).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input file does not exist: {path}")
    return key, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--extractor-model", required=True)
    parser.add_argument("--extractor-revision", required=True)
    parser.add_argument("--parameter", type=parse_key_value, action="append", default=[])
    parser.add_argument("--input", type=parse_input, action="append", default=[])
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    return parser.parse_args()


def automatic_source_inputs(
    *, experiment: str, parameters: dict[str, str]
) -> dict[str, Path]:
    """Identify the exact code and prompt contract used by an inference run."""
    inputs = {
        "source_analysis_plan": PROJECT_ROOT
        / "evaluation/research"
        / ANALYSIS_PLANS.get(
            experiment, "PRESPECIFIED_ANALYSIS_PLAN.md"
        ),
        "source_manifest_writer": Path(__file__).resolve(),
        "source_visual_pipeline": PROJECT_ROOT
        / "evaluation/common/visual_pipeline.py",
        "source_vllm_runner": PROJECT_ROOT / "evaluation/common/vllm_runner.py",
    }
    orchestration_script = ORCHESTRATION_SCRIPTS.get(experiment)
    if orchestration_script:
        inputs["source_orchestration"] = (
            PROJECT_ROOT / "evaluation/research" / orchestration_script
        )

    track = parameters.get("track")
    if experiment == "causal_visual_transformations" and not track:
        track = "minds_eye"
    prompt_mode = parameters.get("prompt_mode")
    if track and prompt_mode:
        evaluation_root = (PROJECT_ROOT / "evaluation").resolve()
        track_root = (evaluation_root / track).resolve()
        if track_root.parent != evaluation_root:
            raise ValueError(f"Invalid evaluation track in provenance: {track!r}")
        runner = track_root / "run_vllm.py"
        prompt = track_root / "prompts" / f"{prompt_mode}.txt"
        if not runner.is_file():
            raise FileNotFoundError(f"Evaluation runner does not exist: {runner}")
        if not prompt.is_file():
            raise FileNotFoundError(f"Prompt template does not exist: {prompt}")
        inputs["source_track_runner"] = runner
        inputs["source_prompt_template"] = prompt

    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Critical provenance source file(s) are missing: " + ", ".join(missing)
        )
    return inputs


def main() -> int:
    args = parse_args()
    submission = args.submission.resolve()
    diagnostics = args.diagnostics.resolve()
    if not submission.is_file() or not diagnostics.is_file():
        raise FileNotFoundError(
            "The submission and diagnostics must exist before recording the run"
        )
    parameters = dict(args.parameter)
    inputs = automatic_source_inputs(
        experiment=args.experiment,
        parameters=parameters,
    )
    supplied_inputs = dict(args.input)
    duplicate_inputs = set(inputs) & set(supplied_inputs)
    if duplicate_inputs:
        raise ValueError(
            "Input names are reserved for automatic provenance: "
            + ", ".join(sorted(duplicate_inputs))
        )
    inputs.update(supplied_inputs)
    outputs = {
        "submission": submission,
        "diagnostics": diagnostics,
    }
    quality_summary = submission.parent / "quality/summary.json"
    if quality_summary.is_file():
        outputs["quality_summary"] = quality_summary
    manifest = build_manifest(
        project_root=PROJECT_ROOT,
        experiment=args.experiment,
        parameters={
            "model": args.model,
            "model_revision": args.model_revision,
            "extractor_model": args.extractor_model,
            "extractor_revision": args.extractor_revision,
            **parameters,
        },
        inputs=inputs,
        outputs=outputs,
        item_count=sum(1 for line in submission.read_text(encoding="utf-8").splitlines() if line.strip()),
    )
    write_json(args.output.resolve(), manifest)
    print(f"Recorded run provenance at {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
