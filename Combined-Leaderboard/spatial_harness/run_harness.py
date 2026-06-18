"""
Spatial Reasoning (Task 3) evaluation harness -- VLMEvalKit-backed.

This reconstructs the analysis paper's evaluation end to end. The paper has no
released code, so we use VLMEvalKit as the data + inference backbone and add the
paper's four diagnostic conditions on top:

    standard       image + question + options
    cot            + "think step by step"
    no_image       gray image (visual-shortcut probe)
    no_image_plus  gray image + "Cannot determine from the image" option,
                   which is the correct answer (hallucination probe)

Pipeline:
  VLMEvalKit downloads each benchmark  ->  we build the 4 conditions per sample
  ->  your model answers  ->  one upload-ready response file.

Usage
-----
    # offline pipeline check (no vlmeval, no model):
    python run_harness.py --dry-run

    # real run with a VLMEvalKit model:
    python run_harness.py --model Qwen2-VL-7B-Instruct --out spatial_responses.json

    # real run with your own model module exposing model_generate(messages)->str:
    python run_harness.py --model my_model --datasets blink mmvp

Install backbone first:  pip install vlmeval   (see README).

The maintainer must run ``build_ground_truth.py`` once so the server can score
the uploaded file. Sample ids here and there are produced by the SAME backend,
so they always line up.
"""

import argparse
import json
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
COMBINED_ROOT = HARNESS_DIR.parent
sys.path.insert(0, str(COMBINED_ROOT / "backend"))

from config import SPATIAL_DATASETS, SPATIAL_MANIFEST_FILE, NO_IMAGE_PLUS_OPTION  # noqa: E402
import registry  # noqa: E402
import vlmeval_backend as backend  # noqa: E402
import conditions as cond  # noqa: E402
from adapters import load_adapter  # noqa: E402

GRAY_DIR = HARNESS_DIR / "_gray_cache"
DEFAULT_OUT = Path("spatial_responses.json")


def _dry_samples(ds_id, n=2):
    for i in range(n):
        yield {
            "sample_id": backend.make_sample_id(ds_id, i),
            "dataset_id": ds_id,
            "index": i,
            "question": f"[dry-run] {ds_id} spatial question {i}",
            "options": {"A": "left", "B": "right", "C": "above", "D": "below"},
            "answer": "A",
            "images": [],
        }


def _resolve(dataset_ids, dry_run):
    if dry_run:
        return {d: d for d in dataset_ids}, []
    resolved, missing = registry.resolve(dataset_ids)
    if missing:
        print(f"  ! skipping (not in installed VLMEvalKit registry): "
              f"{', '.join(missing)}", file=sys.stderr)
    if not resolved:
        raise SystemExit("No datasets could be resolved against VLMEvalKit. "
                         "Is `vlmeval` installed and up to date?")
    return resolved, missing


def run(dataset_ids, model_name, out_path, conditions_list, limit, dry_run):
    no_image_option = NO_IMAGE_PLUS_OPTION
    if SPATIAL_MANIFEST_FILE.exists():
        try:
            with open(SPATIAL_MANIFEST_FILE, "r", encoding="utf-8") as f:
                no_image_option = json.load(f).get("no_image_plus_option",
                                                   no_image_option)
        except Exception:  # noqa: BLE001
            pass

    resolved, _ = _resolve(dataset_ids, dry_run)
    adapter = load_adapter("example_adapter" if dry_run else model_name)

    predictions = {c: {} for c in conditions_list}
    n_samples = 0
    for ds_id, vlm_name in resolved.items():
        print(f"  {ds_id} -> {vlm_name}")
        samples = (_dry_samples(ds_id) if dry_run
                   else backend.iter_samples(ds_id, vlm_name, limit=limit))
        for sample in samples:
            n_samples += 1
            for condition in conditions_list:
                messages, _ = cond.build_messages(
                    sample, condition, no_image_option, GRAY_DIR)
                try:
                    predictions[condition][sample["sample_id"]] = str(adapter(messages))
                except Exception as exc:  # noqa: BLE001 - record + continue
                    predictions[condition][sample["sample_id"]] = ""
                    print(f"  ! {sample['sample_id']} [{condition}] failed: {exc}",
                          file=sys.stderr)

    out = {
        "schema_version": "1.0",
        "task_id": "spatial",
        "model_meta": {"name": "dry-run" if dry_run else model_name},
        "run": {"backbone": "VLMEvalKit", "decoding": "greedy", "temperature": 0},
        "predictions": predictions,
    }
    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {n_samples} samples x {len(conditions_list)} conditions "
          f"to {out_path}")
    return out_path


def _local_score(out_path, model_name):
    """Optional: score against the local ground truth using the server scorer."""
    from scoring.task_scorer import TaskScorer
    try:
        score = TaskScorer("spatial").score(out_path, model_name)
    except FileNotFoundError as exc:
        print(f"  (local scoring skipped: {exc})")
        return
    print(f"\n  Local score  accuracy={score.accuracy:.3f} "
          f"({score.correct_samples}/{score.total_samples})")
    if score.diagnostics:
        d = score.diagnostics
        print(f"  diagnostics  cot_delta={d.cot_delta} "
              f"shortcut={d.shortcut_score} halluc={d.hallucination_resistance}")


def main():
    all_ids = [d["id"] for d in SPATIAL_DATASETS]
    ap = argparse.ArgumentParser(description="Spatial Task-3 evaluation harness")
    ap.add_argument("--model", default="example_adapter",
                    help="'example_adapter', a VLMEvalKit model name, or an "
                         "importable module exposing model_generate(messages)")
    ap.add_argument("--datasets", nargs="+", default=all_ids,
                    help="subset of dataset ids (default: all 13)")
    ap.add_argument("--conditions", nargs="+", default=cond.CONDITIONS,
                    choices=cond.CONDITIONS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap samples per dataset (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="synthetic data + stub model (no vlmeval/downloads)")
    ap.add_argument("--score", action="store_true",
                    help="after writing, score locally against ground_truth.json")
    args = ap.parse_args()

    out_path = run(args.datasets, args.model, args.out,
                   args.conditions, args.limit, args.dry_run)
    if args.score:
        _local_score(out_path, "dry-run" if args.dry_run else args.model)


if __name__ == "__main__":
    main()
