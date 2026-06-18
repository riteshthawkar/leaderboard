"""
Mind's-Eye (mental imagery) evaluation harness.

Reads the public sample set (tasks/minds_eye/questions.json), builds a prompt
for every sample, queries your model through a pluggable adapter, and writes
ONE submission file in the exact format the leaderboard expects:

    {
      "task_id": "minds_eye",
      "model_meta": {"name": "..."},
      "predictions": { "standard": { "<sample_id>": "<answer>", ... } }
    }

Mind's-Eye is multiple-choice, but the options are rendered inside each image
(labelled A/B/C/D, etc.), so the model answers with the option label. Only the
`standard` condition is produced.

You implement ONE function -- `model_generate(payload) -> str` -- in an adapter
module (see `example_adapter` below). Then:

    python run_harness.py --adapter my_model --out minds_eye_responses.json

Use --dry-run to validate the pipeline without a model.

Images are local: each sample's `image_path` is resolved against the dataset
root (the workspace folder that contains `Mind-s-Eye/`). Override with
--image-root if your checkout lives elsewhere.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

# Combined-Leaderboard/minds_eye_harness/run_harness.py -> repo root is two levels up.
HARNESS_DIR = Path(__file__).resolve().parent
COMBINED_ROOT = HARNESS_DIR.parent
DEFAULT_QUESTIONS = COMBINED_ROOT / "tasks" / "minds_eye" / "questions.json"
DEFAULT_IMAGE_ROOT = COMBINED_ROOT.parent          # workspace root


def load_questions(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(sample: dict, image_root: Path) -> dict:
    """Build a prompt payload for one sample.

    Returns a dict your adapter turns into model-specific input. `image` is an
    absolute path to the image file (the options are drawn inside it).
    """
    image = None
    rel = sample.get("image_path")
    if rel:
        candidate = (image_root / rel).resolve()
        if not candidate.exists():
            print(f"  ! image not found for {sample['sample_id']}: {candidate}", file=sys.stderr)
        image = str(candidate)

    text = (sample.get("question") or "").strip()
    text += "\nThe options are shown in the image. Answer with the option label only."
    return {
        "sample_id": sample["sample_id"],
        "condition": "standard",
        "image": image,
        "text": text,
        "task": sample.get("task"),
        "capability": sample.get("capability"),
    }


def example_adapter(payload: dict) -> str:
    """Example model adapter. Replace with a real model call.

    Receives a prompt payload from `build_prompt` and returns the model's raw
    text answer (e.g. "B").
    """
    return "A"


def load_adapter(name: str):
    if name == "example_adapter":
        return example_adapter
    module = importlib.import_module(name)
    fn = getattr(module, "model_generate", None) or getattr(module, "generate", None)
    if fn is None:
        raise SystemExit(f"Adapter '{name}' must define model_generate(payload) -> str")
    return fn


def run(questions_path: Path, adapter_name: str, out_path: Path,
        image_root: Path, dry_run: bool, limit: int = 0):
    data = load_questions(questions_path)
    samples = data.get("samples", [])
    if limit:
        samples = samples[:limit]
    adapter = example_adapter if dry_run else load_adapter(adapter_name)

    predictions = {}
    for i, sample in enumerate(samples, start=1):
        payload = build_prompt(sample, image_root)
        try:
            predictions[sample["sample_id"]] = str(adapter(payload))
        except Exception as exc:  # keep going; record the failure
            predictions[sample["sample_id"]] = ""
            print(f"  ! {sample['sample_id']} failed: {exc}", file=sys.stderr)
        if i % 25 == 0:
            print(f"  ...{i}/{len(samples)}")

    out = {
        "schema_version": "1.0",
        "task_id": "minds_eye",
        "model_meta": {"name": adapter_name if not dry_run else "dry-run"},
        "run": {"decoding": "greedy", "temperature": 0},
        "predictions": {"standard": predictions},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(predictions)} predictions to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Mind's-Eye evaluation harness")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--adapter", default="example_adapter",
                    help="'example_adapter' or an importable module exposing model_generate")
    ap.add_argument("--out", type=Path, default=Path("minds_eye_responses.json"))
    ap.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT,
                    help="folder containing the Mind-s-Eye/ dataset directory")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N samples")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the pipeline with a stub model (no real inference)")
    args = ap.parse_args()
    run(args.questions, args.adapter, args.out, args.image_root, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
