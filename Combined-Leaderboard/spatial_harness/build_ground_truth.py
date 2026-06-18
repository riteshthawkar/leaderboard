"""
Maintainer script: build the SERVER-SIDE ground truth for Task 3.

Run ONCE (by whoever hosts the leaderboard) to download the spatial benchmarks
via VLMEvalKit and extract their answers into the private ground-truth file the
scorer reads. It also refreshes the public manifest with real sample counts and
writes a metadata-only ``questions.json`` (no questions/images are redistributed
-- users fetch those directly from VLMEvalKit when they run the harness).

    python build_ground_truth.py                 # all resolvable datasets
    python build_ground_truth.py --datasets blink mmvp
    python build_ground_truth.py --dry-run       # offline, synthetic, no vlmeval

Outputs (paths from backend/config.py):
  tasks/spatial/ground_truth.json   private; {sample_id: {answer, dataset, group, tags}}
  tasks/spatial/questions.json      public metadata only (no answers, no images)
  tasks/spatial/manifest.json       refreshed counts
"""

import argparse
import json
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
COMBINED_ROOT = HARNESS_DIR.parent
sys.path.insert(0, str(COMBINED_ROOT / "backend"))

from config import (  # noqa: E402
    SPATIAL_DATASETS, SPATIAL_MANIFEST_FILE, NO_IMAGE_PLUS_OPTION, TASKS,
)
import registry  # noqa: E402
import vlmeval_backend as backend  # noqa: E402

_BY_ID = {d["id"]: d for d in SPATIAL_DATASETS}
_PATHS = TASKS["spatial"]["paths"]


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


def build(dataset_ids, dry_run, limit):
    meta = _BY_ID
    if dry_run:
        resolved = {d: d for d in dataset_ids}
        missing = []
    else:
        resolved, missing = registry.resolve(dataset_ids)
        if missing:
            print(f"  ! skipping (not in installed VLMEvalKit registry): "
                  f"{', '.join(missing)}", file=sys.stderr)
        if not resolved:
            raise SystemExit("No datasets could be resolved against VLMEvalKit.")

    ground_truth, questions = {}, []
    per_dataset = {}
    for ds_id, vlm_name in resolved.items():
        info = meta.get(ds_id, {})
        name = info.get("name", ds_id)
        tags = info.get("tags", [])
        dtype = info.get("type", "")
        print(f"  loading {ds_id} -> {vlm_name} ...")

        samples = (_dry_samples(ds_id) if dry_run
                   else backend.iter_samples(ds_id, vlm_name, limit=limit))
        count = 0
        for s in samples:
            if not s.get("answer"):
                continue
            sid = s["sample_id"]
            ground_truth[sid] = {
                "answer": s["answer"],
                "dataset": name,
                "group": ds_id,
                "tags": tags,
            }
            questions.append({
                "sample_id": sid,
                "dataset": name,
                "group": ds_id,
                "type": dtype,
                "tags": tags,
                "n_options": len(s.get("options") or {}),
            })
            count += 1
        per_dataset[ds_id] = count
        print(f"    {count} samples")

    _write_outputs(ground_truth, questions, per_dataset, dataset_ids)
    return per_dataset


def _write_outputs(ground_truth, questions, per_dataset, requested_ids):
    Path(_PATHS["dir"]).mkdir(parents=True, exist_ok=True)

    with open(_PATHS["ground_truth"], "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    with open(_PATHS["questions"], "w", encoding="utf-8") as f:
        json.dump({
            "task_id": "spatial",
            "label": TASKS["spatial"]["label"],
            "note": "Metadata only. Spatial questions/images are not redistributed; "
                    "the harness fetches them from VLMEvalKit at run time.",
            "total_samples": len(questions),
            "conditions": [c for c in __import__("conditions").CONDITIONS],
            "samples": questions,
        }, f, indent=2)

    # refresh manifest counts with what we actually built
    datasets = []
    for d in SPATIAL_DATASETS:
        entry = dict(d)
        entry["actual_n"] = per_dataset.get(d["id"], 0)
        datasets.append(entry)
    with open(SPATIAL_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "task_id": "spatial",
            "version": "1.0",
            "note": "Counts reflect the locally extracted ground truth. Datasets "
                    "are downloaded via VLMEvalKit; we do not redistribute them.",
            "conditions": __import__("conditions").CONDITIONS,
            "no_image_plus_option": NO_IMAGE_PLUS_OPTION,
            "dataset_count": sum(1 for v in per_dataset.values() if v),
            "total_samples": sum(per_dataset.values()),
            "datasets": datasets,
        }, f, indent=2)

    print(f"\nWrote {len(ground_truth)} ground-truth entries.")
    print(f"  ground truth : {_PATHS['ground_truth']}")
    print(f"  questions    : {_PATHS['questions']}")
    print(f"  manifest     : {SPATIAL_MANIFEST_FILE}")


def main():
    ap = argparse.ArgumentParser(description="Build Task-3 spatial ground truth")
    ap.add_argument("--datasets", nargs="+",
                    default=[d["id"] for d in SPATIAL_DATASETS],
                    help="subset of dataset ids (default: all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap samples per dataset (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="synthetic data, no VLMEvalKit/download")
    args = ap.parse_args()
    build(args.datasets, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
